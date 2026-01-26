import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from matcher.liftfeat.modules.model import LiftFeatSPModel
from matcher.liftfeat.modules.liftfeat_wrapper import featureboost_config, load_model


class LiftFeatTorchScript(nn.Module):
    """TorchScript-friendly LiftFeat extractor (B=1).

    Mirrors the essential behavior of `matcher/liftfeat/modules/liftfeat_wrapper.py:LiftFeat.extract`.

    Input:
      - x: (1,3,H,W) float32 in [0,1]

    Output:
      - keypoints: (N,2) float32 in original image coords (x,y)
      - scores: (N,) float32
      - descriptors: (N,64) float32
    """

    def __init__(self, weights_path: Path, detect_threshold: float = 0.005, top_k: int = 4096):
        super().__init__()
        self.detect_threshold = float(detect_threshold)
        self.top_k = int(top_k)

        self.net = LiftFeatSPModel(featureboost_config)
        self.net = load_model(self.net, str(weights_path))
        self.net.eval()

        # NMS uses 5x5 max pool (radius=2)
        self.mp = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    @staticmethod
    def _pad_to_32(x: torch.Tensor):
        # x: (1,3,H,W)
        H = int(x.shape[-2])
        W = int(x.shape[-1])
        _H = ((H + 31) // 32) * 32
        _W = ((W + 31) // 32) * 32
        pad_h = _H - H
        pad_w = _W - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        return x, H, W, _H, _W

    @staticmethod
    def _logits_to_heatmap(kpt_logits: torch.Tensor) -> torch.Tensor:
        # kpt_logits: (1,65,h,w)
        scores_raw = torch.softmax(kpt_logits, dim=1)[:, :64]
        B, _, h_feat, w_feat = scores_raw.shape
        heat = (
            scores_raw.permute(0, 2, 3, 1)
            .reshape(B, h_feat, w_feat, 8, 8)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, 1, h_feat * 8, w_feat * 8)
        )
        return heat

    @staticmethod
    def _sample_bicubic(x: torch.Tensor, kpts_xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (1,C,h,w)  kpts_xy: (N,2) float (x,y)
        # normalize to [-1,1] using align_corners=False mapping in InterpolateSparse2d
        denom = torch.tensor([float(W - 1), float(H - 1)], device=kpts_xy.device, dtype=kpts_xy.dtype)
        grid = 2.0 * (kpts_xy / denom) - 1.0
        grid = grid.view(1, -1, 1, 2).to(x.dtype)
        samp = F.grid_sample(x, grid, mode="bicubic", align_corners=False)
        # (1,C,N,1) -> (N,C)
        return samp.permute(0, 2, 3, 1).squeeze(0).squeeze(1)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4 and int(x.shape[0]) == 1
        x, H0, W0, Hp, Wp = self._pad_to_32(x)

        # Grayscale (matches LiftFeatSPModel.forward1)
        x_gray = x.mean(dim=1, keepdim=True)

        des_map, kpt_logits, d_feats = self.net.forward1(x_gray)
        refined_v = self.net.forward2(des_map, kpt_logits, d_feats)
        refined_map = refined_v.view(1, des_map.shape[2], des_map.shape[3], -1).permute(0, 3, 1, 2)
        refined_map = F.normalize(refined_map, p=2.0, dim=1)

        heat = self._logits_to_heatmap(kpt_logits)

        # NMS peaks
        local_max = self.mp(heat)
        peaks = (heat == local_max) & (heat > self.detect_threshold)
        idx = torch.nonzero(peaks[0, 0])  # (N,2) as (y,x)
        if idx.numel() == 0:
            empty_k = x.new_zeros((0, 2), dtype=torch.float32)
            empty_s = x.new_zeros((0,), dtype=torch.float32)
            empty_d = x.new_zeros((0, 64), dtype=torch.float32)
            return empty_k, empty_s, empty_d

        kpts = idx[:, [1, 0]].to(torch.float32)  # (x,y)

        # Remove padded region
        mask = (kpts[:, 0] < float(W0)) & (kpts[:, 1] < float(H0))
        kpts = kpts[mask]
        if kpts.numel() == 0:
            empty_k = x.new_zeros((0, 2), dtype=torch.float32)
            empty_s = x.new_zeros((0,), dtype=torch.float32)
            empty_d = x.new_zeros((0, 64), dtype=torch.float32)
            return empty_k, empty_s, empty_d

        # Sample scores from heatmap at full resolution
        scores = self._sample_bicubic(heat, kpts, Hp, Wp)[:, 0].to(torch.float32)

        # Sample descriptors (map is at H/8,W/8 but kpts are full-res)
        # Equivalent to the wrapper's sampler: it uses H,W = padded full-res.
        # refined_map is (1,64,Hp/8,Wp/8)
        desc = self._sample_bicubic(refined_map, kpts, Hp, Wp).to(torch.float32)
        desc = F.normalize(desc, p=2.0, dim=1)

        if self.top_k > 0 and int(scores.numel()) > self.top_k:
            _, sel = torch.topk(scores, k=self.top_k, dim=0, sorted=True)
            kpts = kpts.index_select(0, sel)
            scores = scores.index_select(0, sel)
            desc = desc.index_select(0, sel)

        return kpts, scores, desc


def main():
    p = argparse.ArgumentParser(description="LiftFeat TorchScript export for matcher-cpp (B=1)")
    p.add_argument("--weights", type=str, default="matcher/liftfeat/weights/LiftFeat.pth")
    p.add_argument("--detect-threshold", type=float, default=0.005)
    p.add_argument("--topk", type=int, default=4096)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "matcher-cpp" / "liftfeat" / "weights"),
    )
    args = p.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing LiftFeat weights: {weights_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"liftfeat_fp32_k{int(args.topk)}.pt"

    m = LiftFeatTorchScript(weights_path, detect_threshold=args.detect_threshold, top_k=args.topk).eval()
    ts = torch.jit.script(m)
    ts = torch.jit.freeze(ts)
    ts.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
