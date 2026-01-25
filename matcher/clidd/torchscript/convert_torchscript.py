import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from matcher.clidd.modules.model import Model
from matcher.clidd.modules.clidd_wrapper import CLIDD


class CLIDDTorchScript(nn.Module):
    """TorchScript-friendly CLIDD forward for C++ (B=1).

    Mirrors `matcher/clidd/modules/clidd_wrapper.py:CLIDD.forward` but returns
    plain tensors:
      - keypoints: (N, 2) float32 in original image coordinates
      - scores: (N,) float32
      - descriptors: (N, D) float32
    """

    def __init__(self, cfg_name: str, weights_path: Path, top_k: int = 2048, radius: int = 2, score_thresh: float = -5.0, border: int = 4):
        super().__init__()
        self.cfg_name = cfg_name
        self.top_k = int(top_k)
        self.radius = int(radius)
        self.score_thresh = float(score_thresh)
        self.border = int(border)

        cfg_params = CLIDD.cfgs[cfg_name]
        self.model = Model(**cfg_params)
        self.model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
        self.model.eval()

        self.use_nms = self.radius > 0
        r = self.radius if self.radius > 0 else 0
        self.mp = nn.MaxPool2d(r * 2 + 1, 1, r)

    def forward(self, x: torch.Tensor):
        # x: (1,C,H,W)
        B, C, oH, oW = x.shape
        assert B == 1

        nH = (oH // 32) * 32
        nW = (oW // 32) * 32
        size = torch.tensor([float(nW), float(nH)], dtype=torch.float32, device=x.device)
        scale = torch.tensor([float(oW) / float(nW), float(oH) / float(nH)], dtype=torch.float32, device=x.device)

        if oW != nW or oH != nH:
            x = F.interpolate(x, (nH, nW), mode="bilinear", align_corners=True)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        raw_desc, raw_detect = self.model(x)

        if self.use_nms:
            detect1 = raw_detect == self.mp(raw_detect)
        else:
            detect1 = torch.ones_like(raw_detect, dtype=torch.bool)

        # Border suppression
        b = self.border
        if b > 0:
            detect1[..., :, :b] = False
            detect1[..., :, -b:] = False
            detect1[..., :b, :] = False
            detect1[..., -b:, :] = False

        detect2 = raw_detect > self.score_thresh
        detect = torch.logical_and(detect1, detect2)[:, 0]  # (1,H,W)

        # Build coordinate grid without meshgrid() for TorchScript stability.
        Hh = int(detect.shape[-2])
        Ww = int(detect.shape[-1])
        ys = torch.arange(Hh, dtype=torch.float32, device=x.device).view(Hh, 1).expand(Hh, Ww)
        xs = torch.arange(Ww, dtype=torch.float32, device=x.device).view(1, Ww).expand(Hh, Ww)
        ind = torch.stack([xs, ys], dim=-1)  # (H,W,2)

        kpts = ind[detect[0]]
        scores = raw_detect[0, 0, detect[0]].to(torch.float32)

        if kpts.numel() == 0:
            empty_desc = raw_detect.new_zeros((0, 0), dtype=torch.float32)
            return kpts.to(torch.float32), scores, empty_desc

        if self.top_k > 0:
            k = min(self.top_k, int(scores.shape[0]))
            score_top, idx = scores.topk(k)
            scores = score_top
            kpts = kpts[idx]

        # Sample descriptors: raw_desc is a tuple(x1,x2,x3) with batch dim.
        d0 = raw_desc[0][0:1]
        d1 = raw_desc[1][0:1]
        d2 = raw_desc[2][0:1]
        norm_kpts = (kpts + 0.5) / size * 2.0 - 1.0
        norm_kpts = norm_kpts.view(1, -1, 1, 2).to(raw_detect.dtype)
        desc = self.model.sample([d0, d1, d2], norm_kpts, False)[0].to(torch.float32)

        kpts = (kpts * scale).to(torch.float32)
        return kpts, scores, desc


def main():
    p = argparse.ArgumentParser(description="CLIDD TorchScript export for matcher-cpp (B=1)")
    p.add_argument("--weights", required=True, help="Model name like A48, U128")
    p.add_argument("--dtype", choices=["FP32"], default="FP32")
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--score-thresh", type=float, default=-5.0)
    p.add_argument("--border", type=int, default=4)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "matcher-cpp" / "clidd" / "weights"),
    )
    args = p.parse_args()

    cfg = args.weights.upper()
    weights_file = ROOT / "matcher" / "clidd" / "weights" / f"{cfg}.pth"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"clidd_{cfg.lower()}_fp32_k{int(args.topk)}.pt"

    m = CLIDDTorchScript(cfg, weights_file, top_k=args.topk, radius=args.radius, score_thresh=args.score_thresh, border=args.border)
    m.eval()

    # Script on CPU for portability.
    ts = torch.jit.script(m)
    ts = torch.jit.freeze(ts)
    ts.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
