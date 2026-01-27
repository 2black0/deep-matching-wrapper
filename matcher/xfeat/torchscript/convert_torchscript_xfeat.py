import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from matcher.xfeat.modules.xfeat import XFeat
from matcher.xfeat.modules.interpolator import InterpolateSparse2d


class XFeatTorchScript(nn.Module):
    """TorchScript-friendly XFeat sparse feature extractor (B=1).
    
    This wraps XFeat's detectAndCompute functionality for C++ deployment.
    
    Input:
      - x: (1,C,H,W) float32 in [0,1], grayscale or RGB image
    
    Output:
      - keypoints: (N,2) float32 in original image coordinates (x,y)
      - scores: (N,) float32
      - descriptors: (N,64) float32
    """
    
    def __init__(self, weights_path: Path, top_k: int = 4096, detection_threshold: float = 0.05):
        super().__init__()
        self.top_k = int(top_k)
        self.detection_threshold = float(detection_threshold)
        
        # Load XFeat model (force CPU)
        xfeat_full = XFeat(weights=str(weights_path), top_k=top_k, detection_threshold=detection_threshold)
        self.net = xfeat_full.net.cpu()  # Force CPU for TorchScript export
        self.net.eval()
        
        # Interpolator for descriptor sampling
        self.interpolator = InterpolateSparse2d('bicubic')
        
    def preprocess_tensor(self, x: torch.Tensor):
        """Guarantee that image is divisible by 32 to avoid aliasing artifacts."""
        H = int(x.shape[-2])
        W = int(x.shape[-1])
        _H = (H // 32) * 32
        _W = (W // 32) * 32
        rh = float(H) / float(_H)
        rw = float(W) / float(_W)
        
        if H != _H or W != _W:
            x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        
        return x, rh, rw, _H, _W
    
    def get_kpts_heatmap(self, kpts: torch.Tensor, softmax_temp: float = 1.0) -> torch.Tensor:
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return heatmap
    
    def NMS(self, x: torch.Tensor, threshold: float = 0.05, kernel_size: int = 5) -> torch.Tensor:
        """Non-Maximum Suppression to extract keypoints."""
        B, _, H, W = x.shape
        pad = kernel_size // 2
        local_max = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        pos = (x == local_max) & (x > threshold)
        
        # Extract positions
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]  # (y,x) -> (x,y)
        
        pad_val = max([len(p) for p in pos_batched])
        if pad_val == 0:
            return x.new_zeros((B, 0, 2), dtype=torch.long)
        
        # Pad kpts and build (B, N, 2) tensor
        result = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)
        for b in range(len(pos_batched)):
            if len(pos_batched[b]) > 0:
                result[b, :len(pos_batched[b]), :] = pos_batched[b]
        
        return result
    
    def forward(self, x: torch.Tensor):
        assert x.dim() == 4 and int(x.shape[0]) == 1
        
        # Convert to grayscale
        x = x.mean(dim=1, keepdim=True)
        
        # Preprocess (resize to multiple of 32)
        x, rh, rw, _H, _W = self.preprocess_tensor(x)
        
        B, _, _, _ = x.shape
        
        # Forward through network
        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)
        
        # Convert logits to heatmap and extract keypoints
        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=self.detection_threshold, kernel_size=5)
        
        # Check if we have any keypoints
        if mkpts.numel() == 0 or mkpts.shape[1] == 0:
            empty_k = x.new_zeros((0, 2), dtype=torch.float32)
            empty_s = x.new_zeros((0,), dtype=torch.float32)
            empty_d = x.new_zeros((0, 64), dtype=torch.float32)
            return empty_k, empty_s, empty_d
        
        # Compute reliability scores
        # Nearest interpolation for K1h
        grid_nearest = 2.0 * (mkpts[0].float() / torch.tensor([_W-1, _H-1], device=mkpts.device, dtype=torch.float32)) - 1.0
        grid_nearest = grid_nearest.unsqueeze(0).unsqueeze(-2)
        score_k = F.grid_sample(K1h, grid_nearest, mode='nearest', align_corners=False)
        score_k = score_k.permute(0, 2, 3, 1).squeeze(-2).squeeze(0)[:, 0]
        
        # Bilinear interpolation for H1
        grid_bilinear = 2.0 * (mkpts[0].float() / torch.tensor([_W-1, _H-1], device=mkpts.device, dtype=torch.float32)) - 1.0
        grid_bilinear = grid_bilinear.unsqueeze(0).unsqueeze(-2)
        score_h = F.grid_sample(H1, grid_bilinear, mode='bilinear', align_corners=False)
        score_h = score_h.permute(0, 2, 3, 1).squeeze(-2).squeeze(0)[:, 0]
        
        scores = score_k * score_h
        
        # Mark invalid keypoints (all zeros)
        valid_mask = ~torch.all(mkpts[0] == 0, dim=-1)
        scores[~valid_mask] = -1.0
        
        # Select top-k features
        valid_scores = scores[valid_mask]
        valid_kpts = mkpts[0][valid_mask].float()
        
        if valid_scores.numel() == 0:
            empty_k = x.new_zeros((0, 2), dtype=torch.float32)
            empty_s = x.new_zeros((0,), dtype=torch.float32)
            empty_d = x.new_zeros((0, 64), dtype=torch.float32)
            return empty_k, empty_s, empty_d
        
        if valid_scores.numel() > self.top_k:
            _, idxs = torch.topk(valid_scores, k=self.top_k, sorted=True)
            valid_kpts = valid_kpts[idxs]
            valid_scores = valid_scores[idxs]
        
        # Interpolate descriptors at keypoint positions
        grid_desc = 2.0 * (valid_kpts / torch.tensor([_W-1, _H-1], device=valid_kpts.device, dtype=valid_kpts.dtype)) - 1.0
        grid_desc = grid_desc.unsqueeze(0).unsqueeze(-2)
        feats = F.grid_sample(M1, grid_desc, mode='bicubic', align_corners=False)
        feats = feats.permute(0, 2, 3, 1).squeeze(0).squeeze(1)
        
        # L2-Normalize descriptors
        feats = F.normalize(feats, dim=-1)
        
        # Correct keypoint scale back to original image size
        scale = torch.tensor([rw, rh], device=valid_kpts.device, dtype=valid_kpts.dtype)
        valid_kpts = valid_kpts * scale
        
        # Filter out keypoints with non-positive scores
        final_mask = valid_scores > 0
        final_kpts = valid_kpts[final_mask]
        final_scores = valid_scores[final_mask]
        final_feats = feats[final_mask]
        
        return final_kpts, final_scores, final_feats


def main():
    p = argparse.ArgumentParser(description="XFeat (sparse) TorchScript export for matcher-cpp (B=1)")
    p.add_argument("--weights", type=str, default=str(ROOT / "matcher-cpp" / "xfeat" / "weights" / "xfeat.pt"))
    p.add_argument("--topk", type=int, default=4096)
    p.add_argument("--detection-threshold", type=float, default=0.05)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "matcher-cpp" / "xfeat" / "weights"),
    )
    args = p.parse_args()
    
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing XFeat weights: {weights_path}")
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xfeat_fp32_k{int(args.topk)}.pt"
    
    m = XFeatTorchScript(weights_path, top_k=args.topk, detection_threshold=args.detection_threshold)
    m.eval()
    
    print("Scripting model for TorchScript export...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = torch.jit.script(m)
        ts = torch.jit.freeze(ts)
    
    ts.save(str(out_path))
    print(f"Saved: {out_path}")
    print("Note: This model is scripted for variable input sizes.")


if __name__ == "__main__":
    main()
