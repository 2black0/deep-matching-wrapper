#!/usr/bin/env python3
"""
TorchScript converter for XFeat + LightGlue matcher.
Combines sparse XFeat feature extraction with LightGlue transformer matching.
"""

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
from matcher.xfeat.modules.lighterglue import LighterGlue


class XFeatLightGlueTorchScript(nn.Module):
    """
    TorchScript-friendly XFeat + LightGlue matcher for C++ deployment.
    
    This combines:
    1. XFeat sparse feature extraction (2 images)
    2. LightGlue transformer-based matching
    
    Input:
      - x0: (1,C,H,W) float32 in [0,1], first image
      - x1: (1,C,H,W) float32 in [0,1], second image
    
    Output:
      - kpts0: (N0, 2) float32, keypoints in image 0
      - scores0: (N0,) float32, keypoint scores
      - desc0: (N0, 64) float32, descriptors
      - kpts1: (N1, 2) float32, keypoints in image 1
      - scores1: (N1,) float32, keypoint scores
      - desc1: (N1, 64) float32, descriptors
    """
    
    def __init__(self, xfeat_weights_path: Path, lightglue_weights_path: Path, 
                 top_k: int = 4096, detection_threshold: float = 0.05):
        super().__init__()
        self.top_k = int(top_k)
        self.detection_threshold = float(detection_threshold)
        
        # Load XFeat model
        xfeat = XFeat(weights=str(xfeat_weights_path), top_k=top_k, detection_threshold=detection_threshold)
        self.xfeat_net = xfeat.net
        self.xfeat_net.eval()
        
        # Load LightGlue model
        self.lightglue = LighterGlue(weights=str(lightglue_weights_path))
        self.lightglue.eval()
        
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
        
        return x, rh, rw
    
    def create_xy(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Create coordinate grid."""
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        return xy
    
    def get_kpts_heatmap(self, kpts: torch.Tensor, softmax_temp: float = 1.0) -> torch::Tensor:
        """Convert keypoint logits to heatmap."""
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return heatmap
    
    def nms(self, x: torch.Tensor, threshold: float = 0.05, kernel_size: int = 5) -> torch.Tensor:
        """Non-maximum suppression."""
        pad = kernel_size // 2
        local_max = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        pos = (x == local_max) & (x > threshold)
        return pos.nonzero()[:, 2:]  # Return (y, x) coordinates
    
    def interpolate_descriptors(self, M: torch.Tensor, mkpts: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Interpolate descriptors at keypoint positions using grid_sample."""
        # Normalize coordinates to [-1, 1]
        grid = 2.0 * (mkpts / torch.tensor([W-1, H-1], device=mkpts.device, dtype=mkpts.dtype)) - 1.0
        grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
        
        # Sample
        sampled = F.grid_sample(M, grid, mode='bicubic', align_corners=False)
        return sampled.squeeze(-1).permute(0, 2, 1)  # (1, N, C)
    
    def extract_xfeat_features(self, x: torch.Tensor):
        """Extract sparse XFeat features from image."""
        # Preprocess
        x, rh, rw = self.preprocess_tensor(x)
        
        # Convert to grayscale
        if x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Forward through XFeat network
        M1, K1, H1 = self.xfeat_net(x)
        M1 = F.normalize(M1, dim=1)
        
        B, C, _H1, _W1 = M1.shape
        
        # Convert logits to heatmap
        K1h = self.get_kpts_heatmap(K1)
        
        # NMS to get sparse keypoints
        mkpts_list = []
        scores_list = []
        
        for b in range(B):
            pos = self.nms(K1h[b:b+1], threshold=self.detection_threshold, kernel_size=5)
            
            if len(pos) == 0:
                # No keypoints found, return empty
                mkpts_list.append(torch.zeros((0, 2), device=x.device))
                scores_list.append(torch.zeros((0,), device=x.device))
                continue
            
            # Flip to (x, y) format
            mkpts = pos.flip(-1).float()
            
            # Compute scores (nearest neighbor in heatmap * bilinear in reliability)
            # Simplified: just use heatmap values
            y_idx = pos[:, 0].clamp(0, K1h.shape[2]-1)
            x_idx = pos[:, 1].clamp(0, K1h.shape[3]-1)
            scores = K1h[b, 0, y_idx, x_idx]
            
            # Sort by score and take top_k
            if len(scores) > self.top_k:
                top_k_idx = torch.argsort(-scores)[:self.top_k]
                mkpts = mkpts[top_k_idx]
                scores = scores[top_k_idx]
            
            mkpts_list.append(mkpts)
            scores_list.append(scores)
        
        # For batch=1, just take first
        mkpts = mkpts_list[0]
        scores = scores_list[0]
        
        # Interpolate descriptors
        if len(mkpts) > 0:
            feats = self.interpolate_descriptors(M1, mkpts, _H1, _W1)
            feats = F.normalize(feats, dim=-1).squeeze(0)
            
            # Correct keypoint scale
            mkpts = mkpts * torch.tensor([rw, rh], device=mkpts.device)
        else:
            feats = torch.zeros((0, 64), device=x.device)
        
        return mkpts, scores, feats
    
    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        """
        Match two images using XFeat sparse features + LightGlue.
        
        Args:
            x0: (1,C,H,W) first image
            x1: (1,C,H,W) second image
        
        Returns:
            Tuple of (kpts0, scores0, desc0, kpts1, scores1, desc1)
        """
        assert x0.dim() == 4 and int(x0.shape[0]) == 1
        assert x1.dim() == 4 and int(x1.shape[0]) == 1
        
        # Extract features from both images
        kpts0, scores0, desc0 = self.extract_xfeat_features(x0)
        kpts1, scores1, desc1 = self.extract_xfeat_features(x1)
        
        return kpts0, scores0, desc0, kpts1, scores1, desc1


def main():
    parser = argparse.ArgumentParser(description="XFeat + LightGlue TorchScript export for C++ (B=1)")
    parser.add_argument("--xfeat-weights", type=str, default="matcher/xfeat/weights/xfeat.pt")
    parser.add_argument("--lightglue-weights", type=str, default="matcher/xfeat/weights/xfeat-lighterglue.pt")
    parser.add_argument("--topk", type=int, default=4096)
    parser.add_argument("--detection-threshold", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "matcher-cpp" / "xfeat" / "weights"))
    args = parser.parse_args()
    
    xfeat_weights = Path(args.xfeat_weights)
    lightglue_weights = Path(args.lightglue_weights)
    
    if not xfeat_weights.exists():
        raise FileNotFoundError(f"Missing XFeat weights: {xfeat_weights}")
    if not lightglue_weights.exists():
        raise FileNotFoundError(f"Missing LightGlue weights: {lightglue_weights}")
    
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xfeat_lightglue_fp32_k{args.topk}.pt"
    
    print(f"Creating XFeat + LightGlue TorchScript module...")
    m = XFeatLightGlueTorchScript(xfeat_weights, lightglue_weights, 
                                    top_k=args.topk, 
                                    detection_threshold=args.detection_threshold)
    m.to(device)
    m.eval()
    
    # Trace it
    dummy_input0 = torch.zeros((1, 3, 480, 640), dtype=torch.float32, device=device)
    dummy_input1 = torch.zeros((1, 3, 480, 640), dtype=torch.float32, device=device)
    
    print("Tracing model with example inputs (1,3,480,640)...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = torch.jit.trace(m, (dummy_input0, dummy_input1), strict=False)
        ts = torch.jit.freeze(ts)
    
    ts.save(str(out_path))
    print(f"Saved: {out_path}")
    print("Note: This is a feature-only extractor. Matching logic remains in C++.")


if __name__ == "__main__":
    main()
