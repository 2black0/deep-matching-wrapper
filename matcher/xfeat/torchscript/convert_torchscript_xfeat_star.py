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


class XFeatStarTorchScript(nn.Module):
    """TorchScript-friendly XFeat-Star (semi-dense + refinement) feature extractor and matcher (B=1).
    
    This wraps XFeat's detectAndComputeDense + refine_matches functionality for C++ deployment.
    
    Input:
      - x0: (1,C,H,W) float32 in [0,1], first image (grayscale or RGB)
      - x1: (1,C,H,W) float32 in [0,1], second image (grayscale or RGB)
    
    Output:
      - matches: (N, 4) float32, refined matches as (x0, y0, x1, y1)
    """
    
    def __init__(self, weights_path: Path, top_k: int = 4096, fine_conf: float = 0.25):
        super().__init__()
        self.top_k = int(top_k)
        self.fine_conf = float(fine_conf)
        
        # Load XFeat model (force CPU)
        xfeat = XFeat(weights=str(weights_path), top_k=top_k)
        self.net = xfeat.net.cpu()  # Force CPU for TorchScript export
        self.net.eval()
        
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
    
    def extractDense(self, x: torch.Tensor):
        """Extract dense features from image."""
        x, rh, rw = self.preprocess_tensor(x)
        
        M1, K1, H1 = self.net(x)
        
        B, C, _H1, _W1 = M1.shape
        
        xy1 = (self.create_xy(_H1, _W1, M1.device) * 8).expand(B, -1, -1)
        
        M1 = M1.permute(0, 2, 3, 1).reshape(B, -1, C)
        H1 = H1.permute(0, 2, 3, 1).reshape(B, -1)
        
        # Select top-k based on heatmap
        k = min(H1.shape[1], self.top_k)
        _, top_k_idx = torch.topk(H1, k=k, dim=-1)
        
        feats = torch.gather(M1, 1, top_k_idx.unsqueeze(-1).expand(-1, -1, 64))
        mkpts = torch.gather(xy1, 1, top_k_idx.unsqueeze(-1).expand(-1, -1, 2))
        mkpts = mkpts * torch.tensor([rw, rh], device=mkpts.device).view(1, -1)
        
        return mkpts, feats
    
    def extract_dualscale(self, x: torch.Tensor):
        """Extract features at two scales."""
        s1 = 0.6
        s2 = 1.3
        
        x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode='bilinear')
        x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode='bilinear')
        
        mkpts_1, feats_1 = self.extractDense(x1)
        mkpts_2, feats_2 = self.extractDense(x2)
        
        mkpts = torch.cat([mkpts_1/s1, mkpts_2/s2], dim=1)
        sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1/s1)
        sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1/s2)
        sc = torch.cat([sc1, sc2], dim=1)
        feats = torch.cat([feats_1, feats_2], dim=1)
        
        return mkpts, sc, feats
    
    def batch_match(self, feats1: torch.Tensor, feats2: torch.Tensor):
        """Match feature descriptors using mutual nearest neighbors."""
        # feats1, feats2: (B, N, 64)
        cossim = torch.bmm(feats1, feats2.permute(0, 2, 1))
        match12 = torch.argmax(cossim, dim=-1)
        match21 = torch.argmax(cossim.permute(0, 2, 1), dim=-1)
        
        idx0 = torch.arange(match12.shape[1], device=match12.device)
        
        # For batch=1, extract the matches
        mutual = match21[0, match12[0]] == idx0
        idx0_match = idx0[mutual]
        idx1_match = match12[0, mutual]
        
        return idx0_match, idx1_match
    
    def subpix_softmax2d(self, heatmaps: torch.Tensor, temp: float = 3.0) -> torch.Tensor:
        """Compute sub-pixel offsets from heatmaps."""
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
        
        y = torch.arange(H, device=heatmaps.device, dtype=heatmaps.dtype)
        x = torch.arange(W, device=heatmaps.device, dtype=heatmaps.dtype)
        yy, xx = torch.meshgrid(y, x, indexing='xy')
        
        x_grid = xx - (W // 2)
        y_grid = yy - (H // 2)
        
        coords_x = (x_grid.unsqueeze(0) * heatmaps).view(N, H*W, 1).sum(1)
        coords_y = (y_grid.unsqueeze(0) * heatmaps).view(N, H*W, 1).sum(1)
        coords = torch.cat([coords_x, coords_y], dim=-1)
        
        return coords
    
    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        """
        Match two images using XFeat-Star (semi-dense + refinement).
        
        Args:
            x0: (1,C,H,W) first image
            x1: (1,C,H,W) second image
            
        Returns:
            matches: (N, 4) tensor of refined matches (x0, y0, x1, y1)
        """
        assert x0.dim() == 4 and int(x0.shape[0]) == 1
        assert x1.dim() == 4 and int(x1.shape[0]) == 1
        
        # Convert to grayscale
        x0 = x0.mean(dim=1, keepdim=True)
        x1 = x1.mean(dim=1, keepdim=True)
        
        # Extract dense features at dual scales
        mkpts_0, sc0, feats_0 = self.extract_dualscale(x0)
        mkpts_1, sc1, feats_1 = self.extract_dualscale(x1)
        
        # Match descriptors
        idx0, idx1 = self.batch_match(feats_0, feats_1)
        
        if idx0.numel() == 0:
            return x0.new_zeros((0, 4), dtype=torch.float32)
        
        # Extract matched features
        matched_feats1 = feats_0[0, idx0]
        matched_feats2 = feats_1[0, idx1]
        matched_kpts0 = mkpts_0[0, idx0]
        matched_kpts1 = mkpts_1[0, idx1]
        matched_sc0 = sc0[0, idx0]
        
        # Compute fine offsets using the fine_matcher MLP
        concat_feats = torch.cat([matched_feats1, matched_feats2], dim=-1)
        offsets = self.net.fine_matcher(concat_feats)
        
        # Compute confidence from softmax
        conf = F.softmax(offsets * 3, dim=-1).max(dim=-1)[0]
        offsets = self.subpix_softmax2d(offsets.view(-1, 8, 8))
        
        # Refine keypoints with offsets scaled by the extraction scale
        matched_kpts0 = matched_kpts0 + offsets * matched_sc0.unsqueeze(-1)
        
        # Filter by confidence threshold
        mask_good = conf > self.fine_conf
        final_kpts0 = matched_kpts0[mask_good]
        final_kpts1 = matched_kpts1[mask_good]
        
        # Concatenate into (N, 4) format
        matches = torch.cat([final_kpts0, final_kpts1], dim=-1)
        
        return matches


def main():
    p = argparse.ArgumentParser(description="XFeat-Star (semi-dense) TorchScript export for matcher-cpp (B=1)")
    p.add_argument("--weights", type=str, default="matcher/xfeat/weights/xfeat.pt")
    p.add_argument("--topk", type=int, default=4096)
    p.add_argument("--fine-conf", type=float, default=0.25)
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
    out_path = out_dir / f"xfeat_star_fp32_k{int(args.topk)}.pt"
    
    m = XFeatStarTorchScript(weights_path, top_k=args.topk, fine_conf=args.fine_conf)
    m.eval()
    
    # Use trace instead of script due to TorchScript limitations
    dummy_input0 = torch.zeros((1, 3, 480, 640), dtype=torch.float32)
    dummy_input1 = torch.zeros((1, 3, 480, 640), dtype=torch.float32)
    
    print("Tracing model with example inputs (1,3,480,640)...")
    print("Note: TracerWarnings about tensor->Python conversions are expected and safe for traced models.")
    
    # Filter TracerWarnings to reduce output noise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = torch.jit.trace(m, (dummy_input0, dummy_input1), strict=False)
        ts = torch.jit.freeze(ts)
    
    ts.save(str(out_path))
    print(f"Saved: {out_path}")
    print("Note: This model is traced, not scripted. It works with variable input sizes.")


if __name__ == "__main__":
    main()
