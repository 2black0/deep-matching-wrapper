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


class XFeatLightGlueTorchScript(nn.Module):
    """TorchScript-friendly XFeat + LighterGlue matcher (B=1).
    
    This wraps XFeat's detectAndCompute + LighterGlue matching for C++ deployment.
    
    Input:
      - x0: (1,C,H0,W0) float32 in [0,1], first image (grayscale or RGB)
      - x1: (1,C,H1,W1) float32 in [0,1], second image (grayscale or RGB)
    
    Output:
      - mkpts0: (N,2) float32, matched keypoints from image 0
      - mkpts1: (N,2) float32, matched keypoints from image 1
      - all_kpts0: (M0,2) float32, all keypoints from image 0
      - all_kpts1: (M1,2) float32, all keypoints from image 1
      - all_desc0: (M0,64) float32, all descriptors from image 0
      - all_desc1: (M1,64) float32, all descriptors from image 1
    """
    
    def __init__(self, xfeat_weights_path: Path, lightglue_weights_path: Path, 
                 top_k: int = 4096, detection_threshold: float = 0.05, min_conf: float = 0.1):
        super().__init__()
        self.top_k = int(top_k)
        self.detection_threshold = float(detection_threshold)
        self.min_conf = float(min_conf)
        
        # Load XFeat model
        xfeat = XFeat(weights=str(xfeat_weights_path), top_k=top_k, detection_threshold=detection_threshold)
        self.xfeat_net = xfeat.net
        self.xfeat_net.eval()
        
        # Load LighterGlue model
        # Note: LighterGlue uses kornia's LightGlue, which may be complex for TorchScript
        # We'll need to simplify or wrap it appropriately
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            from matcher.xfeat.modules.lighterglue import LighterGlue
            self.lightglue = LighterGlue(weights=str(lightglue_weights_path))
            self.lightglue.net.eval()
    
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
        
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]
        
        pad_val = max([len(p) for p in pos_batched])
        if pad_val == 0:
            return x.new_zeros((B, 0, 2), dtype=torch.long)
        
        result = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)
        for b in range(len(pos_batched)):
            if len(pos_batched[b]) > 0:
                result[b, :len(pos_batched[b]), :] = pos_batched[b]
        
        return result
    
    def detectAndCompute(self, x: torch.Tensor):
        """Extract sparse keypoints and descriptors from an image."""
        # Convert to grayscale
        x = x.mean(dim=1, keepdim=True)
        
        # Preprocess
        x, rh, rw, _H, _W = self.preprocess_tensor(x)
        
        # Forward
        M1, K1, H1 = self.xfeat_net(x)
        M1 = F.normalize(M1, dim=1)
        
        # Get keypoints
        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=self.detection_threshold, kernel_size=5)
        
        if mkpts.numel() == 0 or mkpts.shape[1] == 0:
            empty_k = x.new_zeros((0, 2), dtype=torch.float32)
            empty_s = x.new_zeros((0,), dtype=torch.float32)
            empty_d = x.new_zeros((0, 64), dtype=torch.float32)
            return empty_k, empty_s, empty_d
        
        # Compute scores
        grid_nearest = 2.0 * (mkpts[0].float() / torch.tensor([_W-1, _H-1], device=mkpts.device, dtype=torch.float32)) - 1.0
        grid_nearest = grid_nearest.unsqueeze(0).unsqueeze(-2)
        score_k = F.grid_sample(K1h, grid_nearest, mode='nearest', align_corners=False)
        score_k = score_k.permute(0, 2, 3, 1).squeeze(-2).squeeze(0)[:, 0]
        
        grid_bilinear = 2.0 * (mkpts[0].float() / torch.tensor([_W-1, _H-1], device=mkpts.device, dtype=torch.float32)) - 1.0
        grid_bilinear = grid_bilinear.unsqueeze(0).unsqueeze(-2)
        score_h = F.grid_sample(H1, grid_bilinear, mode='bilinear', align_corners=False)
        score_h = score_h.permute(0, 2, 3, 1).squeeze(-2).squeeze(0)[:, 0]
        
        scores = score_k * score_h
        
        valid_mask = ~torch.all(mkpts[0] == 0, dim=-1)
        scores[~valid_mask] = -1.0
        
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
        
        # Interpolate descriptors
        grid_desc = 2.0 * (valid_kpts / torch.tensor([_W-1, _H-1], device=valid_kpts.device, dtype=valid_kpts.dtype)) - 1.0
        grid_desc = grid_desc.unsqueeze(0).unsqueeze(-2)
        feats = F.grid_sample(M1, grid_desc, mode='bicubic', align_corners=False)
        feats = feats.permute(0, 2, 3, 1).squeeze(0).squeeze(1)
        feats = F.normalize(feats, dim=-1)
        
        # Scale keypoints back to original size
        scale = torch.tensor([rw, rh], device=valid_kpts.device, dtype=valid_kpts.dtype)
        valid_kpts = valid_kpts * scale
        
        # Filter positive scores
        final_mask = valid_scores > 0
        final_kpts = valid_kpts[final_mask]
        final_scores = valid_scores[final_mask]
        final_feats = feats[final_mask]
        
        return final_kpts, final_scores, final_feats
    
    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        """
        Match two images using XFeat + LighterGlue.
        
        NOTE: This is a simplified version that returns the extracted features.
        Full LighterGlue integration requires complex attention mechanisms that
        may not be easily compatible with TorchScript. 
        
        For full C++ deployment, consider:
        1. Exporting XFeat feature extractor separately
        2. Implementing a simpler matcher (like mutual nearest neighbors) in C++
        3. Or exporting LighterGlue separately with proper TorchScript compatibility
        
        Returns:
            tuple of (all_kpts0, all_scores0, all_desc0, all_kpts1, all_scores1, all_desc1)
        """
        assert x0.dim() == 4 and int(x0.shape[0]) == 1
        assert x1.dim() == 4 and int(x1.shape[0]) == 1
        
        # Extract features from both images
        kpts0, scores0, desc0 = self.detectAndCompute(x0)
        kpts1, scores1, desc1 = self.detectAndCompute(x1)
        
        # For TorchScript compatibility, we return all features
        # The matching can be done in C++ using the LighterGlue model
        # or a simpler mutual nearest neighbor matcher
        
        return kpts0, scores0, desc0, kpts1, scores1, desc1


def main():
    p = argparse.ArgumentParser(description="XFeat+LighterGlue TorchScript export for matcher-cpp (B=1)")
    p.add_argument("--xfeat-weights", type=str, default="matcher/xfeat/weights/xfeat.pt")
    p.add_argument("--lightglue-weights", type=str, default="matcher/xfeat/weights/xfeat-lighterglue.pt")
    p.add_argument("--topk", type=int, default=4096)
    p.add_argument("--detection-threshold", type=float, default=0.05)
    p.add_argument("--min-conf", type=float, default=0.1)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "matcher-cpp" / "xfeat" / "weights"),
    )
    args = p.parse_args()
    
    xfeat_weights_path = Path(args.xfeat_weights)
    if not xfeat_weights_path.exists():
        raise FileNotFoundError(f"Missing XFeat weights: {xfeat_weights_path}")
    
    lightglue_weights_path = Path(args.lightglue_weights)
    if not lightglue_weights_path.exists():
        print(f"Warning: LighterGlue weights not found at {lightglue_weights_path}")
        print("Exporting XFeat feature extractor only (without LighterGlue matcher)")
        print("You can implement matching in C++ using mutual nearest neighbors or other methods.")
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xfeat_lightglue_fp32_k{int(args.topk)}.pt"
    
    # Note: Due to TorchScript limitations with LighterGlue (kornia dependency),
    # we export only the feature extractor. The matching logic can be implemented in C++.
    print("Note: Exporting XFeat feature extractor for use with LighterGlue")
    print("Full LighterGlue matching should be implemented in C++ or as a separate module")
    
    print("Note: Attempting to export XFeat + LighterGlue with torch.jit.script...")
    
    try:
        m = XFeatLightGlueTorchScript(
            xfeat_weights_path, 
            lightglue_weights_path,
            top_k=args.topk, 
            detection_threshold=args.detection_threshold,
            min_conf=args.min_conf
        )
        m.eval()
        
        # Script on CPU for portability
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = torch.jit.script(m)
            ts = torch.jit.freeze(ts)
        
        ts.save(str(out_path))
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Error during export: {e}")
        print("\nFalling back to XFeat-only export (without LighterGlue)")
        print("LighterGlue matching should be implemented in C++ or as a separate module.\n")
        
        # Export a simpler version without LighterGlue using trace
        from matcher.xfeat.torchscript.convert_torchscript_xfeat import XFeatTorchScript
        m_simple = XFeatTorchScript(xfeat_weights_path, top_k=args.topk, detection_threshold=args.detection_threshold)
        m_simple.eval()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = torch.jit.script(m_simple)
            ts = torch.jit.freeze(ts)
        
        out_path_simple = out_dir / f"xfeat_fp32_k{int(args.topk)}.pt"
        ts.save(str(out_path_simple))
        print(f"Saved simplified version: {out_path_simple}")


if __name__ == "__main__":
    main()
