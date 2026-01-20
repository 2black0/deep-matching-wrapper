
import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import time
import cv2
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import matchers
from matcher.base_matcher import get_matcher, AVAILABLE_MATCHERS

def load_image(img_path, device):
    """Load image from path and convert to tensor"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img.to(device)

def test_matcher(matcher_name, img0_path=None, img1_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        print(f"\n==================== Testing {matcher_name} ====================")
        print(f"Running {matcher_name} on {device}")
        
        matcher = get_matcher(matcher_name, device)
        if matcher is None:
            return

        # Load images
        if img0_path and img1_path:
            print(f"Loading images: {img0_path} and {img1_path}")
            img0 = load_image(img0_path, device)
            img1 = load_image(img1_path, device)
        else:
            # Fallback to random noise
            print("Using random noise (512x512)")
            img0 = torch.rand(3, 512, 512).to(device) 
            img1 = torch.rand(3, 512, 512).to(device)
        
        # Warm-up
        _ = matcher(img0, img1)
        if device == "cuda":
            torch.cuda.synchronize()

        # Real timing
        start = time.time()
        res = matcher(img0, img1)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        
        mkpts0 = res['matched_kpts0']
        mkpts1 = res['matched_kpts1']
        kpts0 = res['all_kpts0']
        kpts1 = res['all_kpts1']
        desc0 = res['all_desc0']
        desc1 = res['all_desc1']

        # Prepare values
        latency_ms = (end - start) * 1000
        n_matches = len(mkpts0) if mkpts0 is not None else 0
        # Use inliers from matcher result if available
        n_inliers = res.get('num_inliers', 0)
        ratio = n_inliers / n_matches if n_matches > 0 else 0
        
        def get_type_str(x, include_shape=True):
            if isinstance(x, np.ndarray):
                shape_str = f"{x.shape}" if include_shape else ""
                return f"(numpy.ndarray, dtype={x.dtype}, shape={shape_str})" if include_shape else f"(numpy.ndarray, dtype={x.dtype})"
            if isinstance(x, torch.Tensor):
                shape_str = f"{tuple(x.shape)}" if include_shape else ""
                return f"(torch.Tensor, dtype={x.dtype}, device={x.device}, shape={shape_str})" if include_shape else f"(torch.Tensor, dtype={x.dtype}, device={x.device})"
            return f"({type(x).__name__})"

        def format_sample(pts):
            if pts is None or len(pts) == 0: return "[]"
            s = []
            for p in pts[:2]:
                s.append(f"[{p[0]} {p[1]}]")
            return "[" + ",".join(s) + "]"

        print("\nResults:")
        print(f"  Total Keypoints0: {len(kpts0)} (int)")
        print(f"  Total Keypoints1: {len(kpts1)} (int)")
        print(f"  Matched Keypoints: {n_matches} (int)")
        print(f"  Inliers: {n_inliers} (int)")
        print(f"  Ratio: {ratio:.2f} (float)")
        
        # Show sample values for All Keypoints
        if len(kpts0) > 0:
            print(f"  All Keypoints0: {format_sample(kpts0)}, ... {get_type_str(kpts0)}")
        else:
            print(f"  All Keypoints0: [] {get_type_str(kpts0)}")
            
        if len(kpts1) > 0:
            print(f"  All Keypoints1: {format_sample(kpts1)}, ... {get_type_str(kpts1)}")
        else:
            print(f"  All Keypoints1: [] {get_type_str(kpts1)}")
        
        # Show sample values for All Descriptors
        if len(desc0) > 0:
            # For descriptors, show first descriptor vector (first 4 values)
            sample = f"[{desc0[0][:4]}...]" if desc0.shape[1] > 4 else f"[{desc0[0]}]"
            print(f"  All Descriptors0: {sample}, ... {get_type_str(desc0)}")
        else:
            print(f"  All Descriptors0: [] {get_type_str(desc0)}")
            
        if len(desc1) > 0:
            sample = f"[{desc1[0][:4]}...]" if desc1.shape[1] > 4 else f"[{desc1[0]}]"
            print(f"  All Descriptors1: {sample}, ... {get_type_str(desc1)}")
        else:
            print(f"  All Descriptors1: [] {get_type_str(desc1)}")
        
        # Show matched keypoints
        if n_matches > 0:
            print(f"  Matched Keypoints0: {format_sample(mkpts0)}, ... {get_type_str(mkpts0)}")
            print(f"  Matched Keypoints1: {format_sample(mkpts1)}, ... {get_type_str(mkpts1)}")
            
        print(f"\nTime: {latency_ms:.0f} ms")
        print("=======================================================")
        
    except Exception as e:
        print(f"FAILED: {matcher_name}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    all_matchers = AVAILABLE_MATCHERS
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--matcher", type=str, required=True, 
                       help=f"Name of the matcher. Supported: {', '.join(all_matchers)} or 'all' to run all")
    parser.add_argument("--img1", type=str, default="assets/ref.png",
                       help="Path to first image (default: assets/ref.png)")
    parser.add_argument("--img2", type=str, default="assets/tgt.png",
                       help="Path to second image (default: assets/tgt.png)")
    args = parser.parse_args()
    
    # Check if images exist
    img1_path = Path(args.img1)
    img2_path = Path(args.img2)
    
    if not img1_path.exists():
        print(f"Warning: {img1_path} not found, using random noise")
        img1_path = None
        img2_path = None
    elif not img2_path.exists():
        print(f"Warning: {img2_path} not found, using random noise")
        img1_path = None
        img2_path = None
    
    if args.matcher == "all":
        print("Running all matchers...")
        for m in all_matchers:
             test_matcher(m, img1_path, img2_path)
    else:
        test_matcher(args.matcher, img1_path, img2_path)
