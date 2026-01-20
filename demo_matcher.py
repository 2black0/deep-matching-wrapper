
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

def test_matcher(matcher_name, img0_path=None, img1_path=None, output_enabled=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Log buffer
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    try:
        log(f"\n==================== Testing {matcher_name} ====================")
        log(f"Running {matcher_name} on {device}")
        
        matcher = get_matcher(matcher_name, device)
        if matcher is None:
            return

        # Load images
        if img0_path and img1_path:
            log(f"Loading images: {img0_path} and {img1_path}")
            img0 = load_image(img0_path, device)
            img1 = load_image(img1_path, device)
        else:
            # Fallback to random noise
            log("Using random noise (512x512)")
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
        
        # Inliers for visualization
        inliers0 = res.get('inlier_kpts0')
        inliers1 = res.get('inlier_kpts1')

        # Prepare values
        latency_ms = (end - start) * 1000
        n_matches = len(mkpts0) if mkpts0 is not None else 0
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

        log("\nResults:")
        log(f"  Total Keypoints0: {len(kpts0)} (int)")
        log(f"  Total Keypoints1: {len(kpts1)} (int)")
        log(f"  Matched Keypoints: {n_matches} (int)")
        log(f"  Inliers: {n_inliers} (int)")
        log(f"  Ratio: {ratio:.2f} (float)")
        
        # Show sample values for All Keypoints
        if len(kpts0) > 0:
            log(f"  All Keypoints0: {format_sample(kpts0)}, ... {get_type_str(kpts0)}")
        else:
            log(f"  All Keypoints0: [] {get_type_str(kpts0)}")
            
        if len(kpts1) > 0:
            log(f"  All Keypoints1: {format_sample(kpts1)}, ... {get_type_str(kpts1)}")
        else:
            log(f"  All Keypoints1: [] {get_type_str(kpts1)}")
        
        # Show sample values for All Descriptors
        if len(desc0) > 0:
            sample = f"[{desc0[0][:4]}...]" if desc0.shape[1] > 4 else f"[{desc0[0]}]"
            log(f"  All Descriptors0: {sample}, ... {get_type_str(desc0)}")
        else:
            log(f"  All Descriptors0: [] {get_type_str(desc0)}")
            
        if len(desc1) > 0:
            sample = f"[{desc1[0][:4]}...]" if desc1.shape[1] > 4 else f"[{desc1[0]}]"
            log(f"  All Descriptors1: {sample}, ... {get_type_str(desc1)}")
        else:
            log(f"  All Descriptors1: [] {get_type_str(desc1)}")
        
        # Show matched keypoints
        if n_matches > 0:
            log(f"  Matched Keypoints0: {format_sample(mkpts0)}, ... {get_type_str(mkpts0)}")
            log(f"  Matched Keypoints1: {format_sample(mkpts1)}, ... {get_type_str(mkpts1)}")
            
        log(f"\nTime: {latency_ms:.0f} ms")
        log("=======================================================")
        
        # Output handling
        if output_enabled and img0_path and img1_path:
            # Construct output path
            stem1 = Path(img0_path).stem
            stem2 = Path(img1_path).stem
            output_dir = Path(f"outputs/matching/{matcher_name}_{stem1}_{stem2}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save log
            with open(output_dir / "result.txt", "w") as f:
                f.write("\n".join(log_lines))
                
            # Save visualization (result.jpg)
            if inliers0 is not None and len(inliers0) > 0:
                # Reload images for drawing (to ensure CV2 formatting)
                img1_cv = cv2.imread(str(img0_path))
                img2_cv = cv2.imread(str(img1_path))
                
                # Create KeyPoints from inliers (explicit float cast for safety)
                kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in inliers0]
                kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in inliers1]
                
                # Create matches (1-to-1 since they are paired inliers)
                matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]
                
                out_img = cv2.drawMatches(
                    img1_cv, kp1, 
                    img2_cv, kp2, 
                    matches, None,
                    matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    flags=2  # DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                
                vis_path = output_dir / "result.jpg"
                cv2.imwrite(str(vis_path), out_img)
                print(f"Saved output to {output_dir}")
        
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
    parser.add_argument("--output", type=str, choices=['yes', 'no'], default='no',
                       help="Save output visualization and logs (default: no)")
    args = parser.parse_args()
    
    # Check if images exist
    img1_path = Path(args.img1)
    img2_path = Path(args.img2)
    
    output_yes = (args.output == 'yes')
    
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
             test_matcher(m, img1_path, img2_path, output_yes)
    else:
        test_matcher(args.matcher, img1_path, img2_path, output_yes)
