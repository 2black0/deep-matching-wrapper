
import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import time
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import matchers
from matcher.xfeat import XFeatMatcher
from matcher.liftfeat import LiftFeatMatcher
from matcher.gim import GIMMatcher
from matcher.subpx import Keypt2SubpxMatcher
from matcher.edm import EDMMatcher
from matcher.lightglue import SuperPointLightGlueMatcher
from matcher.clidd import CLIDDMatcher
from matcher.eloftr import EfficientLoFTRMatcher
# from matcher.handcrafted import HandcraftedMatcher # To be implemented

def get_matcher(name, device):
    name = name.lower()
    if "xfeat" in name:
        mode = "sparse" # default
        if "star" in name:
            mode = "semi-dense" # check init
        elif "lightglue" in name:
            mode = "lightglue"
        
        if name == "xfeat":
            return XFeatMatcher(device=device, mode='xfeat')
        elif name == "xfeat-star":
            return XFeatMatcher(device=device, mode='xfeat-star')
        elif name == "xfeat-lightglue":
            return XFeatMatcher(device=device, mode='xfeat-lightglue')
        elif name == "xfeat-subpx":
            return Keypt2SubpxMatcher(device=device, mode='xfeat-subpx')
        elif name == "xfeat-lightglue-subpx":
            return Keypt2SubpxMatcher(device=device, mode='xfeat-lightglue-subpx')
            
    elif "liftfeat" in name:
        return LiftFeatMatcher(device=device)
        
    elif "superpoint-lightglue" in name and "subpx" not in name:
        return SuperPointLightGlueMatcher(device=device)

    elif "gim" in name:
        # GIM is SuperPoint+LightGlue finetuned on 100h of data
        return GIMMatcher(device=device) 
        
    elif "subpx" in name:
         # Handle subpx variants that weren't caught by xfeat-subpx block above (check strict ordering)
         # Actually xfeat subpx is handled in first block.
         # So this is mainly for superpoint-lightglue-subpx
         if "superpoint-lightglue" in name:
             return Keypt2SubpxMatcher(device=device, mode='superpoint-lightglue-subpx')
        
    elif "edm" in name:
        return EDMMatcher(device=device)

    elif "clidd" in name:
        return CLIDDMatcher(device=device, model_name=name)

    elif "eloftr" in name or "efficient-loftr" in name:
        return EfficientLoFTRMatcher(device=device)
        
    elif "orb" in name or "sift" in name:
        from matcher.handcrafted import HandcraftedMatcher
        return HandcraftedMatcher(device=device, method=name)
        
    print(f"Unknown matcher: {name}")
    return None

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
        
        if mkpts0 is None:
            n_matches = 0
            n_inliers = 0
        else:
             n_matches = len(mkpts0)
             # Basic RANSAC for inliers check (using opencv)
             if n_matches >= 4:
                 try:
                    import cv2
                    H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
                    if inliers is not None:
                         n_inliers = inliers.sum()
                    else:
                         n_inliers = 0
                 except:
                    n_inliers = 0
             else:
                 n_inliers = n_matches # Assume valid if few? or 0.
        
        print("\nResults:")
        print(f"  Matched Keypoints: {n_matches}")
        print(f"  Inliers: {n_inliers}")
        
        def safe_shape(x):
            if hasattr(x, "shape"): return x.shape
            return "None"
        
        def safe_count(x):
            if hasattr(x, "shape") and len(x.shape) > 0: 
                return x.shape[0]
            return 0
            
        print(f"  Total Keypoints0: {safe_count(kpts0)}")
        print(f"  Total Keypoints1: {safe_count(kpts1)}")
        print(f"  All Keypoints0: {safe_shape(kpts0)}")
        print(f"  All Keypoints1: {safe_shape(kpts1)}")
        print(f"  All Descriptors0: {safe_shape(desc0)}")
        print(f"  All Descriptors1: {safe_shape(desc1)}")
        
        if n_matches > 0:
            # Print first 2 matches as sample
            print(f"  Matched Keypoints0: {mkpts0[:2]}")
            print(f"  Matched Keypoints1: {mkpts1[:2]}")
            
        print(f"SUCCESS: {matcher_name} | Time: {end-start:.3f}s")
        print("=======================================================")
        
    except Exception as e:
        print(f"FAILED: {matcher_name}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    all_matchers = [
        "xfeat", "xfeat-star", "xfeat-lightglue",
        "liftfeat", 
        "gim-lightglue", 
        "edm",
        "orb-nn", "sift-nn", "sift-lightglue",
        "superpoint-lightglue",
        "xfeat-subpx", "xfeat-lightglue-subpx", "superpoint-lightglue-subpx",
        "clidd-a48", "clidd-n64", "clidd-t64", "clidd-s64", "clidd-m64", "clidd-l64", "clidd-g128", "clidd-e128", "clidd-u128",
        "eloftr"
    ]
    
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
