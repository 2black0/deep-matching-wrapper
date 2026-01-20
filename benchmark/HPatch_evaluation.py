import os
import sys
import cv2
import json
import torch
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matcher.base_matcher import get_matcher, available_models

def homo_trans(coord, H):
    """
    Transform coordinates using homography H.
    """
    kpt_num = coord.shape[0]
    homo_coord = np.concatenate((coord, np.ones((kpt_num, 1))), axis=-1)
    proj_coord = np.matmul(H, homo_coord.T).T
    proj_coord = proj_coord / (proj_coord[:, 2][..., None] + 1e-8)
    return proj_coord[:, 0:2]

def compute_mma(mkpts0, mkpts1, H_gt, thresholds=[3, 5, 7]):
    """
    Compute Mean Matching Accuracy (MMA) at specified thresholds.
    """
    if len(mkpts0) == 0:
        return {f'mma@{t}': 0.0 for t in thresholds}
        
    # Project mkpts0 to image 1 using GT Homography
    mkpts0_proj = homo_trans(mkpts0, H_gt)
    
    # Calculate euclidean distance
    dist = np.linalg.norm(mkpts0_proj - mkpts1, axis=1)
    
    mma = {}
    for t in thresholds:
        mma[f'mma@{t}'] = np.mean(dist <= t)
        
    return mma

def compute_homography_error(mkpts0, mkpts1, H_gt, image_shape, ransac_method='cv-magsac', ransac_th=3.0):
    """
    Compute Homography Error (MHA) using corner reprojection.
    """
    if len(mkpts0) < 4:
        return {'H_error': float('inf'), 'num_inliers': 0}
        
    # Estimate Homography
    if ransac_method == 'cv-magsac':
        method = cv2.USAC_MAGSAC
    elif ransac_method == 'cv-ransac':
        method = cv2.RANSAC
    elif ransac_method == 'cv-lmeds':
        method = cv2.LMEDS
    else:
        method = cv2.RANSAC # Default fallback
        
    H_pred, mask = cv2.findHomography(mkpts0, mkpts1, method, ransac_th)
    
    if H_pred is None:
        return {'H_error': float('inf'), 'num_inliers': 0}
        
    inliers = np.sum(mask) if mask is not None else 0
    
    # Corner reprojection error
    h, w = image_shape[:2]
    corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    
    corners_gt = homo_trans(corners, H_gt)
    corners_pred = homo_trans(corners, H_pred)
    
    error = np.mean(np.linalg.norm(corners_gt - corners_pred, axis=1))
    
    return {'H_error': error, 'num_inliers': inliers}

def compute_auc(errors, thresholds=[1, 3, 5, 10]):
    """
    Compute AUC for homography errors.
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))
    aucs = []
    
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapezoid(y, x) / thr)
        
    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

def evaluate_pair(matcher, seq_path, ref_idx, target_idx, args):
    ref_img_path = os.path.join(seq_path, f"{ref_idx}.ppm")
    target_img_path = os.path.join(seq_path, f"{target_idx}.ppm")
    h_path = os.path.join(seq_path, f"H_{ref_idx}_{target_idx}")
    
    # Load images
    img0_orig = cv2.imread(ref_img_path)
    img1_orig = cv2.imread(target_img_path)
    try:
        H_gt = np.loadtxt(h_path)
    except Exception as e:
        print(f"Error loading Homography {h_path}: {e}")
        return None
    
    if img0_orig is None or img1_orig is None:
        print(f"Error loading {ref_img_path} or {target_img_path}")
        return None

    # Resize
    h, w = img0_orig.shape[:2]
    scale = args.resize / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    
    img0 = cv2.resize(img0_orig, (nw, nh))
    img1 = cv2.resize(img1_orig, (nw, nh))
    
    # Adjust GT homography for resize
    # H_new = T * H_old * inv(T) where T is scaling matrix
    T = np.diag([scale, scale, 1])
    H_gt_scaled = T @ H_gt @ np.linalg.inv(T)
    
    # Match
    t_img0 = torch.from_numpy(img0.astype(np.float32)/255.0).permute(2, 0, 1)
    t_img1 = torch.from_numpy(img1.astype(np.float32)/255.0).permute(2, 0, 1)
    
    t_start = time.time()
    result = matcher(t_img0, t_img1)
    t_end = time.time()
    time_ms = (t_end - t_start) * 1000
    
    mkpts0 = result['matched_kpts0']
    mkpts1 = result['matched_kpts1']
    matcher_inliers = result.get('num_inliers', -1)
    
    # 1. MMA (Pre-RANSAC)
    mma_metrics = compute_mma(mkpts0, mkpts1, H_gt_scaled, thresholds=[3, 5, 7])
    
    # 2. Homography Error (Post-RANSAC)
    # Note: Using args.outlier. Valid choices are checked by argparse.
    h_metrics = compute_homography_error(mkpts0, mkpts1, H_gt_scaled, img0.shape, 
                                         ransac_method=args.outlier, ransac_th=3.0)
    
    return {
        **mma_metrics,
        **h_metrics,
        'matcher_inliers': matcher_inliers,
        'time_ms': time_ms,
        'num_matches': len(mkpts0)
    }

import time

if __name__ == "__main__":
    usage_str = '%(prog)s [-h] [--matcher MATCHER] [--outlier OUTLIER] [--data_root DATA_ROOT] [--resize RESIZE] [--output_path OUTPUT_PATH] [--device DEVICE]'
    parser = argparse.ArgumentParser(description='HPatches evaluation script', formatter_class=argparse.ArgumentDefaultsHelpFormatter, usage=usage_str)
    parser.add_argument('--matcher', type=str, default='orb-nn', choices=available_models, help='Matcher model name')
    parser.add_argument('--outlier', type=str, default='cv-magsac', choices=['cv-magsac', 'cv-ransac', 'cv-lmeds'], help='Outlier rejection method')
    parser.add_argument('--data_root', type=str, default='/home/ardyseto/dataset/hpatches-sequences-release', help='Dataset root path')
    parser.add_argument('--resize', type=int, default=480, help='Image long-side resize')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = f"outputs/hpatch-{args.matcher}"
    os.makedirs(args.output_path, exist_ok=True)
    
    # Init Matcher
    # Pass 'outlier' to matcher just in case it uses it internally, but we also do our own robust estimation for MHA
    matcher = get_matcher(args.matcher, device=args.device, outlier=args.outlier)
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root not found at {args.data_root}")
        sys.exit(1)

    sequences = sorted([x for x in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, x))])
    
    print(f"Running HPatches evaluation on {len(sequences)} sequences...")
    print(f"Matcher: {args.matcher}, Outlier: {args.outlier}, Resize: {args.resize}")
    print(f"Output Path: {args.output_path}")
    
    results = []
    
    try:
        for seq in tqdm(sequences):
            seq_path = os.path.join(args.data_root, seq)
            # Match 1-2, 1-3, 1-4, 1-5, 1-6
            for i in range(2, 7):
                res = evaluate_pair(matcher, seq_path, 1, i, args)
                if res:
                    res['pair_name'] = f"{seq}_1_{i}"
                    res['seq_type'] = 'illumination' if seq.startswith('i_') else 'viewpoint'
                    results.append(res)
                else: 
                     # Handle failure case
                     pass
                    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted. Calculating stats...")

    if not results:
        print("No results found.")
        sys.exit(0)

    # --- Aggregation and Reporting ---
    
    categories = {
        'Total': results,
        'Illumination': [r for r in results if r['seq_type'] == 'illumination'],
        'Viewpoint': [r for r in results if r['seq_type'] == 'viewpoint']
    }

    # Time Stats (Total only)
    times = [r['time_ms'] for r in results]
    time_stats = {
        'min_time': np.min(times),
        'max_time': np.max(times),
        'avg_time': np.mean(times)
    }

    summary = f"HPatches Evaluation Results\n"
    summary += "="*40 + "\n"
    summary += f"Matcher: {args.matcher}\n"
    summary += f"Outlier: {args.outlier}\n"
    summary += f"Data: {args.data_root}\n"
    summary += f"Pairs: {len(results)}\n"
    summary += f"Date: {datetime.datetime.now().isoformat()}\n"
    
    for cat_name, cat_results in categories.items():
        if not cat_results: continue
        
        summary += "-"*40 + "\n"
        summary += f"Subset: {cat_name} ({len(cat_results)} pairs)\n"
        
        # 1. AUC
        h_errors = [r['H_error'] for r in cat_results]
        auc_metrics = compute_auc(h_errors, thresholds=[1, 3, 5, 10])
        summary += "  AUC (GlueFactory):\n"
        for k, v in auc_metrics.items():
            summary += f"    {k.upper()}: {v*100:.2f}\n"

        # 2. MHA
        mha_metrics = {}
        for t in [3, 5, 7]:
            mha_metrics[f'mha@{t}'] = np.mean([r['H_error'] <= t for r in cat_results]) * 100
        summary += "  MHA (LiftFeat):\n"
        for k, v in mha_metrics.items():
            summary += f"    {k.upper()}: {v:.2f}%\n"

        # 3. MMA
        mean_mma = {}
        for t in [3, 5, 7]:
            mean_mma[f'mean_mma@{t}'] = np.mean([r[f'mma@{t}'] for r in cat_results]) * 100
        summary += "  MMA (Matching):\n"
        for k, v in mean_mma.items():
            summary += f"    {k.upper()}: {v:.2f}%\n"

    summary += "-"*40 + "\n"
    summary += "Timing (ms):\n"
    summary += f"  Min: {time_stats['min_time']:.2f}\n"
    summary += f"  Max: {time_stats['max_time']:.2f}\n"
    summary += f"  Avg: {time_stats['avg_time']:.2f}\n"
    summary += "="*40 + "\n"
    summary += "Parameters:\n"
    summary += json.dumps(vars(args), indent=4)

    # Save Detailed CSV
    csv_path = os.path.join(args.output_path, "pair_results.csv")
    with open(csv_path, "w") as f:
        # Define columns
        cols = ["PairName", "SeqType", "NumMatches", "Inliers", "TimeMS", "H_Error"]
        
        # MMA
        mma_th = [3, 5, 7]
        cols.extend([f"MMA@{t}" for t in mma_th])
        
        # MHA (Pass/Fail)
        mha_th = [3, 5, 7]
        cols.extend([f"MHA_Acc@{t}" for t in mha_th])
        
        # AUC
        auc_th = [1, 3, 5, 10]
        cols.extend([f"AUC@{t}" for t in auc_th])
        
        f.write(",".join(cols) + "\n")
        
        for r in results:
            row = [
                r['pair_name'], r['seq_type'], str(r['num_matches']), str(r['num_inliers']), f"{r['time_ms']:.2f}", f"{r['H_error']:.4f}"
            ]
            
            # MMA
            for t in mma_th:
                row.append(f"{r[f'mma@{t}']:.4f}")
            
            # MHA Acc
            err = r['H_error']
            for t in mha_th:
                acc = 1 if err <= t else 0
                row.append(str(acc))
                
            # AUC
            for t in auc_th:
                # AUC per pair = max(0, 1 - error/threshold)
                auc_val = max(0.0, 1.0 - err / t)
                row.append(f"{auc_val:.4f}")
                
            f.write(",".join(row) + "\n")


    
    with open(os.path.join(args.output_path, "results.txt"), "w") as f:
        f.write(summary)
        
    print("\n" + "="*40)
    print(summary)
