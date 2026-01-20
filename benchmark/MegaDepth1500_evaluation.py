
import os
import sys
import cv2
import json
import torch
import poselib
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matcher.base_matcher import BaseMatcher, get_matcher, available_models

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    
    if n < 1e-6:
        # If GT translation is effectively zero (pure rotation)
        if np.linalg.norm(t_gt) < ignore_gt_t_thr:
             t_err = 0
        else:
             # Estimated translation is zero but GT is not -> max error
             t_err = 180.0
    else:
        t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
        t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
        
    # handle pure rotation case if ignore_gt_t_thr > 0
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # handle numerical errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err

def intrinsics_to_camera(K):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }

def estimate_pose_poselib(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    M, info = poselib.estimate_relative_pose(
        kpts0, kpts1,
        intrinsics_to_camera(K0),
        intrinsics_to_camera(K1),
        {
            "max_epipolar_error": thresh,
            "success_prob": conf,
            "min_iterations": 20,
            "max_iterations": 10000,
        },
    )
    if M is None:
        return None
    return M.R, M.t, info["inliers"]

def estimate_pose_opencv(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    # Essential Matrix
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1,
        cameraMatrix1=K0, cameraMatrix2=K1,
        method=cv2.RANSAC,
        prob=conf,
        threshold=thresh
    )
    if E is None:
        return None
    
    # Recover Pose
    _, R, t, mask = cv2.recoverPose(E, kpts0, kpts1, K0, K1, mask=mask)
    return R, t.ravel(), mask.ravel()

import time

def compute_pose_error(matcher, data, args):
    # Load and resize images
    img0_orig = cv2.imread(os.path.join(args.data_root, data['pair_names'][0]))
    img1_orig = cv2.imread(os.path.join(args.data_root, data['pair_names'][1]))
    
    if img0_orig is None or img1_orig is None:
        print(f"Error loading images: {data['pair_names']}")
        return None

    # Calculate resize scale
    def get_resize_size(h, w, size):
        scale = size / max(h, w)
        return int(h * scale), int(w * scale), scale

    h0, w0 = img0_orig.shape[:2]
    h1, w1 = img1_orig.shape[:2]
    
    nh0, nw0, scale0 = get_resize_size(h0, w0, args.resize)
    nh1, nw1, scale1 = get_resize_size(h1, w1, args.resize)
    
    img0 = cv2.resize(img0_orig, (nw0, nh0))
    img1 = cv2.resize(img1_orig, (nw1, nh1))
    
    # Match
    # Convert to tensor for BaseMatcher
    t_img0 = torch.from_numpy(img0.astype(np.float32)/255.0).permute(2, 0, 1)
    t_img1 = torch.from_numpy(img1.astype(np.float32)/255.0).permute(2, 0, 1)
    
    # Run matcher
    t_start = time.time()
    result = matcher(t_img0, t_img1)
    t_end = time.time()
    time_ms = (t_end - t_start) * 1000
    
    mkpts0 = result['matched_kpts0']
    mkpts1 = result['matched_kpts1']
    matcher_inliers = result.get('num_inliers', -1)
    
    if len(mkpts0) < 5:
        return {'R_err': float('inf'), 't_err': float('inf'), 'inliers': 0, 'matcher_inliers': matcher_inliers, 'time_ms': time_ms}

    # Rescale keypoints to original resolution
    mkpts0 = mkpts0 / scale0
    mkpts1 = mkpts1 / scale1
    
    K0 = np.array(data['K0'])
    K1 = np.array(data['K1'])
    T_0to1 = np.array(data['T_0to1'])

    best_inliers = -1
    best_ret = None
    
    # Robustness loop
    for _ in range(args.robustness):
        if args.estimator == 'poselib':
            ret = estimate_pose_poselib(mkpts0, mkpts1, K0, K1, args.ransac_th)
        else:
            ret = estimate_pose_opencv(mkpts0, mkpts1, K0, K1, args.ransac_th)
            
        if ret is not None:
            R, t, inliers = ret
            if np.sum(inliers) > best_inliers:
                best_inliers = np.sum(inliers)
                best_ret = ret
    
    if best_ret is None:
         return {'R_err': float('inf'), 't_err': float('inf'), 'inliers': 0, 'matcher_inliers': matcher_inliers, 'time_ms': time_ms}

    R, t, inliers = best_ret
    t_err, R_err = relative_pose_error(T_0to1, R, t)
    
    return {
        'R_err': R_err,
        't_err': t_err,
        'inliers': best_inliers,
        'matcher_inliers': matcher_inliers,
        'time_ms': time_ms
    }

def compute_auc(errors, thresholds=[5, 10, 20]):
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))
    aucs = []
    
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapezoid(y, x) / thr)
        
    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

if __name__ == "__main__":
    usage_str = '%(prog)s [-h] [--matcher MATCHER] [--outlier OUTLIER] [--data_root DATA_ROOT] [--json_file JSON_FILE] [--resize RESIZE] [--estimator ESTIMATOR] [--ransac_th RANSAC_TH] [--robustness ROBUSTNESS] [--device DEVICE]'
    parser = argparse.ArgumentParser(description='MegaDepth evaluation script', formatter_class=argparse.ArgumentDefaultsHelpFormatter, usage=usage_str)
    parser.add_argument('--matcher', type=str, default='orb-nn', choices=available_models, help='Matcher model name')
    parser.add_argument('--outlier', type=str, default='cv-magsac', choices=['cv-magsac', 'cv-ransac', 'cv-gc-ransac', 'pl-lo-ransac', 'pl-prosac', 'degensac'], help='Outlier rejection method (passed to matcher)')
    parser.add_argument('--data_root', type=str, default='/home/ardyseto/dataset/megadepth_test_1500', help='Dataset root path')
    parser.add_argument('--json_file', type=str, default='/home/ardyseto/dataset/megadepth_1500.json', help='Dataset JSON file')
    parser.add_argument('--resize', type=int, default=832, help='Image long-side resize for matching')
    parser.add_argument('--estimator', type=str, default='poselib', choices=['poselib', 'opencv'], help='Pose estimator library')
    parser.add_argument('--ransac_th', type=float, default=1.0, help='RANSAC threshold (pixels)')
    parser.add_argument('--robustness', type=int, default=1, help='Number of RANSAC runs per pair')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--max', type=int, default=None, help='Maximum number of pairs to evaluate')
    
    args = parser.parse_args()

    # Determine output path
    if args.output_path is None:
        args.output_path = f"outputs/megadepth/{args.matcher}"
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load Data
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found at {args.json_file}")
        sys.exit(1)
        
    with open(args.json_file, 'r') as f:
        data_list = json.load(f)
        
    if args.max:
        data_list = data_list[:args.max]
        
    # Init Matcher
    # Pass 'outlier' to matcher. BaseMatcher uses this for its internal RANSAC (which we capture in 'matcher_inliers').
    matcher = get_matcher(args.matcher, device=args.device, outlier=args.outlier)
    
    pair_results = [] # Store detailed results
    print(f"Running evaluation on {len(data_list)} pairs using {args.matcher}...")
    print(f"Resize: {args.resize}, Estimator: {args.estimator} (th={args.ransac_th}), Robustness: {args.robustness}x")
    print(f"Matcher Outlier: {args.outlier}")
    print(f"Output Path: {args.output_path}")

    try:
        for data in tqdm(data_list):
            res = compute_pose_error(matcher, data, args)
            pair_name = f"{data['pair_names'][0]}-{data['pair_names'][1]}"
            if res:
                err = max(res['R_err'], res['t_err'])
                pair_results.append({
                    "name": pair_name,
                    "R_err": res['R_err'],
                    "t_err": res['t_err'],
                    "max_err": err,
                    "inliers_est": res['inliers'],
                    "inliers_matcher": res['matcher_inliers'],
                    "time_ms": res['time_ms']
                })
            else:
                 # Should fail gracefully and return infs above, but just in case
                 pair_results.append({
                    "name": pair_name,
                    "R_err": float('inf'),
                    "t_err": float('inf'),
                    "max_err": float('inf'),
                    "inliers_est": 0,
                    "inliers_matcher": 0,
                    "time_ms": 0.0
                })
                
    except KeyboardInterrupt:
        print("\nEvaluation interrupted. Calculating stats on processed pairs...")
        
    if not pair_results:
        print("No results found.")
        sys.exit(0)
    
    # Save per-pair results to CSV
    csv_path = os.path.join(args.output_path, "pair_results.csv")
    with open(csv_path, "w") as f:
        # Header
        f.write("PairName,R_err,t_err,max_err,inliers_est,inliers_matcher,time_ms\n")
        for res in pair_results:
            f.write(f"{res['name']},{res['R_err']:.4f},{res['t_err']:.4f},{res['max_err']:.4f},{res['inliers_est']},{res['inliers_matcher']},{res['time_ms']:.2f}\n")

    # Compute AUC
    errors = [r['max_err'] for r in pair_results]
    aucs = compute_auc(errors)
    
    # Compute Time Stats
    times = [r['time_ms'] for r in pair_results]
    min_time = np.min(times)
    max_time = np.max(times)
    avg_time = np.mean(times)
    
    # Save Combined Results (Params + Summaries)
    summary = f"Evaluation Results\n"
    summary += "="*40 + "\n"
    summary += f"Matcher: {args.matcher} (Outlier: {args.outlier})\n"
    summary += f"Pairs: {len(pair_results)}\n"
    summary += f"Date: {datetime.datetime.now().isoformat()}\n"
    summary += "-" * 40 + "\n"
    summary += "AUC Metrics:\n"
    summary += f"AUC @  5 deg: {aucs['auc@5']*100:.2f}\n"
    summary += f"AUC @ 10 deg: {aucs['auc@10']*100:.2f}\n"
    summary += f"AUC @ 20 deg: {aucs['auc@20']*100:.2f}\n"
    summary += "-" * 40 + "\n"
    summary += "Time Statistics (ms):\n"
    summary += f"Min: {min_time:.2f} ms\n"
    summary += f"Max: {max_time:.2f} ms\n"
    summary += f"Avg: {avg_time:.2f} ms\n"
    summary += "="*40 + "\n"
    summary += "Parameters:\n"
    summary += json.dumps(vars(args), indent=4) + "\n"
    
    with open(os.path.join(args.output_path, "results.txt"), "w") as f:
        f.write(summary)

    print("\n" + "="*40)
    print(summary)
