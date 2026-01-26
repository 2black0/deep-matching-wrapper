import os
import warnings

# Mengabaikan warning numpy di level sistem
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import sys
import numpy as np
from pathlib import Path
import time
import cv2
import onnxruntime as ort

# Setup path agar bisa akses file dari root
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

def load_image(img_path, size=(640, 480)):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_orig, size)
    input_tensor = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return input_tensor[None, ...], img_resized

def simple_nms(heatmap, radius=2, threshold=0.01):
    # Implementasi NMS dengan MaxPool seperti di liftfeat_wrapper.py:36-50
    # Baseline menggunakan kernel_size=5, stride=1, padding=2 di NonMaxSuppression.__init__
    # yang berarti radius=2 (2*radius+1 = 5)
    kernel_size = 5
    pad = 2
    
    # MaxPool-like operation using dilation (no SciPy dependency).
    # This is equivalent to MaxPool2d(kernel=5, stride=1, padding=2) on a single-channel map.
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    local_max = cv2.dilate(heatmap.astype(np.float32), kernel, iterations=1)

    # Float equality can be brittle; allow a tiny epsilon.
    eps = 1e-6
    peaks = (heatmap >= (local_max - eps)) & (heatmap > threshold)
    return peaks


def _remap_sample(feat_chw, x, y, H, W, interpolation=cv2.INTER_CUBIC):
    """Sample CxH'xW' map at sparse (x,y) defined in original HxW coords.

    Matches torch.grid_sample(..., align_corners=False) coordinate mapping used by
    matcher/liftfeat/modules/interpolator.py.
    """
    C, h_f, w_f = feat_chw.shape

    # grid_sample with align_corners=False:
    # u = 2*x/(W-1) - 1
    # x_src = ((u+1)*w_f - 1)/2 = x*(w_f/(W-1)) - 0.5
    map_x = (x.astype(np.float32) * (w_f / float(W - 1)) - 0.5).reshape(-1, 1)
    map_y = (y.astype(np.float32) * (h_f / float(H - 1)) - 0.5).reshape(-1, 1)

    # OpenCV remap only supports up to 4 channels, so remap each channel.
    out = np.empty((len(x), C), dtype=np.float32)
    for i in range(C):
        sampled_i = cv2.remap(
            feat_chw[i].astype(np.float32),
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out[:, i] = sampled_i.reshape(-1)
    return out

def extract_keypoints(heatmap, descriptors_map, top_k=4096, threshold=0.005, debug=False):
    heatmap = heatmap[0, 0]
    H, W = heatmap.shape
    
    # Debug: check heatmap statistics
    # print(f"Heatmap stats: min={heatmap.min():.6f}, max={heatmap.max():.6f}, mean={heatmap.mean():.6f}")
    
    peaks = simple_nms(heatmap, radius=2, threshold=threshold)
    y, x = np.where(peaks)

    if debug:
        print(
            f"Heatmap stats: min={float(heatmap.min()):.6f}, max={float(heatmap.max()):.6f}, mean={float(heatmap.mean()):.6f}, peaks={len(x)}"
        )

    if len(x) == 0:
        kpts = np.zeros((0, 2), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        descs = np.zeros((0, 64), dtype=np.float32)
        return kpts, scores, descs

    # Keypoints are in (x, y) pixel coordinates.
    kpts = np.stack([x, y], axis=1).astype(np.float32)
    
    # Logic dari InterpolateSparse2d (interpolator.py)
    desc_map = descriptors_map[0]  # (C, H/8, W/8)
    C, h_f, w_f = desc_map.shape

    # Descriptor map is expected to be normalized before sampling in the PyTorch wrapper.
    # Keep the same behavior here.
    denom = np.linalg.norm(desc_map, axis=0, keepdims=True)
    desc_map = desc_map / (denom + 1e-8)

    # Sample scores using the same sampling rule as the PyTorch wrapper.
    # (wrapper uses InterpolateSparse2d("bicubic") for scores too)
    scores = _remap_sample(heatmap[None, ...], x, y, H, W, interpolation=cv2.INTER_CUBIC)[:, 0]

    # Keep top-k by sampled scores (wrapper doesn't hard-limit, but this keeps runtime sane)
    if top_k is not None and top_k > 0 and len(scores) > top_k:
        sel = np.argsort(scores)[::-1][:top_k]
        x, y = x[sel], y[sel]
        kpts = kpts[sel]
        scores = scores[sel]

    # Bicubic sampling aligned with torch.grid_sample(align_corners=False)
    descs = _remap_sample(desc_map, x, y, H, W, interpolation=cv2.INTER_CUBIC)
        
    # L2 Normalization (PENTING: dilakukan setelah interpolasi)
    norm = np.linalg.norm(descs, axis=1, keepdims=True)
    descs = descs / (norm + 1e-8)
    
    return kpts, scores, descs

def match_descriptors(desc0, desc1, min_cossim=-1.0):
    """Mutual nearest neighbor matching like liftfeat_wrapper.py.

    Since descriptors are L2-normalized, dot product = cosine similarity.
    """
    if len(desc0) == 0 or len(desc1) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    sims = desc0 @ desc1.T
    match01 = sims.argmax(axis=1)
    match10 = sims.argmax(axis=0)

    idx0 = np.arange(len(match01))
    mutual = match10[match01] == idx0

    if min_cossim > 0:
        good = sims[idx0, match01] > float(min_cossim)
        keep = mutual & good
    else:
        keep = mutual

    return idx0[keep].astype(np.int64), match01[keep].astype(np.int64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, required=True)
    parser.add_argument("--img2", type=str, required=True)
    parser.add_argument("--weights", type=str, default="matcher/liftfeat/weights/liftfeat_fp32_640x480.onnx")
    parser.add_argument("--detect-threshold", type=float, default=0.005)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--min-cossim", type=float, default=-1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--draw-all", action="store_true", help="Draw all matches (including outliers)")
    args = parser.parse_args()

    # Inference Session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(args.weights, providers=providers)
    device_name = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
    
    input0, img0_vis = load_image(args.img1)
    input1, img1_vis = load_image(args.img2)
    
    if "fp16" in args.weights.lower():
        input0 = input0.astype(np.float16)
        input1 = input1.astype(np.float16)

    print(f"\n==================== Testing liftfeat-onnx ====================")
    print(f"Running {Path(args.weights).stem} on {device_name}")
    print(f"Loading images: {args.img1} and {args.img2}")

    start = time.time()
    out0 = session.run(None, {'image': input0})
    out1 = session.run(None, {'image': input1})
    end = time.time()
    latency_ms = (end - start) * 1000 / 2 

    # Model outputs: kpt_logits (B,65,h,w) and descriptors_map (B,64,h,w)
    kpt0, desc_map0 = out0
    kpt1, desc_map1 = out1

    def softmax(x, axis=1):
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

    def logits_to_heatmap(kpt_logits):
        # Match liftfeat_wrapper.py exactly
        scores_raw = softmax(kpt_logits, axis=1)[:, :64]
        B, _, h_feat, w_feat = scores_raw.shape
        heat = (
            scores_raw.transpose(0, 2, 3, 1)
            .reshape(B, h_feat, w_feat, 8, 8)
            .transpose(0, 1, 3, 2, 4)
            .reshape(B, 1, h_feat * 8, w_feat * 8)
        )
        return heat.astype(np.float32)

    heat0 = logits_to_heatmap(kpt0)
    heat1 = logits_to_heatmap(kpt1)

    kpts0, scores0, desc0 = extract_keypoints(
        heat0, desc_map0, top_k=args.top_k, threshold=args.detect_threshold, debug=args.debug
    )
    kpts1, scores1, desc1 = extract_keypoints(
        heat1, desc_map1, top_k=args.top_k, threshold=args.detect_threshold, debug=args.debug
    )

    idx0, idx1 = match_descriptors(desc0, desc1, min_cossim=args.min_cossim)
    mkpts0, mkpts1 = kpts0[idx0], kpts1[idx1]

    # Calculate Inliers using RANSAC
    num_inliers = 0
    inlier_mask = None
    if len(mkpts0) > 4:
        _, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
        num_inliers = int(mask.sum()) if mask is not None else 0
        inlier_mask = mask.reshape(-1).astype(bool) if mask is not None else None

    # Print Results matching your requested format
    print(f"\nResults:")
    print(f"  Total Keypoints0: {len(kpts0)} (int)")
    print(f"  Total Keypoints1: {len(kpts1)} (int)")
    print(f"  Matched Keypoints: {len(mkpts0)} (int)")
    print(f"  Inliers: {num_inliers} (int)")
    print(f"  Ratio: {(num_inliers/len(mkpts0) if len(mkpts0)>0 else 0):.2f} (float)")
    
    print(f"  All Keypoints0: {kpts0[:2].tolist()}, ... (numpy.ndarray, dtype={kpts0.dtype}, shape={kpts0.shape})")
    print(f"  All Keypoints1: {kpts1[:2].tolist()}, ... (numpy.ndarray, dtype={kpts1.dtype}, shape={kpts1.shape})")
    if len(desc0) > 0:
        print(f"  All Descriptors0: [{desc0[0, :4].tolist()}...], ... (numpy.ndarray, dtype={desc0.dtype}, shape={desc0.shape})")
    else:
        print(f"  All Descriptors0: [] (numpy.ndarray, dtype={desc0.dtype}, shape={desc0.shape})")
    if len(desc1) > 0:
        print(f"  All Descriptors1: [{desc1[0, :4].tolist()}...], ... (numpy.ndarray, dtype={desc1.dtype}, shape={desc1.shape})")
    else:
        print(f"  All Descriptors1: [] (numpy.ndarray, dtype={desc1.dtype}, shape={desc1.shape})")
    print(f"  Matched Keypoints0: {mkpts0[:2].tolist()}, ... (numpy.ndarray, dtype={mkpts0.dtype}, shape={mkpts0.shape})")
    print(f"  Matched Keypoints1: {mkpts1[:2].tolist()}, ... (numpy.ndarray, dtype={mkpts1.dtype}, shape={mkpts1.shape})")

    print(f"\nTime: {latency_ms:.0f} ms")
    print("=======================================================")

    # Visualization
    h0, w0 = img0_vis.shape[:2]
    h1, w1 = img1_vis.shape[:2]
    vis_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    vis_img[:h0, :w0] = img0_vis
    vis_img[:h1, w0:] = img1_vis

    draw_inliers_only = not args.draw_all

    for i, (p0, p1) in enumerate(zip(mkpts0, mkpts1)):
        is_inlier = bool(inlier_mask[i]) if inlier_mask is not None else False
        if draw_inliers_only and inlier_mask is not None and not is_inlier:
            continue

        # Green=inlier, Red=outlier
        color = (0, 255, 0) if is_inlier else (255, 0, 0)
        pt0 = (int(p0[0]), int(p0[1]))
        pt1 = (int(p1[0] + w0), int(p1[1]))
        cv2.line(vis_img, pt0, pt1, color, 1, cv2.LINE_AA)
        cv2.circle(vis_img, pt0, 2, (255, 0, 0), -1)
        cv2.circle(vis_img, pt1, 2, (255, 0, 0), -1)

    save_path = "matcher/liftfeat/onnx/result.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"Saved matching result to: {save_path}")

if __name__ == "__main__":
    main()
