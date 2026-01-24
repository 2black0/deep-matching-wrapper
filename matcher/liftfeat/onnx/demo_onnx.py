import os
import warnings
from scipy.ndimage import map_coordinates

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
from scipy.spatial.distance import cdist

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
    
    # MaxPool dengan scipy maximum_filter
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(heatmap, size=kernel_size, mode='constant')
    
    # Threshold disesuaikan (default 0.01)
    peaks = (heatmap == local_max) & (heatmap > threshold)
    return peaks

def extract_keypoints(heatmap, descriptors_map, top_k=4096, threshold=0.01):
    heatmap = heatmap[0, 0]
    H, W = heatmap.shape
    
    # Debug: check heatmap statistics
    # print(f"Heatmap stats: min={heatmap.min():.6f}, max={heatmap.max():.6f}, mean={heatmap.mean():.6f}")
    
    peaks = simple_nms(heatmap, radius=2, threshold=threshold) 
    y, x = np.where(peaks)
    scores = heatmap[y, x]
    
    # Top-k selection (baseline menggunakan 4096 sebagai default)
    indices = np.argsort(scores)[::-1][:top_k]
    y, x, scores = y[indices], x[indices], scores[indices]
    kpts = np.stack([x, y], axis=1).astype(np.float32)
    
    # Logic dari InterpolateSparse2d (interpolator.py)
    desc_map = descriptors_map[0] # (C, H/8, W/8)
    C, h_f, w_f = desc_map.shape
    
    # Normalize coordinates seperti di interpolator.py:20
    # normgrid: 2. * (x/(torch.tensor([W-1, H-1]))) - 1.
    # Tapi untuk grid_sample, kita perlu normalize ke descriptor map resolution
    grid_x = 2.0 * (x / (W - 1)) - 1.0
    grid_y = 2.0 * (y / (H - 1)) - 1.0
    
    # Convert normalized grid to descriptor map coordinates for map_coordinates
    # map_coordinates expects coordinates in [0, h_f-1] and [0, w_f-1]
    desc_x = (grid_x + 1.0) * 0.5 * (w_f - 1)
    desc_y = (grid_y + 1.0) * 0.5 * (h_f - 1)
    
    descs = np.zeros((len(x), C), dtype=np.float32)
    for i in range(C):
        # Menggunakan order=3 untuk simulasi Bicubic Interpolation (interpolator.py)
        descs[:, i] = map_coordinates(desc_map[i], [desc_y, desc_x], 
                                     order=3, mode='constant', cval=0.0)
        
    # L2 Normalization (PENTING: dilakukan setelah interpolasi)
    norm = np.linalg.norm(descs, axis=1, keepdims=True)
    descs = descs / (norm + 1e-8)
    
    return kpts, scores, descs

def match_descriptors(desc0, desc1, threshold=0.8):
    dists = cdist(desc0, desc1, metric='cosine')
    idx0 = np.arange(len(desc0))
    idx1 = np.argmin(dists, axis=1)
    idx1_back = np.argmin(dists, axis=0)
    mutual = (idx1_back[idx1] == idx0)
    good = dists[idx0, idx1] < threshold
    mask = mutual & good
    return idx0[mask], idx1[mask]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, required=True)
    parser.add_argument("--img2", type=str, required=True)
    parser.add_argument("--weights", type=str, default="matcher/liftfeat/weights/liftfeat_fp32_640x480.onnx")
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

    kpts0, scores0, desc0 = extract_keypoints(out0[0], out0[1])
    kpts1, scores1, desc1 = extract_keypoints(out1[0], out1[1])

    idx0, idx1 = match_descriptors(desc0, desc1)
    mkpts0, mkpts1 = kpts0[idx0], kpts1[idx1]

    # Calculate Inliers using RANSAC
    num_inliers = 0
    if len(mkpts0) > 4:
        _, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
        num_inliers = int(mask.sum()) if mask is not None else 0

    # Print Results matching your requested format
    print(f"\nResults:")
    print(f"  Total Keypoints0: {len(kpts0)} (int)")
    print(f"  Total Keypoints1: {len(kpts1)} (int)")
    print(f"  Matched Keypoints: {len(mkpts0)} (int)")
    print(f"  Inliers: {num_inliers} (int)")
    print(f"  Ratio: {(num_inliers/len(mkpts0) if len(mkpts0)>0 else 0):.2f} (float)")
    
    print(f"  All Keypoints0: {kpts0[:2].tolist()}, ... (numpy.ndarray, dtype={kpts0.dtype}, shape={kpts0.shape})")
    print(f"  All Keypoints1: {kpts1[:2].tolist()}, ... (numpy.ndarray, dtype={kpts1.dtype}, shape={kpts1.shape})")
    print(f"  All Descriptors0: [{desc0[0, :4].tolist()}...], ... (numpy.ndarray, dtype={desc0.dtype}, shape={desc0.shape})")
    print(f"  All Descriptors1: [{desc1[0, :4].tolist()}...], ... (numpy.ndarray, dtype={desc1.dtype}, shape={desc1.shape})")
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

    for i, (p0, p1) in enumerate(zip(mkpts0, mkpts1)):
        color = (0, 255, 0) # Inliers default color
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