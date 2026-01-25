import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort


def load_image(path: str, size_wh: tuple[int, int]):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    W, H = size_wh
    img_resized = cv2.resize(img_rgb, (W, H))
    x = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return x[None, ...], img_resized, (orig_w, orig_h)


def _dilate(x: np.ndarray, radius: int):
    k = 2 * radius + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(x.astype(np.float32), kernel, iterations=1)


def simple_nms(scores_hw: np.ndarray, nms_radius: int):
    """NMS port of matcher/lightglue/modules/superpoint.py:simple_nms.

    scores_hw: (H, W)
    returns: suppressed scores (H, W)
    """
    if nms_radius <= 0:
        return scores_hw

    zeros = np.zeros_like(scores_hw, dtype=np.float32)
    local_max = _dilate(scores_hw, nms_radius)
    eps = 1e-6
    max_mask = scores_hw >= (local_max - eps)

    for _ in range(2):
        supp = _dilate(max_mask.astype(np.float32), nms_radius) > 0
        supp_scores = np.where(supp, zeros, scores_hw)
        new_local_max = _dilate(supp_scores, nms_radius)
        new_max_mask = supp_scores >= (new_local_max - eps)
        max_mask = max_mask | (new_max_mask & (~supp))

    return np.where(max_mask, scores_hw, zeros).astype(np.float32)


def sample_descriptors_at_keypoints(
    keypoints_xy: np.ndarray,
    desc_map: np.ndarray,
    image_hw: tuple[int, int],
    s: int = 8,
):
    """Port of matcher/lightglue/modules/superpoint.py:sample_descriptors.

    keypoints_xy: (N,2) in pixel coords in the resized image
    desc_map: (256, H/8, W/8)
    image_hw: (H, W)
    returns: (N, 256)
    """
    H, W = image_hw
    C, h, w = desc_map.shape
    if len(keypoints_xy) == 0:
        return np.zeros((0, C), dtype=np.float32)

    k = keypoints_xy.astype(np.float32).copy()
    k[:, 0] = k[:, 0] - s / 2.0 + 0.5
    k[:, 1] = k[:, 1] - s / 2.0 + 0.5

    denom_x = (w * s - s / 2.0 - 0.5)
    denom_y = (h * s - s / 2.0 - 0.5)
    u = k[:, 0] / float(denom_x) * 2.0 - 1.0
    v = k[:, 1] / float(denom_y) * 2.0 - 1.0

    # align_corners=True mapping to descriptor map pixel coordinates
    map_x = ((u + 1.0) * 0.5 * (w - 1)).reshape(-1, 1).astype(np.float32)
    map_y = ((v + 1.0) * 0.5 * (h - 1)).reshape(-1, 1).astype(np.float32)

    out = np.empty((len(k), C), dtype=np.float32)
    for i in range(C):
        sampled_i = cv2.remap(
            desc_map[i].astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out[:, i] = sampled_i.reshape(-1)

    # L2 normalize
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / (norm + 1e-8)
    return out


def filter_matches(log_assignment: np.ndarray, th: float = 0.1):
    # Port of matcher/lightglue/modules/lightglue.py:filter_matches
    scores = log_assignment[:, :-1, :-1]
    max0_val = scores.max(axis=2)
    max1_val = scores.max(axis=1)
    m0 = scores.argmax(axis=2)
    m1 = scores.argmax(axis=1)

    idx0 = np.arange(m0.shape[1])[None, :]
    idx1 = np.arange(m1.shape[1])[None, :]
    mutual0 = idx0 == np.take_along_axis(m1, m0, axis=1)
    mutual1 = idx1 == np.take_along_axis(m0, m1, axis=1)

    ms0 = np.exp(max0_val)
    zero = np.array(0.0, dtype=ms0.dtype)
    mscores0 = np.where(mutual0, ms0, zero)
    mscores1 = np.where(mutual1, np.take_along_axis(mscores0, m1, axis=1), zero)
    valid0 = mutual0 & (mscores0 > float(th))
    valid1 = mutual1 & np.take_along_axis(valid0, m1, axis=1)

    m0_out = np.where(valid0, m0, -1)
    m1_out = np.where(valid1, m1, -1)
    return m0_out, m1_out, mscores0, mscores1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, required=True)
    parser.add_argument("--img2", type=str, required=True)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--size", nargs=2, type=int, default=[640, 480], help="Width Height")
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--detect-threshold", type=float, default=0.0005)
    parser.add_argument("--remove-borders", type=int, default=4)
    parser.add_argument("--min-conf", type=float, default=0.1)
    parser.add_argument("--draw-all", action="store_true")
    args = parser.parse_args()

    W, H = int(args.size[0]), int(args.size[1])
    dtype = args.dtype.lower()
    weights_dir = Path(__file__).parent.parent / "weights"

    sp_path = weights_dir / f"superpoint_backbone_{dtype}_{W}x{H}.onnx"
    lg_path = weights_dir / f"superpoint_lightglue_{dtype}_k{int(args.top_k)}.onnx"
    if not sp_path.exists():
        raise FileNotFoundError(f"Missing SuperPoint ONNX: {sp_path}")
    if not lg_path.exists():
        raise FileNotFoundError(f"Missing LightGlue ONNX: {lg_path}")

    inp0, img0_vis, orig0 = load_image(args.img1, (W, H))
    inp1, img1_vis, orig1 = load_image(args.img2, (W, H))
    if dtype == "fp16":
        inp0 = inp0.astype(np.float16)
        inp1 = inp1.astype(np.float16)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sp_sess = ort.InferenceSession(str(sp_path), providers=providers)
    device_name = "cuda" if "CUDAExecutionProvider" in sp_sess.get_providers() else "cpu"
    lg_sess = ort.InferenceSession(str(lg_path), providers=providers)

    print(f"\n==================== Testing superpoint-lightglue-onnx ====================")
    print(f"Backbone: {sp_path.stem} | Matcher: {lg_path.stem} | Device: {device_name}")
    print(f"Loading images: {args.img1} and {args.img2}")

    start = time.time()
    s0, d0m = sp_sess.run(None, {"image": inp0})
    s1, d1m = sp_sess.run(None, {"image": inp1})
    end = time.time()
    latency_ms = (end - start) * 1000 / 2

    s0 = s0.astype(np.float32)[0, 0]
    s1 = s1.astype(np.float32)[0, 0]
    d0m = d0m.astype(np.float32)[0]
    d1m = d1m.astype(np.float32)[0]

    # NMS + borders
    s0_nms = simple_nms(s0, nms_radius=int(args.nms_radius))
    s1_nms = simple_nms(s1, nms_radius=int(args.nms_radius))
    b = int(args.remove_borders)
    if b > 0:
        s0_nms[:b, :] = -1
        s0_nms[-b:, :] = -1
        s0_nms[:, :b] = -1
        s0_nms[:, -b:] = -1
        s1_nms[:b, :] = -1
        s1_nms[-b:, :] = -1
        s1_nms[:, :b] = -1
        s1_nms[:, -b:] = -1

    y0, x0 = np.where(s0_nms > float(args.detect_threshold))
    y1, x1 = np.where(s1_nms > float(args.detect_threshold))
    scores0 = s0_nms[y0, x0].astype(np.float32)
    scores1 = s1_nms[y1, x1].astype(np.float32)

    # Top-k selection
    order0 = np.argsort(scores0)[::-1][: min(int(args.top_k), len(scores0))]
    order1 = np.argsort(scores1)[::-1][: min(int(args.top_k), len(scores1))]
    x0, y0, scores0 = x0[order0], y0[order0], scores0[order0]
    x1, y1, scores1 = x1[order1], y1[order1], scores1[order1]

    kpts0_resized = np.stack([x0.astype(np.float32), y0.astype(np.float32)], axis=1)
    kpts1_resized = np.stack([x1.astype(np.float32), y1.astype(np.float32)], axis=1)

    # Sample descriptors (align_corners=True equivalent)
    desc0 = sample_descriptors_at_keypoints(kpts0_resized, d0m, image_hw=(H, W), s=8)
    desc1 = sample_descriptors_at_keypoints(kpts1_resized, d1m, image_hw=(H, W), s=8)

    # Rescale keypoints back to original image coords
    ow0, oh0 = orig0
    ow1, oh1 = orig1
    rw0, rh0 = ow0 / float(W), oh0 / float(H)
    rw1, rh1 = ow1 / float(W), oh1 / float(H)
    kpts0 = kpts0_resized * np.array([rw0, rh0], dtype=np.float32)
    kpts1 = kpts1_resized * np.array([rw1, rh1], dtype=np.float32)

    # Pad to fixed K for LightGlue
    K = int(args.top_k)
    kp0 = np.zeros((1, K, 2), dtype=np.float32)
    kp1 = np.zeros((1, K, 2), dtype=np.float32)
    ds0 = np.zeros((1, K, 256), dtype=np.float32)
    ds1 = np.zeros((1, K, 256), dtype=np.float32)
    n0, n1 = len(kpts0), len(kpts1)
    kp0[0, :n0] = kpts0
    kp1[0, :n1] = kpts1
    ds0[0, :n0] = desc0
    ds1[0, :n1] = desc1

    if dtype == "fp16":
        kp0_i = kp0.astype(np.float16)
        kp1_i = kp1.astype(np.float16)
        ds0_i = ds0.astype(np.float16)
        ds1_i = ds1.astype(np.float16)
    else:
        kp0_i, kp1_i, ds0_i, ds1_i = kp0, kp1, ds0, ds1

    sz0 = np.array([[ow0, oh0]], dtype=np.int64)
    sz1 = np.array([[ow1, oh1]], dtype=np.int64)
    (log_assignment,) = lg_sess.run(
        None,
        {
            "keypoints0": kp0_i,
            "descriptors0": ds0_i,
            "keypoints1": kp1_i,
            "descriptors1": ds1_i,
            "image_size0": sz0,
            "image_size1": sz1,
        },
    )
    log_assignment = log_assignment.astype(np.float32)
    m0, _, _, _ = filter_matches(log_assignment, th=float(args.min_conf))
    idx0 = np.where(m0[0] > -1)[0].astype(np.int64)
    idx1 = m0[0, idx0].astype(np.int64)

    # Drop padded matches
    keep = (idx0 < n0) & (idx1 < n1)
    idx0, idx1 = idx0[keep], idx1[keep]
    mkpts0, mkpts1 = kpts0[idx0], kpts1[idx1]

    num_inliers = 0
    inlier_mask = None
    if len(mkpts0) > 4:
        _, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
        num_inliers = int(mask.sum()) if mask is not None else 0
        inlier_mask = mask.reshape(-1).astype(bool) if mask is not None else None

    print("\nResults:")
    print(f"  Total Keypoints0: {len(kpts0)} (int)")
    print(f"  Total Keypoints1: {len(kpts1)} (int)")
    print(f"  Matched Keypoints: {len(mkpts0)} (int)")
    print(f"  Inliers: {num_inliers} (int)")
    print(f"  Ratio: {(num_inliers/len(mkpts0) if len(mkpts0)>0 else 0):.2f} (float)")
    print(f"\nTime: {latency_ms:.0f} ms")
    print("=======================================================")

    # Visualization (draw inliers by default)
    h0, w0 = img0_vis.shape[:2]
    h1, w1 = img1_vis.shape[:2]
    vis_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    vis_img[:h0, :w0] = img0_vis
    vis_img[:h1, w0:] = img1_vis

    draw_inliers_only = not args.draw_all
    sx0, sy0 = w0 / float(ow0), h0 / float(oh0)
    sx1, sy1 = w1 / float(ow1), h1 / float(oh1)
    for i, (p0, p1) in enumerate(zip(mkpts0, mkpts1)):
        is_inlier = bool(inlier_mask[i]) if inlier_mask is not None else False
        if draw_inliers_only and inlier_mask is not None and not is_inlier:
            continue
        color = (0, 255, 0) if is_inlier else (255, 0, 0)
        pt0 = (int(p0[0] * sx0), int(p0[1] * sy0))
        pt1 = (int(p1[0] * sx1 + w0), int(p1[1] * sy1))
        cv2.line(vis_img, pt0, pt1, color, 1, cv2.LINE_AA)
        cv2.circle(vis_img, pt0, 2, (255, 0, 0), -1)
        cv2.circle(vis_img, pt1, 2, (255, 0, 0), -1)

    save_path = str(Path(__file__).parent / "result.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"Saved matching result to: {save_path}")


if __name__ == "__main__":
    main()
