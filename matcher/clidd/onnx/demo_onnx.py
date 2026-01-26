import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import sys
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def load_image(img_path, size=(640, 480)):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size)
    inp = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return inp[None, ...], img_resized


def clidd_match(desc0, desc1, beta=20.0, min_score=0.01):
    """Numpy port of matcher/clidd/modules/clidd_wrapper.py:CLIDD.match."""
    if desc0.shape[0] == 0 or desc1.shape[0] == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    sim = desc0 @ desc1.T
    dist = np.exp((sim - 1.0) * float(beta))
    sum1 = dist.sum(axis=1, keepdims=True)
    sum2 = dist.sum(axis=0, keepdims=True)
    dist = (dist * dist) / (sum1 * sum2 + 1e-12)

    nn12 = dist.argmax(axis=1)
    nn21 = dist.argmax(axis=0)

    ids1 = np.arange(dist.shape[0])
    mutual = ids1 == nn21[nn12]

    scores = dist[ids1, nn12]
    keep = mutual & (scores > float(min_score))

    return ids1[keep].astype(np.int64), nn12[keep].astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Model name (A48, E128, etc.)")
    parser.add_argument("--img1", type=str, required=True)
    parser.add_argument("--img2", type=str, required=True)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--size", nargs=2, type=int, default=[640, 480], help="Width Height")
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--score-thresh", type=float, default=-5.0)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--min-match", type=float, default=0.01)
    parser.add_argument("--draw-all", action="store_true", help="Draw all matches (including outliers)")
    args = parser.parse_args()

    cfg = args.weights.upper()
    W, H = args.size

    weights_dir = Path(__file__).parent.parent / "weights"
    onnx_path = weights_dir / f"clidd_{cfg.lower()}_{args.dtype}_{W}x{H}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    inp0, img0_vis = load_image(args.img1, size=(W, H))
    inp1, img1_vis = load_image(args.img2, size=(W, H))

    if args.dtype == "fp16":
        inp0 = inp0.astype(np.float16)
        inp1 = inp1.astype(np.float16)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    device_name = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"

    print(f"\n==================== Testing clidd-onnx ====================")
    print(f"Running {onnx_path.stem} on {device_name}")
    print(f"Loading images: {args.img1} and {args.img2}")

    start = time.time()
    k0, s0, d0 = session.run(None, {"image": inp0})
    k1, s1, d1 = session.run(None, {"image": inp1})
    end = time.time()
    latency_ms = (end - start) * 1000 / 2

    # Outputs are (B, topk, ...)
    k0, s0, d0 = k0[0].astype(np.float32), s0[0].astype(np.float32), d0[0].astype(np.float32)
    k1, s1, d1 = k1[0].astype(np.float32), s1[0].astype(np.float32), d1[0].astype(np.float32)

    # Filter obvious invalid fillers (if model produced fewer than topk real points)
    keep0 = s0 > max(args.score_thresh, -1e7)
    keep1 = s1 > max(args.score_thresh, -1e7)
    kpts0, scores0, desc0 = k0[keep0], s0[keep0], d0[keep0]
    kpts1, scores1, desc1 = k1[keep1], s1[keep1], d1[keep1]

    idx0, idx1 = clidd_match(desc0, desc1, beta=args.beta, min_score=args.min_match)
    mkpts0, mkpts1 = kpts0[idx0], kpts1[idx1]

    num_inliers = 0
    inlier_mask = None
    if len(mkpts0) > 4:
        _, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
        num_inliers = int(mask.sum()) if mask is not None else 0
        inlier_mask = mask.reshape(-1).astype(bool) if mask is not None else None

    print(f"\nResults:")
    print(f"  Total Keypoints0: {len(kpts0)} (int)")
    print(f"  Total Keypoints1: {len(kpts1)} (int)")
    print(f"  Matched Keypoints: {len(mkpts0)} (int)")
    print(f"  Inliers: {num_inliers} (int)")
    print(f"  Ratio: {(num_inliers/len(mkpts0) if len(mkpts0)>0 else 0):.2f} (float)")

    print(
        f"  All Keypoints0: {kpts0[:2].tolist()}, ... (numpy.ndarray, dtype={kpts0.dtype}, shape={kpts0.shape})"
    )
    print(
        f"  All Keypoints1: {kpts1[:2].tolist()}, ... (numpy.ndarray, dtype={kpts1.dtype}, shape={kpts1.shape})"
    )
    if len(desc0) > 0:
        print(
            f"  All Descriptors0: [{desc0[0, :4].tolist()}...], ... (numpy.ndarray, dtype={desc0.dtype}, shape={desc0.shape})"
        )
    else:
        print(f"  All Descriptors0: [] (numpy.ndarray, dtype={desc0.dtype}, shape={desc0.shape})")
    if len(desc1) > 0:
        print(
            f"  All Descriptors1: [{desc1[0, :4].tolist()}...], ... (numpy.ndarray, dtype={desc1.dtype}, shape={desc1.shape})"
        )
    else:
        print(f"  All Descriptors1: [] (numpy.ndarray, dtype={desc1.dtype}, shape={desc1.shape})")
    if len(mkpts0) > 0:
        print(
            f"  Matched Keypoints0: {mkpts0[:2].tolist()}, ... (numpy.ndarray, dtype={mkpts0.dtype}, shape={mkpts0.shape})"
        )
        print(
            f"  Matched Keypoints1: {mkpts1[:2].tolist()}, ... (numpy.ndarray, dtype={mkpts1.dtype}, shape={mkpts1.shape})"
        )

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
        color = (0, 255, 0) if is_inlier else (255, 0, 0)
        pt0 = (int(p0[0]), int(p0[1]))
        pt1 = (int(p1[0] + w0), int(p1[1]))
        cv2.line(vis_img, pt0, pt1, color, 1, cv2.LINE_AA)
        cv2.circle(vis_img, pt0, 2, (255, 0, 0), -1)
        cv2.circle(vis_img, pt1, 2, (255, 0, 0), -1)

    save_path = str(Path(__file__).parent / "result.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"Saved matching result to: {save_path}")


if __name__ == "__main__":
    main()
