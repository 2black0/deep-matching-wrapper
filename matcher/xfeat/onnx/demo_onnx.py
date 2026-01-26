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


def load_image(path: str, size_hw: tuple[int, int]):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    W, H = size_hw
    img_resized = cv2.resize(img_rgb, (W, H))
    x = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return x[None, ...], img_resized, (orig_w, orig_h)


def softmax(x: np.ndarray, axis: int):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def get_kpts_heatmap(kpt_logits: np.ndarray, softmax_temp: float = 1.0):
    # Matches matcher/xfeat/modules/xfeat.py:get_kpts_heatmap
    scores = softmax(kpt_logits * float(softmax_temp), axis=1)[:, :64]
    B, _, h, w = scores.shape
    heat = (
        scores.transpose(0, 2, 3, 1)
        .reshape(B, h, w, 8, 8)
        .transpose(0, 1, 3, 2, 4)
        .reshape(B, 1, h * 8, w * 8)
    )
    return heat.astype(np.float32)


def nms_peaks(heatmap_hw: np.ndarray, threshold: float = 0.05, kernel_size: int = 5):
    # heatmap_hw: (H, W)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    local_max = cv2.dilate(heatmap_hw.astype(np.float32), kernel, iterations=1)
    eps = 1e-6
    peaks = (heatmap_hw >= (local_max - eps)) & (heatmap_hw > float(threshold))
    y, x = np.where(peaks)
    return x.astype(np.int64), y.astype(np.int64)


def _remap_sample_chw(feat_chw: np.ndarray, x: np.ndarray, y: np.ndarray, H: int, W: int, interpolation):
    """Sample CxH'xW' feature at (x,y) in original HxW coords.

    Matches torch.grid_sample(..., align_corners=False).
    """
    C, h_f, w_f = feat_chw.shape
    if len(x) == 0:
        return np.zeros((0, C), dtype=np.float32)

    map_x = (x.astype(np.float32) * (w_f / float(W - 1)) - 0.5).reshape(-1, 1)
    map_y = (y.astype(np.float32) * (h_f / float(H - 1)) - 0.5).reshape(-1, 1)

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


def extract_sparse(
    desc_map: np.ndarray,
    kpt_logits: np.ndarray,
    rel_map: np.ndarray,
    orig_size_wh: tuple[int, int],
    resized_size_wh: tuple[int, int],
    top_k: int = 4096,
    det_thresh: float = 0.05,
    softmax_temp: float = 1.0,
):
    # Shapes:
    # desc_map: (1,64,H/8,W/8)
    # kpt_logits: (1,65,H/8,W/8)
    # rel_map: (1,1,H/8,W/8)
    W, H = resized_size_wh

    heat = get_kpts_heatmap(kpt_logits, softmax_temp=softmax_temp)[0, 0]  # (H,W)
    x, y = nms_peaks(heat, threshold=det_thresh, kernel_size=5)

    if len(x) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 64), dtype=np.float32),
        )

    # Reliability sampling at sparse keypoints.
    # nearest(K1h) * bilinear(H1)
    s_nearest = heat[y, x]
    rel = _remap_sample_chw(rel_map[0].astype(np.float32), x, y, H, W, interpolation=cv2.INTER_LINEAR)[:, 0]
    scores = (s_nearest * rel).astype(np.float32)

    # top-k selection by score
    order = np.argsort(scores)[::-1]
    if top_k is not None and top_k > 0:
        order = order[: min(int(top_k), len(order))]
    x, y, scores = x[order], y[order], scores[order]

    # descriptors: bicubic sampling of descriptor map
    descs = _remap_sample_chw(desc_map[0].astype(np.float32), x, y, H, W, interpolation=cv2.INTER_CUBIC)
    norm = np.linalg.norm(descs, axis=1, keepdims=True)
    descs = descs / (norm + 1e-8)

    kpts = np.stack([x.astype(np.float32), y.astype(np.float32)], axis=1)

    # Rescale keypoints back to original image size
    ow, oh = orig_size_wh
    rw, rh = ow / float(W), oh / float(H)
    kpts = kpts * np.array([rw, rh], dtype=np.float32)

    # Match PyTorch behavior: keep only scores > 0
    valid = scores > 0
    return kpts[valid], scores[valid], descs[valid]


def match_mnn(desc0: np.ndarray, desc1: np.ndarray, min_cossim: float = 0.82):
    if len(desc0) == 0 or len(desc1) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    sims = desc0 @ desc1.T
    m01 = sims.argmax(axis=1)
    m10 = sims.argmax(axis=0)
    idx0 = np.arange(len(m01))
    mutual = m10[m01] == idx0
    if min_cossim > 0:
        good = sims[idx0, m01] > float(min_cossim)
        keep = mutual & good
    else:
        keep = mutual
    return idx0[keep].astype(np.int64), m01[keep].astype(np.int64)


def filter_matches(log_assignment: np.ndarray, th: float = 0.1):
    # Port of kornia.feature.lightglue.filter_matches for scores=log_assignment
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
    ms1 = np.where(mutual1, np.take_along_axis(ms0, m1, axis=1), 0.0)
    valid0 = mutual0 & (ms0 > float(th))
    valid1 = mutual1 & np.take_along_axis(valid0, m1, axis=1)

    m0_out = np.where(valid0, m0, -1)
    m1_out = np.where(valid1, m1, -1)
    return m0_out, m1_out, ms0, ms1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matcher", choices=["xfeat", "lightglue"], default="xfeat")
    parser.add_argument("--img1", type=str, required=True)
    parser.add_argument("--img2", type=str, required=True)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--size", nargs=2, type=int, default=[640, 480], help="Width Height")
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--detect-threshold", type=float, default=0.05)
    parser.add_argument("--softmax-temp", type=float, default=1.0)
    parser.add_argument("--min-cossim", type=float, default=0.82, help="Only for matcher=xfeat")
    parser.add_argument("--min-conf", type=float, default=0.1, help="Only for matcher=lightglue")
    parser.add_argument("--draw-all", action="store_true")
    args = parser.parse_args()

    W, H = int(args.size[0]), int(args.size[1])
    dtype = args.dtype.lower()

    weights_dir = Path(__file__).parent.parent / "weights"
    backbone_path = weights_dir / f"xfeat_backbone_{dtype}_{W}x{H}.onnx"
    if not backbone_path.exists():
        raise FileNotFoundError(f"Missing backbone ONNX: {backbone_path}")

    inp0, img0_vis, orig0 = load_image(args.img1, (W, H))
    inp1, img1_vis, orig1 = load_image(args.img2, (W, H))

    if dtype == "fp16":
        inp0 = inp0.astype(np.float16)
        inp1 = inp1.astype(np.float16)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(backbone_path), providers=providers)
    device_name = "cuda" if "CUDAExecutionProvider" in sess.get_providers() else "cpu"

    print(f"\n==================== Testing xfeat-onnx ====================")
    print(f"Mode: {args.matcher} | Backbone: {backbone_path.stem} | Device: {device_name}")
    print(f"Loading images: {args.img1} and {args.img2}")

    start = time.time()
    d0m, k0l, r0m = sess.run(None, {"image": inp0})
    d1m, k1l, r1m = sess.run(None, {"image": inp1})
    end = time.time()
    latency_ms = (end - start) * 1000 / 2

    d0m = d0m.astype(np.float32)
    d1m = d1m.astype(np.float32)
    k0l = k0l.astype(np.float32)
    k1l = k1l.astype(np.float32)
    r0m = r0m.astype(np.float32)
    r1m = r1m.astype(np.float32)

    kpts0, scores0, desc0 = extract_sparse(
        d0m,
        k0l,
        r0m,
        orig_size_wh=orig0,
        resized_size_wh=(W, H),
        top_k=args.top_k,
        det_thresh=args.detect_threshold,
        softmax_temp=args.softmax_temp,
    )
    kpts1, scores1, desc1 = extract_sparse(
        d1m,
        k1l,
        r1m,
        orig_size_wh=orig1,
        resized_size_wh=(W, H),
        top_k=args.top_k,
        det_thresh=args.detect_threshold,
        softmax_temp=args.softmax_temp,
    )

    if args.matcher == "xfeat":
        idx0, idx1 = match_mnn(desc0, desc1, min_cossim=args.min_cossim)
    else:
        lg_path = weights_dir / f"xfeat_lighterglue_{dtype}_k{int(args.top_k)}.onnx"
        if not lg_path.exists():
            raise FileNotFoundError(f"Missing lightglue ONNX: {lg_path}")

        # Pad to fixed length
        n = int(args.top_k)
        n0, n1 = len(kpts0), len(kpts1)
        kp0 = np.zeros((1, n, 2), dtype=np.float32)
        kp1 = np.zeros((1, n, 2), dtype=np.float32)
        ds0 = np.zeros((1, n, 64), dtype=np.float32)
        ds1 = np.zeros((1, n, 64), dtype=np.float32)
        kp0[0, : min(n0, n)] = kpts0[: min(n0, n)]
        kp1[0, : min(n1, n)] = kpts1[: min(n1, n)]
        ds0[0, : min(n0, n)] = desc0[: min(n0, n)]
        ds1[0, : min(n1, n)] = desc1[: min(n1, n)]

        sz0 = np.array([[orig0[0], orig0[1]]], dtype=np.int64)
        sz1 = np.array([[orig1[0], orig1[1]]], dtype=np.int64)

        if dtype == "fp16":
            kp0_i = kp0.astype(np.float16)
            kp1_i = kp1.astype(np.float16)
            ds0_i = ds0.astype(np.float16)
            ds1_i = ds1.astype(np.float16)
        else:
            kp0_i, kp1_i, ds0_i, ds1_i = kp0, kp1, ds0, ds1

        lg_sess = ort.InferenceSession(str(lg_path), providers=providers)
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
        m0, _, ms0, _ = filter_matches(log_assignment, th=args.min_conf)
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
    for i, (p0, p1) in enumerate(zip(mkpts0, mkpts1)):
        is_inlier = bool(inlier_mask[i]) if inlier_mask is not None else False
        if draw_inliers_only and inlier_mask is not None and not is_inlier:
            continue
        color = (0, 255, 0) if is_inlier else (255, 0, 0)
        # mkpts are in original image coords; convert for the resized visualization
        sx0, sy0 = w0 / float(orig0[0]), h0 / float(orig0[1])
        sx1, sy1 = w1 / float(orig1[0]), h1 / float(orig1[1])
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
