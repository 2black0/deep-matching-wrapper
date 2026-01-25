import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "matcher-onnx"))

from _ort import create_session
from base_matcher import BaseMatcher


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def to_chw_float01(img_rgb: np.ndarray) -> np.ndarray:
    return img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0


def extract_patches_gray(img_chw: np.ndarray, kpts_xy: np.ndarray, patch_radius: int = 5) -> np.ndarray:
    """Replicates Keypt2Subpx patch extraction (grayscale, padded, floor coords)."""
    assert img_chw.ndim == 3 and img_chw.shape[0] in (1, 3)
    C, H, W = img_chw.shape

    if C == 3:
        r, g, b = img_chw[0], img_chw[1], img_chw[2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    else:
        gray = img_chw[0].astype(np.float32)

    ps = 2 * patch_radius + 1
    padded = np.pad(gray, ((patch_radius, patch_radius), (patch_radius, patch_radius)), mode="constant")
    Hp, Wp = padded.shape

    if len(kpts_xy) == 0:
        return np.zeros((1, 0, 1, ps, ps), dtype=np.float32)

    corners = np.floor(kpts_xy).astype(np.int64)
    corners[:, 0] = np.clip(corners[:, 0], 0, Wp - ps)
    corners[:, 1] = np.clip(corners[:, 1], 0, Hp - ps)

    patches = np.empty((len(corners), ps, ps), dtype=np.float32)
    for i, (x0, y0) in enumerate(corners):
        patches[i] = padded[y0 : y0 + ps, x0 : x0 + ps]

    return patches[:, None, :, :][None, ...]  # (1, N, 1, ps, ps)


def _match_indices(mkpts: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    """Map mkpts -> indices into kpts using rounding, then NN fallback."""
    if len(mkpts) == 0 or len(kpts) == 0:
        return np.zeros((0,), dtype=np.int64)
    d = {tuple(np.round(p, 3)): i for i, p in enumerate(kpts)}
    out = np.empty((len(mkpts),), dtype=np.int64)
    for i, p in enumerate(mkpts):
        key = tuple(np.round(p, 3))
        j = d.get(key)
        if j is not None:
            out[i] = j
            continue
        dist = np.linalg.norm(kpts - p[None, :], axis=1)
        out[i] = int(np.argmin(dist))
    return out


def run_demo(args):
    img0_rgb = load_rgb(Path(args.img1))
    img1_rgb = load_rgb(Path(args.img2))

    base_w, base_h = args.size
    img0_resized = cv2.resize(img0_rgb, (base_w, base_h))
    img1_resized = cv2.resize(img1_rgb, (base_w, base_h))
    img0 = to_chw_float01(img0_resized)
    img1 = to_chw_float01(img1_resized)

    # Base matcher (ONNX)
    if args.matcher in ("xfeat-subpx", "xfeat-lightglue-subpx"):
        from xfeat import XFeatONNXMatcher

        mode = "xfeat" if args.matcher == "xfeat-subpx" else "lightglue"
        base = XFeatONNXMatcher(device=args.device, mode=mode, dtype=args.dtype, size=(base_w, base_h), top_k=args.top_k)
        refiner_path = ROOT / "matcher-onnx" / "weights" / "subpx" / f"k2s_xfeat_refiner_{args.dtype}.onnx"
        use_score = False
    else:
        from superpoint_lightglue import SuperPointLightGlueONNXMatcher

        base = SuperPointLightGlueONNXMatcher(
            device=args.device,
            dtype=args.dtype,
            size=(base_w, base_h),
            top_k=args.top_k,
        )
        refiner_path = ROOT / "matcher-onnx" / "weights" / "subpx" / f"k2s_splg_refiner_{args.dtype}.onnx"
        use_score = True

    if not refiner_path.exists():
        raise FileNotFoundError(
            f"Missing SubPX refiner ONNX: {refiner_path}. Run: pixi run python matcher/subpx/onnx/convert-onnx.py --matcher all --dtype {args.dtype.upper()}"
        )
    ref_sess = create_session(str(refiner_path), args.device)

    # Run base matcher
    _ = base(img0, img1)  # warmup
    t0 = time.time()
    res = base(img0, img1)
    t1 = time.time()

    mkpts0 = res["matched_kpts0"].astype(np.float32)
    mkpts1 = res["matched_kpts1"].astype(np.float32)
    kpts0 = res["all_kpts0"].astype(np.float32)
    kpts1 = res["all_kpts1"].astype(np.float32)
    desc0 = res["all_desc0"].astype(np.float32)
    desc1 = res["all_desc1"].astype(np.float32)

    if len(mkpts0) == 0:
        return res, None

    idx0 = _match_indices(mkpts0, kpts0)
    idx1 = _match_indices(mkpts1, kpts1)
    mdesc0 = desc0[idx0]
    mdesc1 = desc1[idx1]
    desc_mean = ((mdesc0 + mdesc1) / 2.0)[None, ...].astype(np.float32)

    patch0 = extract_patches_gray(img0, mkpts0)
    patch1 = extract_patches_gray(img1, mkpts1)

    if args.dtype == "fp16":
        patch0_i = patch0.astype(np.float16)
        patch1_i = patch1.astype(np.float16)
        desc_mean_i = desc_mean.astype(np.float16)
    else:
        patch0_i, patch1_i, desc_mean_i = patch0, patch1, desc_mean

    # Optional score patches (SuperPoint variant)
    if use_score:
        sp_path = ROOT / "matcher-onnx" / "weights" / "lightglue" / f"superpoint_backbone_{args.dtype}_{base_w}x{base_h}.onnx"
        if not sp_path.exists():
            raise FileNotFoundError(f"Missing SuperPoint backbone ONNX for scores: {sp_path}")
        sp_sess = create_session(str(sp_path), args.device)

        inp0 = img0[None, ...].astype(np.float16 if args.dtype == "fp16" else np.float32)
        inp1 = img1[None, ...].astype(np.float16 if args.dtype == "fp16" else np.float32)
        scores0, _ = sp_sess.run(None, {"image": inp0})
        scores1, _ = sp_sess.run(None, {"image": inp1})
        scores0 = scores0.astype(np.float32)[0, 0]
        scores1 = scores1.astype(np.float32)[0, 0]

        score0 = extract_patches_gray(scores0[None, ...], mkpts0)  # treat as 1xHxW
        score1 = extract_patches_gray(scores1[None, ...], mkpts1)
        if args.dtype == "fp16":
            score0_i = score0.astype(np.float16)
            score1_i = score1.astype(np.float16)
        else:
            score0_i, score1_i = score0, score1

        (delta0,) = ref_sess.run(None, {"patch": patch0_i, "scorepatch": score0_i, "desc_mean": desc_mean_i})
        (delta1,) = ref_sess.run(None, {"patch": patch1_i, "scorepatch": score1_i, "desc_mean": desc_mean_i})
    else:
        (delta0,) = ref_sess.run(None, {"patch": patch0_i, "desc_mean": desc_mean_i})
        (delta1,) = ref_sess.run(None, {"patch": patch1_i, "desc_mean": desc_mean_i})

    delta0 = delta0.astype(np.float32)[0]
    delta1 = delta1.astype(np.float32)[0]

    sub_mkpts0 = mkpts0 + delta0
    sub_mkpts1 = mkpts1 + delta1

    # Rescale to original image sizes for visualization.
    sx0, sy0 = img0_rgb.shape[1] / float(base_w), img0_rgb.shape[0] / float(base_h)
    sx1, sy1 = img1_rgb.shape[1] / float(base_w), img1_rgb.shape[0] / float(base_h)
    sub_mkpts0_o = sub_mkpts0 * np.array([sx0, sy0], dtype=np.float32)
    sub_mkpts1_o = sub_mkpts1 * np.array([sx1, sy1], dtype=np.float32)

    r = BaseMatcher(device="cpu")
    H, in0, in1 = r.compute_ransac(sub_mkpts0_o, sub_mkpts1_o)

    out = {
        **res,
        "matched_kpts0": sub_mkpts0_o,
        "matched_kpts1": sub_mkpts1_o,
        "inlier_kpts0": in0,
        "inlier_kpts1": in1,
        "num_inliers": int(len(in0)),
        "H": H,
    }
    out["latency_ms_base"] = (t1 - t0) * 1000.0
    return out, (img0_rgb, img1_rgb)


def draw_and_save(out: dict, img0_rgb: np.ndarray, img1_rgb: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    in0 = out.get("inlier_kpts0", np.empty((0, 2), dtype=np.float32))
    in1 = out.get("inlier_kpts1", np.empty((0, 2), dtype=np.float32))

    img0_bgr = cv2.cvtColor(img0_rgb, cv2.COLOR_RGB2BGR)
    img1_bgr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
    kp0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in in0]
    kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in in1]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(kp0))]
    vis = cv2.drawMatches(img0_bgr, kp0, img1_bgr, kp1, matches, None, matchColor=(0, 255, 0), flags=2)
    cv2.imwrite(str(out_dir / "result.jpg"), vis)

    lines = []
    lines.append(f"matcher: {out_dir.name}")
    lines.append(f"num_matches: {len(out['matched_kpts0'])}")
    lines.append(f"num_inliers: {out['num_inliers']}")
    lines.append(f"base_latency_ms: {out.get('latency_ms_base', 0.0):.1f}")
    (out_dir / "result.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SubPX ONNX demo (base matcher + ONNX subpixel refiner)")
    p.add_argument(
        "--matcher",
        required=True,
        choices=["xfeat-subpx", "xfeat-lightglue-subpx", "superpoint-lightglue-subpx"],
    )
    p.add_argument("--img1", type=str, default="assets/ref.png")
    p.add_argument("--img2", type=str, default="assets/tgt.png")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--size", nargs=2, type=int, default=[640, 480], help="Width Height")
    p.add_argument("--top-k", type=int, default=1024)
    p.add_argument("--output", choices=["yes", "no"], default="yes")
    args = p.parse_args()

    args.size = (int(args.size[0]), int(args.size[1]))

    out, imgs = run_demo(args)
    print(f"matches={len(out['matched_kpts0'])} inliers={out['num_inliers']} base_ms={out.get('latency_ms_base', 0.0):.1f}")

    if args.output == "yes" and imgs is not None:
        stem1 = Path(args.img1).stem
        stem2 = Path(args.img2).stem
        out_dir = ROOT / "outputs" / "matching-onnx" / f"{args.matcher}_{args.dtype}_{stem1}_{stem2}"
        draw_and_save(out, imgs[0], imgs[1], out_dir)
        print(f"saved: {out_dir}")
