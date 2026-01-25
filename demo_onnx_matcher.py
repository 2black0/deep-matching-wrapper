import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# Add matcher-onnx folder to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "matcher-onnx"))

from base_matcher import get_matcher, AVAILABLE_MATCHERS


def load_image(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return x


def test_matcher(
    matcher_name: str,
    img0_path: Path | None = None,
    img1_path: Path | None = None,
    output_enabled: bool = False,
    device: str = "cpu",
    dtype: str = "fp32",
):
    log_lines: list[str] = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    try:
        log(f"\n==================== Testing {matcher_name} (onnx) ====================")

        matcher = get_matcher(matcher_name, device=device, dtype=dtype)

        if img0_path and img1_path:
            log(f"Loading images: {img0_path} and {img1_path}")
            img0 = load_image(img0_path)
            img1 = load_image(img1_path)
        else:
            log("Using random noise (512x512)")
            img0 = np.random.rand(3, 512, 512).astype(np.float32)
            img1 = np.random.rand(3, 512, 512).astype(np.float32)

        # Warm-up
        _ = matcher(img0, img1)

        start = time.time()
        res = matcher(img0, img1)
        end = time.time()
        latency_ms = (end - start) * 1000

        mkpts0 = res["matched_kpts0"]
        mkpts1 = res["matched_kpts1"]
        kpts0 = res["all_kpts0"]
        kpts1 = res["all_kpts1"]
        desc0 = res["all_desc0"]
        desc1 = res["all_desc1"]
        inliers0 = res.get("inlier_kpts0")
        inliers1 = res.get("inlier_kpts1")

        n_matches = int(len(mkpts0)) if mkpts0 is not None else 0
        n_inliers = int(res.get("num_inliers", 0))
        ratio = n_inliers / n_matches if n_matches > 0 else 0.0

        def get_type_str(x):
            if isinstance(x, np.ndarray):
                return f"(numpy.ndarray, dtype={x.dtype}, shape={x.shape})"
            return f"({type(x).__name__})"

        def format_sample(pts: np.ndarray):
            if pts is None or len(pts) == 0:
                return "[]"
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

        if len(kpts0) > 0:
            log(f"  All Keypoints0: {format_sample(kpts0)}, ... {get_type_str(kpts0)}")
        else:
            log(f"  All Keypoints0: [] {get_type_str(kpts0)}")

        if len(kpts1) > 0:
            log(f"  All Keypoints1: {format_sample(kpts1)}, ... {get_type_str(kpts1)}")
        else:
            log(f"  All Keypoints1: [] {get_type_str(kpts1)}")

        if isinstance(desc0, np.ndarray) and desc0.size > 0:
            sample = f"[{desc0[0][:4]}...]" if desc0.ndim == 2 and desc0.shape[1] > 4 else f"[{desc0[0]}]"
            log(f"  All Descriptors0: {sample}, ... {get_type_str(desc0)}")
        else:
            log(f"  All Descriptors0: [] {get_type_str(desc0)}")

        if isinstance(desc1, np.ndarray) and desc1.size > 0:
            sample = f"[{desc1[0][:4]}...]" if desc1.ndim == 2 and desc1.shape[1] > 4 else f"[{desc1[0]}]"
            log(f"  All Descriptors1: {sample}, ... {get_type_str(desc1)}")
        else:
            log(f"  All Descriptors1: [] {get_type_str(desc1)}")

        if n_matches > 0:
            log(f"  Matched Keypoints0: {format_sample(mkpts0)}, ... {get_type_str(mkpts0)}")
            log(f"  Matched Keypoints1: {format_sample(mkpts1)}, ... {get_type_str(mkpts1)}")

        log(f"\nTime: {latency_ms:.0f} ms")
        log("=======================================================")

        if output_enabled and img0_path and img1_path:
            stem1 = img0_path.stem
            stem2 = img1_path.stem
            output_dir = Path(f"outputs/matching-onnx/{matcher_name}_{dtype}_{stem1}_{stem2}")
            output_dir.mkdir(parents=True, exist_ok=True)

            (output_dir / "result.txt").write_text("\n".join(log_lines))

            if inliers0 is not None and len(inliers0) > 0:
                img1_cv = cv2.imread(str(img0_path))
                img2_cv = cv2.imread(str(img1_path))
                kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in inliers0]
                kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in inliers1]
                matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]
                out_img = cv2.drawMatches(
                    img1_cv,
                    kp1,
                    img2_cv,
                    kp2,
                    matches,
                    None,
                    matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    flags=2,
                )
                cv2.imwrite(str(output_dir / "result.jpg"), out_img)
                print(f"Saved output to {output_dir}")

    except Exception:
        print(f"FAILED: {matcher_name}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matcher",
        type=str,
        required=True,
        choices=(AVAILABLE_MATCHERS + ["all"]),
        help=f"Name of the matcher. Supported: {', '.join(AVAILABLE_MATCHERS)} or 'all' to run all",
    )
    parser.add_argument("--img1", type=str, default="assets/ref.png")
    parser.add_argument("--img2", type=str, default="assets/tgt.png")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--output", type=str, choices=["yes", "no"], default="no")
    args = parser.parse_args()

    img1_path = Path(args.img1)
    img2_path = Path(args.img2)
    output_yes = args.output == "yes"

    if not img1_path.exists() or not img2_path.exists():
        print("Warning: images not found, using random noise")
        img1_path = None
        img2_path = None

    if args.matcher == "all":
        print("Running all ONNX matchers...")
        for m in AVAILABLE_MATCHERS:
            test_matcher(m, img1_path, img2_path, output_yes, device=args.device, dtype=args.dtype)
    else:
        test_matcher(args.matcher, img1_path, img2_path, output_yes, device=args.device, dtype=args.dtype)
