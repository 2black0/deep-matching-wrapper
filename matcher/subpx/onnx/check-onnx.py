import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _providers():
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _pth_dir() -> Path:
    return ROOT / "matcher" / "subpx" / "weights"


def _onnx_dir() -> Path:
    return ROOT / "matcher-onnx" / "weights" / "subpx"


def _session(path: Path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=_providers())


def check_xfeat_refiner(dtype: str = "FP32", num_kpts: int = 1024):
    onnx_path = _onnx_dir() / f"k2s_xfeat_refiner_{dtype.lower()}.onnx"
    w_path = _pth_dir() / "k2s_xfeat_pretrained.pth"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX: {onnx_path}")
    if not w_path.exists():
        raise FileNotFoundError(f"Missing weights: {w_path}")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float16 if dtype == "FP16" else torch.float32

    from matcher.subpx.modules.keypt2subpx import Keypt2Subpx

    state = torch.load(str(w_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    m = Keypt2Subpx(output_dim=64, use_score=False).to(device).eval()
    m.net.load_state_dict(state)
    if dtype == "FP16":
        m = m.half()

    patch = torch.randn(1, int(num_kpts), 1, 11, 11, device=device, dtype=dt)
    desc_mean = torch.randn(1, int(num_kpts), 64, device=device, dtype=dt)

    with torch.no_grad():
        t_delta = (m.net(patch, scorepatch=None, desc=desc_mean) * 2.5).float().cpu().numpy()

    sess = _session(onnx_path)
    inputs = {
        "patch": patch.detach().cpu().numpy(),
        "desc_mean": desc_mean.detach().cpu().numpy(),
    }
    (o_delta,) = sess.run(None, inputs)
    o_delta = o_delta.astype(np.float32)

    print(f"\n=== SubPX XFeat refiner ({dtype}, N={int(num_kpts)}) ===")
    print(f"delta MSE:     {mse(t_delta, o_delta):.6e} | max_abs: {max_abs(t_delta, o_delta):.6e}")


def check_splg_refiner(dtype: str = "FP32", num_kpts: int = 1024):
    onnx_path = _onnx_dir() / f"k2s_splg_refiner_{dtype.lower()}.onnx"
    w_path = _pth_dir() / "k2s_splg_pretrained.pth"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX: {onnx_path}")
    if not w_path.exists():
        raise FileNotFoundError(f"Missing weights: {w_path}")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float16 if dtype == "FP16" else torch.float32

    from matcher.subpx.modules.keypt2subpx import Keypt2Subpx

    state = torch.load(str(w_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    m = Keypt2Subpx(output_dim=256, use_score=True).to(device).eval()
    m.net.load_state_dict(state)
    if dtype == "FP16":
        m = m.half()

    patch = torch.randn(1, int(num_kpts), 1, 11, 11, device=device, dtype=dt)
    scorepatch = torch.randn(1, int(num_kpts), 1, 11, 11, device=device, dtype=dt)
    desc_mean = torch.randn(1, int(num_kpts), 256, device=device, dtype=dt)

    with torch.no_grad():
        t_delta = (m.net(patch, scorepatch=scorepatch, desc=desc_mean) * 2.5).float().cpu().numpy()

    sess = _session(onnx_path)
    inputs = {
        "patch": patch.detach().cpu().numpy(),
        "scorepatch": scorepatch.detach().cpu().numpy(),
        "desc_mean": desc_mean.detach().cpu().numpy(),
    }
    (o_delta,) = sess.run(None, inputs)
    o_delta = o_delta.astype(np.float32)

    print(f"\n=== SubPX SuperPoint+LightGlue refiner ({dtype}, N={int(num_kpts)}) ===")
    print(f"delta MSE:     {mse(t_delta, o_delta):.6e} | max_abs: {max_abs(t_delta, o_delta):.6e}")


def main():
    parser = argparse.ArgumentParser(description="Keypt2Subpx (SubPX) ONNX check")
    parser.add_argument(
        "--matcher",
        choices=["xfeat-subpx", "superpoint-lightglue-subpx", "all"],
        default="all",
    )
    parser.add_argument("--dtype", choices=["FP32", "FP16", "BOTH"], default="BOTH")
    parser.add_argument("--num-kpts", type=int, default=1024)
    args = parser.parse_args()

    dtypes = [args.dtype] if args.dtype in ("FP32", "FP16") else ["FP32", "FP16"]

    if args.matcher in ("xfeat-subpx", "all"):
        for dt in dtypes:
            check_xfeat_refiner(dtype=dt, num_kpts=int(args.num_kpts))

    if args.matcher in ("superpoint-lightglue-subpx", "all"):
        for dt in dtypes:
            check_splg_refiner(dtype=dt, num_kpts=int(args.num_kpts))


if __name__ == "__main__":
    main()
