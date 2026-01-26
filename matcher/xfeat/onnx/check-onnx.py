import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "matcher-onnx"))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _providers():
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _onnx_dir() -> Path:
    return ROOT / "matcher-onnx" / "weights" / "xfeat"


def _pth_dir() -> Path:
    return ROOT / "matcher" / "xfeat" / "weights"


def _load_pair_resized(img1: str, img2: str, size_wh: tuple[int, int]):
    W, H = size_wh

    def load_one(p: str):
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Failed to load image: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))
        x_np = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        x_t = torch.from_numpy(x_np)
        return x_np, x_t

    n0, t0 = load_one(img1)
    n1, t1 = load_one(img2)
    return (n0, n1), (t0, t1)


def check_xfeat_backbone(size=(640, 480), dtype="FP32"):
    W, H = size
    pth_path = _pth_dir() / "xfeat.pt"
    onnx_path = _onnx_dir() / f"xfeat_backbone_{dtype.lower()}_{W}x{H}.onnx"
    if not pth_path.exists():
        raise FileNotFoundError(f"Missing weights: {pth_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX: {onnx_path}")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, H, W, device=device)

    from matcher.xfeat.modules.model import XFeatModel

    net = XFeatModel().to(device).eval()
    net.load_state_dict(torch.load(str(pth_path), map_location="cpu"))
    with torch.no_grad():
        t_desc, t_kpt, t_rel = net(x)
        t_desc = F.normalize(t_desc, dim=1)

    x_np = x.detach().cpu().numpy()
    if dtype == "FP16":
        x_np = x_np.astype(np.float16)

    sess = ort.InferenceSession(str(onnx_path), providers=_providers())
    o_desc, o_kpt, o_rel = sess.run(None, {"image": x_np})

    # Ensure comparable types
    o_desc = o_desc.astype(np.float32)
    o_kpt = o_kpt.astype(np.float32)
    o_rel = o_rel.astype(np.float32)

    t_desc_np = t_desc.detach().cpu().numpy()
    t_kpt_np = t_kpt.detach().cpu().numpy()
    t_rel_np = t_rel.detach().cpu().numpy()

    print(f"\n=== XFeat Backbone Check ({dtype}, {W}x{H}) ===")
    print(f"Descriptor map MSE: {mse(t_desc_np, o_desc):.6e} | max_abs: {max_abs(t_desc_np, o_desc):.6e}")
    print(f"Kpt logits MSE:     {mse(t_kpt_np, o_kpt):.6e} | max_abs: {max_abs(t_kpt_np, o_kpt):.6e}")
    print(f"Reliability MSE:    {mse(t_rel_np, o_rel):.6e} | max_abs: {max_abs(t_rel_np, o_rel):.6e}")


def check_lighterglue(size=(640, 480), dtype="FP32", num_kpts=1024):
    W, H = size
    pth_path = _pth_dir() / "xfeat-lighterglue.pt"
    onnx_path = _onnx_dir() / f"xfeat_lighterglue_{dtype.lower()}_k{num_kpts}.onnx"

    if not pth_path.exists():
        raise FileNotFoundError(f"Missing weights: {pth_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX: {onnx_path}")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float16 if dtype == "FP16" else torch.float32

    # Build the same matcher as convert-onnx.py
    from kornia.feature.lightglue import LightGlue

    conf = {
        "name": "xfeat",
        "input_dim": 64,
        "descriptor_dim": 96,
        "add_scale_ori": False,
        "add_laf": False,
        "scale_coef": 1.0,
        "n_layers": 6,
        "num_heads": 1,
        "flash": False,
        "mp": False,
        "depth_confidence": -1,
        "width_confidence": -1,
        "filter_threshold": 0.1,
        "weights": None,
    }

    LightGlue.default_conf = conf
    lg = LightGlue(None).to(device).eval()

    state_dict = torch.load(str(pth_path), map_location="cpu")
    for i in range(lg.conf.n_layers):
        pattern = (f"self_attn.{i}", f"transformers.{i}.self_attn")
        state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        pattern = (f"cross_attn.{i}", f"transformers.{i}.cross_attn")
        state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        state_dict = {k.replace("matcher.", ""): v for k, v in state_dict.items()}
    lg.load_state_dict(state_dict, strict=False)

    if dtype == "FP16":
        lg = lg.half()

    def core_log_assignment(keypoints0, descriptors0, keypoints1, descriptors1, image_size0, image_size1):
        from kornia.feature.lightglue import normalize_keypoints

        kpts0 = normalize_keypoints(keypoints0, image_size0).clone()
        kpts1 = normalize_keypoints(keypoints1, image_size1).clone()

        desc0 = descriptors0.detach().contiguous()
        desc1 = descriptors1.detach().contiguous()
        desc0 = lg.input_proj(desc0)
        desc1 = lg.input_proj(desc1)
        encoding0 = lg.posenc(kpts0)
        encoding1 = lg.posenc(kpts1)

        for i in range(lg.conf.n_layers):
            desc0, desc1 = lg.transformers[i](desc0, desc1, encoding0, encoding1)

        la = lg.log_assignment[lg.conf.n_layers - 1]
        mdesc0, mdesc1 = la.final_proj(desc0), la.final_proj(desc1)
        d = mdesc0.shape[-1]
        mdesc0, mdesc1 = mdesc0 / (d**0.25), mdesc1 / (d**0.25)
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = la.matchability(desc0)
        z1 = la.matchability(desc1)

        certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
        scores0 = F.log_softmax(sim, 2)
        scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
        s00 = scores0 + scores1 + certainties

        last_col = F.logsigmoid(-z0.squeeze(-1))
        last_row = F.logsigmoid(-z1.squeeze(-1))
        top = torch.cat([s00, last_col.unsqueeze(-1)], dim=2)
        bottom_right = sim.new_zeros((sim.shape[0], 1, 1))
        bottom = torch.cat([last_row.unsqueeze(1), bottom_right], dim=2)
        return torch.cat([top, bottom], dim=1)

    wh = torch.tensor([W, H], device=device, dtype=dt)
    k0 = torch.rand(1, num_kpts, 2, device=device, dtype=dt) * wh
    k1 = torch.rand(1, num_kpts, 2, device=device, dtype=dt) * wh
    d0 = F.normalize(torch.randn(1, num_kpts, 64, device=device, dtype=dt), dim=-1)
    d1 = F.normalize(torch.randn(1, num_kpts, 64, device=device, dtype=dt), dim=-1)
    s0 = torch.tensor([[W, H]], device=device, dtype=torch.int64)
    s1 = torch.tensor([[W, H]], device=device, dtype=torch.int64)

    with torch.no_grad():
        t_log = core_log_assignment(k0, d0, k1, d1, s0, s1).float().cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=_providers())
    inputs = {
        "keypoints0": k0.float().cpu().numpy() if dtype == "FP32" else k0.cpu().numpy(),
        "descriptors0": d0.float().cpu().numpy() if dtype == "FP32" else d0.cpu().numpy(),
        "keypoints1": k1.float().cpu().numpy() if dtype == "FP32" else k1.cpu().numpy(),
        "descriptors1": d1.float().cpu().numpy() if dtype == "FP32" else d1.cpu().numpy(),
        "image_size0": s0.cpu().numpy(),
        "image_size1": s1.cpu().numpy(),
    }
    (o_log,) = sess.run(None, inputs)
    o_log = o_log.astype(np.float32)

    print(f"\n=== LighterGlue Check ({dtype}, k={num_kpts}) ===")
    print(f"log_assignment MSE: {mse(t_log, o_log):.6e} | max_abs: {max_abs(t_log, o_log):.6e}")


def check_finematcher(dtype: str = "FP32", num_pairs: int = 1024):
    pth_path = _pth_dir() / "xfeat.pt"
    onnx_path = _onnx_dir() / f"xfeat_finematcher_{dtype.lower()}.onnx"
    if not pth_path.exists():
        raise FileNotFoundError(f"Missing weights: {pth_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX: {onnx_path}")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float16 if dtype == "FP16" else torch.float32

    from matcher.xfeat.modules.model import XFeatModel

    net = XFeatModel().to(device).eval()
    net.load_state_dict(torch.load(str(pth_path), map_location="cpu"))
    if dtype == "FP16":
        net = net.half()

    pairs = torch.randn(1, int(num_pairs), 128, device=device, dtype=dt)
    with torch.no_grad():
        t_out = net.fine_matcher(pairs.reshape(-1, 128)).reshape(1, int(num_pairs), 64)
        t_out = t_out.float().cpu().numpy()

    pairs_np = pairs.cpu().numpy() if dtype == "FP16" else pairs.float().cpu().numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=_providers())
    (o_out,) = sess.run(None, {"pairs": pairs_np})
    o_out = o_out.astype(np.float32)

    print(f"\n=== XFeat FineMatcher Check ({dtype}, num_pairs={int(num_pairs)}) ===")
    print(f"offset_logits MSE:  {mse(t_out, o_out):.6e} | max_abs: {max_abs(t_out, o_out):.6e}")


def run_end_to_end(img1: str, img2: str, size=(640, 480), top_k: int = 1024, device: str | None = None):
    # Force same resize for torch and onnx so coords are comparable.
    (n0, n1), (t0, t1) = _load_pair_resized(img1, img2, size)

    dev = device
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    from matcher.xfeat import XFeatMatcher
    from xfeat import XFeatONNXMatcher

    print("\n=== End-to-End Matching (resized inputs) ===")

    def _timed(call):
        t0s = time.time()
        out = call()
        t1s = time.time()
        return out, (t1s - t0s) * 1000

    # --- PyTorch baselines ---
    for mode in ["xfeat", "xfeat-star", "xfeat-lightglue"]:
        m = XFeatMatcher(device=dev, mode=mode, max_num_keypoints=top_k)
        out, ms = _timed(lambda: m(t0.to(dev), t1.to(dev)))
        print(
            f"torch {mode:15s} | k0={len(out['all_kpts0']):5d} k1={len(out['all_kpts1']):5d} "
            f"matches={len(out['matched_kpts0']):5d} inliers={out.get('num_inliers', 0):5d} | {ms:6.1f} ms"
        )

    # --- ONNX (xfeat / xfeat-lightglue) ---
    # Note: torch XFeatMatcher(xfeat) uses min_cossim=-1 by default.
    for dtype in ["fp32", "fp16"]:
        m = XFeatONNXMatcher(device=dev, mode="xfeat", dtype=dtype, size=size, top_k=top_k, min_cossim=-1.0)
        out, ms = _timed(lambda: m(n0, n1))
        print(
            f"onnx xfeat ({dtype})     | k0={len(out['all_kpts0']):5d} k1={len(out['all_kpts1']):5d} "
            f"matches={len(out['matched_kpts0']):5d} inliers={out.get('num_inliers', 0):5d} | {ms:6.1f} ms"
        )

    for dtype in ["fp32", "fp16"]:
        m = XFeatONNXMatcher(device=dev, mode="lightglue", dtype=dtype, size=size, top_k=top_k)
        out, ms = _timed(lambda: m(n0, n1))
        print(
            f"onnx xfeat-lightglue ({dtype}) | k0={len(out['all_kpts0']):5d} k1={len(out['all_kpts1']):5d} "
            f"matches={len(out['matched_kpts0']):5d} inliers={out.get('num_inliers', 0):5d} | {ms:6.1f} ms"
        )

    for dtype in ["fp32", "fp16"]:
        try:
            m = XFeatONNXMatcher(device=dev, mode="star", dtype=dtype, size=size, top_k=top_k)
            out, ms = _timed(lambda: m(n0, n1))
            print(
                f"onnx xfeat-star ({dtype}) | k0={len(out['all_kpts0']):5d} k1={len(out['all_kpts1']):5d} "
                f"matches={len(out['matched_kpts0']):5d} inliers={out.get('num_inliers', 0):5d} | {ms:6.1f} ms"
            )
        except FileNotFoundError as e:
            print(f"onnx xfeat-star ({dtype}) | missing weights: {e}")


def main():
    parser = argparse.ArgumentParser(description="XFeat ONNX vs PyTorch checks")
    parser.add_argument(
        "--matcher",
        choices=["all", "xfeat", "xfeat-star", "xfeat-lightglue", "backbone", "finematcher", "lightglue"],
        default="all",
    )
    parser.add_argument("--dtype", choices=["FP32", "FP16", "BOTH"], default="BOTH")
    parser.add_argument("--size", nargs=2, type=int, default=[640, 480], help="Width Height")
    parser.add_argument("--num-kpts", type=int, default=1024, help="Keypoints for LightGlue and ONNX pipelines")
    parser.add_argument("--num-pairs", type=int, default=1024, help="Pairs for finematcher check")
    parser.add_argument("--img1", type=str, default="assets/ref.png")
    parser.add_argument("--img2", type=str, default="assets/tgt.png")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda (default: auto)")
    args = parser.parse_args()

    dtypes = [args.dtype] if args.dtype in ("FP32", "FP16") else ["FP32", "FP16"]
    size = (int(args.size[0]), int(args.size[1]))

    if args.matcher in ("all", "backbone", "xfeat", "xfeat-star", "xfeat-lightglue"):
        for dt in dtypes:
            check_xfeat_backbone(size=size, dtype=dt)

    if args.matcher in ("all", "finematcher", "xfeat-star"):
        for dt in dtypes:
            check_finematcher(dtype=dt, num_pairs=int(args.num_pairs))

    if args.matcher in ("all", "lightglue", "xfeat-lightglue"):
        for dt in dtypes:
            check_lighterglue(size=size, dtype=dt, num_kpts=int(args.num_kpts))

    if args.matcher in ("all", "xfeat", "xfeat-star", "xfeat-lightglue"):
        run_end_to_end(args.img1, args.img2, size=size, top_k=int(args.num_kpts), device=args.device)


if __name__ == "__main__":
    main()
