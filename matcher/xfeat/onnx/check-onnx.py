import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort


sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def check_xfeat(size=(640, 480), dtype="FP32"):
    W, H = size
    weights_dir = Path(__file__).parent.parent / "weights"

    pth_path = weights_dir / "xfeat.pt"
    onnx_path = weights_dir / f"xfeat_backbone_{dtype.lower()}_{W}x{H}.onnx"
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

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    o_desc, o_kpt, o_rel = sess.run(None, {"image": x_np})

    # Ensure comparable types
    o_desc = o_desc.astype(np.float32)
    o_kpt = o_kpt.astype(np.float32)
    o_rel = o_rel.astype(np.float32)

    print(f"\n=== XFeat Backbone Check ({dtype}, {W}x{H}) ===")
    print(f"Descriptor map MSE: {mse(t_desc.cpu().numpy(), o_desc):.6e}")
    print(f"Kpt logits MSE:     {mse(t_kpt.cpu().numpy(), o_kpt):.6e}")
    print(f"Reliability MSE:    {mse(t_rel.cpu().numpy(), o_rel):.6e}")


def check_lightglue(size=(640, 480), dtype="FP32", num_kpts=1024):
    W, H = size
    weights_dir = Path(__file__).parent.parent / "weights"
    pth_path = weights_dir / "xfeat-lighterglue.pt"
    onnx_path = weights_dir / f"xfeat_lighterglue_{dtype.lower()}_k{num_kpts}.onnx"

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

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
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
    print(f"log_assignment MSE: {mse(t_log, o_log):.6e}")


def main():
    parser = argparse.ArgumentParser(description="XFeat / LighterGlue ONNX check")
    parser.add_argument("--matcher", choices=["xfeat", "lightglue"], required=True)
    parser.add_argument("--dtype", choices=["FP32", "FP16"], default="FP32")
    parser.add_argument("--size", nargs=2, type=int, default=[640, 480], help="Width Height")
    parser.add_argument("--num-kpts", type=int, default=1024)
    args = parser.parse_args()

    if args.matcher == "xfeat":
        check_xfeat(tuple(args.size), dtype=args.dtype)
    else:
        check_lightglue(tuple(args.size), dtype=args.dtype, num_kpts=args.num_kpts)


if __name__ == "__main__":
    main()
