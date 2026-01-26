import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort


sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


class SuperPointBackboneBaseline(nn.Module):
    def __init__(self, weights_path: str):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)

        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = nn.Conv2d(c5, 256, 1, 1, 0)

        self.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.eval()

    def forward(self, image: torch.Tensor):
        if image.shape[1] == 3:
            r = image[:, 0:1]
            g = image[:, 1:2]
            b = image[:, 2:3]
            image = 0.2989 * r + 0.5870 * g + 0.1140 * b

        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        scores = torch.softmax(semi, dim=1)[:, :-1]
        B, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(B, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(B, 1, h * 8, w * 8)

        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        desc = F.normalize(desc, p=2, dim=1)

        return scores.float(), desc.float()


def build_lightglue_baseline(weights_path: str):
    from matcher.lightglue.modules.lightglue import LightGlue

    lg = LightGlue(
        features=None,  # type: ignore[arg-type]
        input_dim=256,
        descriptor_dim=256,
        add_scale_ori=False,
        n_layers=9,
        num_heads=4,
        flash=False,
        mp=False,
        depth_confidence=-1,
        width_confidence=-1,
        filter_threshold=0.1,
    ).eval()

    state_dict = torch.load(weights_path, map_location="cpu")
    new_state = {}
    for k, v in state_dict.items():
        k = k.replace("matcher.", "")
        for i in range(lg.conf.n_layers):
            k = k.replace(f"self_attn.{i}.", f"transformers.{i}.self_attn.")
            k = k.replace(f"cross_attn.{i}.", f"transformers.{i}.cross_attn.")
        new_state[k] = v
    lg.load_state_dict(new_state, strict=False)
    return lg


def normalize_keypoints(kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    return (kpts - shift[..., None, :]) / scale[..., None, None]


def lightglue_log_assignment(
    lg,
    keypoints0: torch.Tensor,
    descriptors0: torch.Tensor,
    keypoints1: torch.Tensor,
    descriptors1: torch.Tensor,
    image_size0: torch.Tensor,
    image_size1: torch.Tensor,
):
    k0 = normalize_keypoints(keypoints0, image_size0).clone()
    k1 = normalize_keypoints(keypoints1, image_size1).clone()
    d0 = lg.input_proj(descriptors0.detach().contiguous())
    d1 = lg.input_proj(descriptors1.detach().contiguous())
    e0 = lg.posenc(k0)
    e1 = lg.posenc(k1)
    for i in range(lg.conf.n_layers):
        d0, d1 = lg.transformers[i](d0, d1, e0, e1)

    la = lg.log_assignment[lg.conf.n_layers - 1]
    m0, m1 = la.final_proj(d0), la.final_proj(d1)
    d = m0.shape[-1]
    m0, m1 = m0 / (d**0.25), m1 / (d**0.25)
    sim = torch.einsum("bmd,bnd->bmn", m0, m1)
    z0 = la.matchability(d0)
    z1 = la.matchability(d1)

    cert = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    s0 = F.log_softmax(sim, 2)
    s1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    s00 = s0 + s1 + cert
    last_col = F.logsigmoid(-z0.squeeze(-1))
    last_row = F.logsigmoid(-z1.squeeze(-1))
    top = torch.cat([s00, last_col.unsqueeze(-1)], dim=2)
    bottom_right = sim.new_zeros((sim.shape[0], 1, 1))
    bottom = torch.cat([last_row.unsqueeze(1), bottom_right], dim=2)
    return torch.cat([top, bottom], dim=1)


def main():
    parser = argparse.ArgumentParser(description="SuperPoint+LightGlue ONNX check")
    parser.add_argument("--size", nargs=2, type=int, default=[640, 480], help="Width Height")
    parser.add_argument("--num-kpts", type=int, default=1024)
    args = parser.parse_args()

    W, H = args.size
    n = int(args.num_kpts)
    weights_dir = Path(__file__).parent.parent / "weights"

    sp_pth = weights_dir / "superpoint_v1.pth"
    lg_pth = weights_dir / "superpoint_lightglue.pth"
    sp_onnx32 = weights_dir / f"superpoint_backbone_fp32_{W}x{H}.onnx"
    sp_onnx16 = weights_dir / f"superpoint_backbone_fp16_{W}x{H}.onnx"
    lg_onnx32 = weights_dir / f"superpoint_lightglue_fp32_k{n}.onnx"
    lg_onnx16 = weights_dir / f"superpoint_lightglue_fp16_k{n}.onnx"

    for p in [sp_pth, lg_pth, sp_onnx32, lg_onnx32]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    has_fp16 = sp_onnx16.exists() and lg_onnx16.exists()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # ------------------------------------------------------------------
    # SuperPoint backbone
    # ------------------------------------------------------------------
    x = torch.randn(1, 3, H, W, device=device)
    sp = SuperPointBackboneBaseline(str(sp_pth)).to(device).eval()
    with torch.no_grad():
        t_s32, t_d32 = sp(x)
        t_s16, t_d16 = sp.half()(x.half())

    sess_sp32 = ort.InferenceSession(str(sp_onnx32), providers=providers)
    sess_sp16 = ort.InferenceSession(str(sp_onnx16), providers=providers) if sp_onnx16.exists() else None
    o_s32, o_d32 = sess_sp32.run(None, {"image": x.detach().cpu().numpy().astype(np.float32)})
    if sess_sp16 is not None:
        o_s16, o_d16 = sess_sp16.run(None, {"image": x.detach().cpu().numpy().astype(np.float16)})
    else:
        o_s16, o_d16 = None, None

    # ------------------------------------------------------------------
    # LightGlue core
    # ------------------------------------------------------------------
    lg = build_lightglue_baseline(str(lg_pth)).to(device)
    lg16 = build_lightglue_baseline(str(lg_pth)).to(device).half() if has_fp16 else None

    wh = torch.tensor([W, H], device=device, dtype=torch.float32)
    k0 = torch.rand(1, n, 2, device=device) * wh
    k1 = torch.rand(1, n, 2, device=device) * wh
    d0 = F.normalize(torch.randn(1, n, 256, device=device), dim=-1)
    d1 = F.normalize(torch.randn(1, n, 256, device=device), dim=-1)
    s0 = torch.tensor([[W, H]], device=device, dtype=torch.float32)
    s1 = torch.tensor([[W, H]], device=device, dtype=torch.float32)

    with torch.no_grad():
        t_l32 = lightglue_log_assignment(lg, k0, d0, k1, d1, s0, s1)
        if lg16 is not None:
            t_l16 = lightglue_log_assignment(lg16, k0.half(), d0.half(), k1.half(), d1.half(), s0.half(), s1.half())
        else:
            t_l16 = None

    sess_lg32 = ort.InferenceSession(str(lg_onnx32), providers=providers)
    sess_lg16 = ort.InferenceSession(str(lg_onnx16), providers=providers) if lg_onnx16.exists() else None

    (o_l32,) = sess_lg32.run(
        None,
        {
            "keypoints0": k0.detach().cpu().numpy().astype(np.float32),
            "descriptors0": d0.detach().cpu().numpy().astype(np.float32),
            "keypoints1": k1.detach().cpu().numpy().astype(np.float32),
            "descriptors1": d1.detach().cpu().numpy().astype(np.float32),
            "image_size0": np.array([[W, H]], dtype=np.int64),
            "image_size1": np.array([[W, H]], dtype=np.int64),
        },
    )
    if sess_lg16 is not None:
        (o_l16,) = sess_lg16.run(
            None,
            {
                "keypoints0": k0.detach().cpu().numpy().astype(np.float16),
                "descriptors0": d0.detach().cpu().numpy().astype(np.float16),
                "keypoints1": k1.detach().cpu().numpy().astype(np.float16),
                "descriptors1": d1.detach().cpu().numpy().astype(np.float16),
                "image_size0": np.array([[W, H]], dtype=np.int64),
                "image_size1": np.array([[W, H]], dtype=np.int64),
            },
        )
    else:
        o_l16 = None

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n=== Comparing SuperPoint+LightGlue ({W}x{H}, k={n}) ===")
    print("\n-----------------------------------------------")
    print("Metric                        | FP32 MSE       | FP16 MSE")
    print("-----------------------------------------------")
    fp32_sp_s = mse(t_s32.cpu().numpy(), o_s32.astype(np.float32))
    fp32_sp_d = mse(t_d32.cpu().numpy(), o_d32.astype(np.float32))
    fp32_lg = mse(t_l32.cpu().numpy(), o_l32.astype(np.float32))

    if o_s16 is not None and o_d16 is not None:
        fp16_sp_s = mse(t_s16.float().cpu().numpy(), o_s16.astype(np.float32))
        fp16_sp_d = mse(t_d16.float().cpu().numpy(), o_d16.astype(np.float32))
    else:
        fp16_sp_s = float("nan")
        fp16_sp_d = float("nan")

    if t_l16 is not None and o_l16 is not None:
        fp16_lg = mse(t_l16.float().cpu().numpy(), o_l16.astype(np.float32))
    else:
        fp16_lg = float("nan")

    print(f"SuperPoint scores             | {fp32_sp_s:.6e} | {fp16_sp_s:.6e}")
    print(f"SuperPoint descriptors_map    | {fp32_sp_d:.6e} | {fp16_sp_d:.6e}")
    print(f"LightGlue log_assignment      | {fp32_lg:.6e} | {fp16_lg:.6e}")

    if not has_fp16:
        print("\nNote: FP16 check skipped (missing FP16 ONNX files for this size/k).")
    print("-----------------------------------------------")


if __name__ == "__main__":
    main()
