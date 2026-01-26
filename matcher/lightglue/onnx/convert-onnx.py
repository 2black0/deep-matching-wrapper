import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.append(str(Path(__file__).parent.parent.parent.parent))


class SuperPointBackboneExport(nn.Module):
    """Export SuperPoint dense outputs.

    Outputs:
      - scores: (B, 1, H, W) dense probability heatmap (pre-NMS)
      - descriptors: (B, 256, H/8, W/8) L2-normalized descriptor map

    Note: sparse post-processing (NMS, border removal, top-k, descriptor sampling)
    is intentionally kept outside ONNX.
    """

    def __init__(self, weights_path: str):
        super().__init__()

        # IMPORTANT: matcher/lightglue/modules/superpoint.py:SuperPoint.__init__
        # downloads weights by default. For ONNX conversion we must be fully offline and
        # strictly load from the local weights path.

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0)

        state = torch.load(weights_path, map_location="cpu")
        self.load_state_dict(state)
        self.eval()

    def forward(self, image: torch.Tensor):
        # image: (B,3,H,W) or (B,1,H,W)
        if image.shape[1] == 3:
            r = image[:, 0:1]
            g = image[:, 1:2]
            b = image[:, 2:3]
            image = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # Shared Encoder
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

        # Dense keypoint scores (pre-NMS)
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        scores = torch.softmax(semi, dim=1)[:, :-1]  # (B,64,h,w)
        B, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(B, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(B, 1, h * 8, w * 8)

        # Dense descriptors
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        desc = F.normalize(desc, p=2, dim=1)

        return scores.float(), desc.float()


class SuperPointLightGlueCoreExport(nn.Module):
    """Export the LightGlue core for SuperPoint.

    This model outputs the continuous log-assignment matrix.
    Discrete matches are computed outside ONNX using `filter_matches`.
    """

    def __init__(self, weights_path: str):
        super().__init__()
        from matcher.lightglue.modules.lightglue import LightGlue

        # Configure for export stability:
        # - disable flash attention
        # - disable pruning/early-stop
        self.lg = LightGlue(
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
        # Match the remapping logic used in matcher/lightglue/__init__.py
        new_state = {}
        for k, v in state_dict.items():
            k = k.replace("matcher.", "")
            for i in range(self.lg.conf.n_layers):
                k = k.replace(f"self_attn.{i}.", f"transformers.{i}.self_attn.")
                k = k.replace(f"cross_attn.{i}.", f"transformers.{i}.cross_attn.")
            new_state[k] = v
        self.lg.load_state_dict(new_state, strict=False)

    @staticmethod
    def _normalize_keypoints(kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        # Matches matcher/lightglue/modules/lightglue.py:normalize_keypoints
        if not isinstance(size, torch.Tensor):
            size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
        size = size.to(kpts)
        shift = size / 2
        scale = size.max(-1).values / 2
        return (kpts - shift[..., None, :]) / scale[..., None, None]

    def forward(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size0: torch.Tensor,
        image_size1: torch.Tensor,
    ):
        # Normalize keypoints
        kpts0 = self._normalize_keypoints(keypoints0, image_size0).clone()
        kpts1 = self._normalize_keypoints(keypoints1, image_size1).clone()

        # Project descriptors
        desc0 = descriptors0.detach().contiguous()
        desc1 = descriptors1.detach().contiguous()
        desc0 = self.lg.input_proj(desc0)
        desc1 = self.lg.input_proj(desc1)

        # Positional encodings
        enc0 = self.lg.posenc(kpts0)
        enc1 = self.lg.posenc(kpts1)

        # Transformer stack (no pruning/early-stop)
        for i in range(self.lg.conf.n_layers):
            desc0, desc1 = self.lg.transformers[i](desc0, desc1, enc0, enc1)

        # MatchAssignment (re-implemented without in-place writes)
        la = self.lg.log_assignment[self.lg.conf.n_layers - 1]
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
        scores = torch.cat([top, bottom], dim=1)

        return scores.float()


def export_superpoint(args) -> Path:
    weights_dir = Path(__file__).parent.parent / "weights"
    sp_weights = weights_dir / "superpoint_v1.pth"
    if not sp_weights.exists():
        raise FileNotFoundError(f"Missing weights: {sp_weights}")

    W, H = args.size
    out_path = weights_dir / f"superpoint_backbone_{args.dtype.lower()}_{W}x{H}.onnx"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperPointBackboneExport(str(sp_weights)).to(device).eval()

    if args.dtype == "FP16":
        model = model.half()
        dummy = torch.randn(1, 3, H, W, device=device, dtype=torch.float16)
    else:
        dummy = torch.randn(1, 3, H, W, device=device, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["scores", "descriptors_map"],
        opset_version=18,
        do_constant_folding=True,
    )

    return out_path


def export_lightglue(args) -> Path:
    weights_dir = Path(__file__).parent.parent / "weights"
    lg_weights = weights_dir / "superpoint_lightglue.pth"
    if not lg_weights.exists():
        raise FileNotFoundError(f"Missing weights: {lg_weights}")

    n = int(args.num_kpts)
    out_path = weights_dir / f"superpoint_lightglue_{args.dtype.lower()}_k{n}.onnx"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperPointLightGlueCoreExport(str(lg_weights)).to(device).eval()

    if args.dtype == "FP16":
        model = model.half()
        dt = torch.float16
    else:
        dt = torch.float32

    # Inputs are in pixel coordinates.
    wh = torch.tensor([args.size[0], args.size[1]], device=device, dtype=dt)
    k0 = torch.rand(1, n, 2, device=device, dtype=dt) * wh
    k1 = torch.rand(1, n, 2, device=device, dtype=dt) * wh
    d0 = F.normalize(torch.randn(1, n, 256, device=device, dtype=dt), dim=-1)
    d1 = F.normalize(torch.randn(1, n, 256, device=device, dtype=dt), dim=-1)
    s0 = torch.tensor([[args.size[0], args.size[1]]], device=device, dtype=torch.int64)
    s1 = torch.tensor([[args.size[0], args.size[1]]], device=device, dtype=torch.int64)

    torch.onnx.export(
        model,
        (k0, d0, k1, d1, s0, s1),
        str(out_path),
        input_names=[
            "keypoints0",
            "descriptors0",
            "keypoints1",
            "descriptors1",
            "image_size0",
            "image_size1",
        ],
        output_names=["log_assignment"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=True,
    )

    return out_path


def main():
    parser = argparse.ArgumentParser(description="SuperPoint + LightGlue ONNX export")
    parser.add_argument("--component", choices=["superpoint", "lightglue"], required=True)
    parser.add_argument("--dtype", choices=["FP32", "FP16"], default="FP32")
    parser.add_argument("--size", nargs=2, type=int, default=[640, 480], help="Width Height")
    parser.add_argument("--num-kpts", type=int, default=1024, help="Only used for component=lightglue")
    args = parser.parse_args()

    if args.component == "superpoint":
        out = export_superpoint(args)
    else:
        out = export_lightglue(args)

    print(f"Exported: {out}")


if __name__ == "__main__":
    main()
