import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# Allow running as a script from repo root
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


class XFeatBackboneExport(nn.Module):
    """Export XFeat backbone outputs.

    Outputs are dense maps; keypoint selection and matching are done outside ONNX.

    Returns:
      - descriptors_map: (B, 64, H/8, W/8) L2-normalized along channel dim
      - kpt_logits:      (B, 65, H/8, W/8) keypoint logits
      - reliability:     (B,  1, H/8, W/8) reliability map
    """

    def __init__(self, weights_path: str):
        super().__init__()
        from matcher.xfeat.modules.model import XFeatModel

        self.net = XFeatModel()
        state = torch.load(weights_path, map_location="cpu")
        self.net.load_state_dict(state)
        self.net.eval()

    def forward(self, image: torch.Tensor):
        feats, kpt_logits, reliability = self.net(image)
        feats = F.normalize(feats, dim=1)
        return feats.float(), kpt_logits.float(), reliability.float()


class LighterGlueExport(nn.Module):
    """Export a fixed-shape LightGlue matcher for XFeat descriptors.

    For ONNX-friendliness we export the continuous output `log_assignment`.
    Matches can be computed outside ONNX from this matrix.
    """

    def __init__(self, weights_path: str, n_layers: int = 6, num_heads: int = 1, descriptor_dim: int = 96):
        super().__init__()
        from kornia.feature.lightglue import LightGlue

        conf = {
            "name": "xfeat",
            "input_dim": 64,
            "descriptor_dim": int(descriptor_dim),
            "add_scale_ori": False,
            "add_laf": False,
            "scale_coef": 1.0,
            "n_layers": int(n_layers),
            "num_heads": int(num_heads),
            "flash": False,  # ONNX export compatibility
            "mp": False,
            "depth_confidence": -1,
            "width_confidence": -1,  # disable pruning (export stability)
            "filter_threshold": 0.1,
            "weights": None,
        }

        LightGlue.default_conf = conf
        self.net = LightGlue(None).eval()

        state_dict = torch.load(weights_path, map_location="cpu")
        # Rename old state dict entries (kept from matcher/xfeat/modules/lighterglue.py)
        for i in range(self.net.conf.n_layers):
            pattern = (f"self_attn.{i}", f"transformers.{i}.self_attn")
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = (f"cross_attn.{i}", f"transformers.{i}.cross_attn")
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            state_dict = {k.replace("matcher.", ""): v for k, v in state_dict.items()}

        self.net.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size0: torch.Tensor,
        image_size1: torch.Tensor,
    ):
        # We intentionally bypass LightGlue.forward/_forward because Kornia includes
        # runtime checks using `.item()` which are not `torch.export`-friendly.
        # This reproduces the core computation that produces `log_assignment`.
        from kornia.feature.lightglue import normalize_keypoints

        kpts0 = normalize_keypoints(keypoints0, image_size0).clone()
        kpts1 = normalize_keypoints(keypoints1, image_size1).clone()

        desc0 = descriptors0.detach().contiguous()
        desc1 = descriptors1.detach().contiguous()
        desc0 = self.net.input_proj(desc0)
        desc1 = self.net.input_proj(desc1)
        encoding0 = self.net.posenc(kpts0)
        encoding1 = self.net.posenc(kpts1)

        for i in range(self.net.conf.n_layers):
            desc0, desc1 = self.net.transformers[i](desc0, desc1, encoding0, encoding1)

        # Re-implement MatchAssignment without in-place writes.
        # Kornia's implementation builds the (m+1)x(n+1) matrix via index_put,
        # which turns into ScatterND in ONNX and can be numerically unstable on GPU.
        la = self.net.log_assignment[self.net.conf.n_layers - 1]
        mdesc0, mdesc1 = la.final_proj(desc0), la.final_proj(desc1)
        d = mdesc0.shape[-1]
        mdesc0, mdesc1 = mdesc0 / (d**0.25), mdesc1 / (d**0.25)
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = la.matchability(desc0)
        z1 = la.matchability(desc1)

        certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
        scores0 = F.log_softmax(sim, 2)
        scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
        s00 = scores0 + scores1 + certainties  # (B, m, n)

        last_col = F.logsigmoid(-z0.squeeze(-1))  # (B, m)
        last_row = F.logsigmoid(-z1.squeeze(-1))  # (B, n)

        top = torch.cat([s00, last_col.unsqueeze(-1)], dim=2)  # (B, m, n+1)
        bottom_right = sim.new_zeros((sim.shape[0], 1, 1))
        bottom = torch.cat([last_row.unsqueeze(1), bottom_right], dim=2)  # (B, 1, n+1)
        scores = torch.cat([top, bottom], dim=1)  # (B, m+1, n+1)

        return scores.float()


def export_xfeat(args) -> Path:
    weights_dir = Path(__file__).parent.parent / "weights"
    weights_path = weights_dir / "xfeat.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    W, H = args.size
    out_path = weights_dir / f"xfeat_backbone_{args.dtype.lower()}_{W}x{H}.onnx"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XFeatBackboneExport(str(weights_path)).to(device).eval()

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
        output_names=["descriptors_map", "kpt_logits", "reliability"],
        opset_version=18,
        do_constant_folding=True,
    )

    return out_path


def export_lightglue(args) -> Path:
    weights_dir = Path(__file__).parent.parent / "weights"
    weights_path = weights_dir / "xfeat-lighterglue.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    n = int(args.num_kpts)
    out_path = weights_dir / f"xfeat_lighterglue_{args.dtype.lower()}_k{n}.onnx"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LighterGlueExport(str(weights_path)).to(device).eval()

    if args.dtype == "FP16":
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Keypoints are in pixel coordinates; sample valid locations to satisfy LightGlue checks.
    wh = torch.tensor([args.size[0], args.size[1]], device=device, dtype=dtype)
    keypoints0 = torch.rand(1, n, 2, device=device, dtype=dtype) * wh
    keypoints1 = torch.rand(1, n, 2, device=device, dtype=dtype) * wh
    descriptors0 = F.normalize(torch.randn(1, n, 64, device=device, dtype=dtype), dim=-1)
    descriptors1 = F.normalize(torch.randn(1, n, 64, device=device, dtype=dtype), dim=-1)
    # image_size is conceptually integer (W, H)
    image_size0 = torch.tensor([[args.size[0], args.size[1]]], device=device, dtype=torch.int64)
    image_size1 = torch.tensor([[args.size[0], args.size[1]]], device=device, dtype=torch.int64)

    torch.onnx.export(
        model,
        (keypoints0, descriptors0, keypoints1, descriptors1, image_size0, image_size1),
        str(out_path),
        input_names=["keypoints0", "descriptors0", "keypoints1", "descriptors1", "image_size0", "image_size1"],
        output_names=["log_assignment"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=True,
    )

    return out_path


def main():
    parser = argparse.ArgumentParser(description="XFeat / LighterGlue ONNX export")
    parser.add_argument("--matcher", choices=["xfeat", "lightglue"], required=True)
    parser.add_argument("--dtype", choices=["FP32", "FP16"], default="FP32")
    parser.add_argument("--size", nargs=2, type=int, default=[640, 480], help="Width Height")
    parser.add_argument("--num-kpts", type=int, default=1024, help="Only used for lightglue export")
    args = parser.parse_args()

    if args.matcher == "xfeat":
        out_path = export_xfeat(args)
    else:
        out_path = export_lightglue(args)

    print(f"Exported: {out_path}")


if __name__ == "__main__":
    main()
