import argparse
import sys
from pathlib import Path
import warnings

import torch
import torch.nn as nn


# Allow running as a script from repo root
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# torch.onnx dynamic shape exporter may emit a harmless warning when the same
# dynamic dimension is reused across multiple inputs.
warnings.filterwarnings("ignore", message=r".*axis name: num_pairs.*")


class Keypt2SubpxRefinerNoScoreExport(nn.Module):
    """Export wrapper for Keypt2Subpx AttnTuner (no scorepatch).

    Inputs:
      - patch: (B, N, 1, 11, 11) grayscale patches in [0,1] (or [-1,1])
      - desc_mean: (B, N, D) descriptor mean, where D matches output_dim

    Output:
      - delta: (B, N, 2) subpixel delta to add to keypoints (already scaled by 2.5)
    """

    def __init__(self, output_dim: int, state_dict: dict):
        super().__init__()
        from matcher.subpx.modules.keypt2subpx import Keypt2Subpx

        k2s = Keypt2Subpx(output_dim=output_dim, use_score=False)
        k2s.net.load_state_dict(state_dict)
        k2s.eval()
        self.net = k2s.net

    def forward(self, patch: torch.Tensor, desc_mean: torch.Tensor):
        # net returns BxNx2 in pixel units (centered), Keypt2Subpx multiplies by 2.5
        coord = self.net(patch, scorepatch=None, desc=desc_mean)
        return coord * 2.5


class Keypt2SubpxRefinerWithScoreExport(nn.Module):
    """Export wrapper for Keypt2Subpx AttnTuner (with scorepatch).

    Inputs:
      - patch: (B, N, 1, 11, 11)
      - scorepatch: (B, N, 1, 11, 11)
      - desc_mean: (B, N, D)

    Output:
      - delta: (B, N, 2) subpixel delta to add to keypoints (already scaled by 2.5)
    """

    def __init__(self, output_dim: int, state_dict: dict):
        super().__init__()
        from matcher.subpx.modules.keypt2subpx import Keypt2Subpx

        k2s = Keypt2Subpx(output_dim=output_dim, use_score=True)
        k2s.net.load_state_dict(state_dict)
        k2s.eval()
        self.net = k2s.net

    def forward(self, patch: torch.Tensor, scorepatch: torch.Tensor, desc_mean: torch.Tensor):
        coord = self.net(patch, scorepatch=scorepatch, desc=desc_mean)
        return coord * 2.5


def _repo_root() -> Path:
    return ROOT


def _pth_weights_dir() -> Path:
    return _repo_root() / "matcher" / "subpx" / "weights"


def _onnx_out_dir() -> Path:
    out_dir = _repo_root() / "matcher-onnx" / "weights" / "subpx"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_state_dict(weights_path: Path) -> dict:
    state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected weights format: {weights_path}")
    return state


def export_refiner(args, *, output_dim: int, use_score: bool, weights_name: str, out_stem: str) -> Path:
    weights_dir = _pth_weights_dir()
    weights_path = weights_dir / weights_name
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    state_dict = _load_state_dict(weights_path)

    out_dir = _onnx_out_dir()
    out_path = out_dir / f"{out_stem}_{args.dtype.lower()}.onnx"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_score:
        model = Keypt2SubpxRefinerWithScoreExport(output_dim=output_dim, state_dict=state_dict).to(device).eval()
    else:
        model = Keypt2SubpxRefinerNoScoreExport(output_dim=output_dim, state_dict=state_dict).to(device).eval()

    n = int(args.num_kpts)
    dt = torch.float16 if args.dtype == "FP16" else torch.float32
    if args.dtype == "FP16":
        model = model.half()

    patch = torch.randn(1, n, 1, 11, 11, device=device, dtype=dt)
    desc = torch.randn(1, n, int(output_dim), device=device, dtype=dt)

    # Use the new exporter with dynamic shapes for N.
    from torch.export import Dim

    num_pairs = Dim("num_pairs", min=1)

    if use_score:
        scorepatch = torch.randn(1, n, 1, 11, 11, device=device, dtype=dt)
        dynamic_shapes = {
            "patch": {1: num_pairs},
            "scorepatch": {1: num_pairs},
            "desc_mean": {1: num_pairs},
        }
        torch.onnx.export(
            model,
            (patch, scorepatch, desc),
            str(out_path),
            input_names=["patch", "scorepatch", "desc_mean"],
            output_names=["delta"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
        )
    else:
        dynamic_shapes = {
            "patch": {1: num_pairs},
            "desc_mean": {1: num_pairs},
        }
        torch.onnx.export(
            model,
            (patch, desc),
            str(out_path),
            input_names=["patch", "desc_mean"],
            output_names=["delta"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
        )

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Keypt2Subpx (SubPX) ONNX export")
    parser.add_argument(
        "--matcher",
        choices=["xfeat-subpx", "xfeat-lightglue-subpx", "superpoint-lightglue-subpx", "all"],
        required=True,
    )
    parser.add_argument("--dtype", choices=["FP32", "FP16"], default="FP32")
    parser.add_argument("--num-kpts", type=int, default=1024, help="Dummy N for export; ONNX supports dynamic N")
    args = parser.parse_args()

    exported: list[Path] = []

    if args.matcher in ("xfeat-subpx", "xfeat-lightglue-subpx", "all"):
        # Same refiner weights/config for both XFeat variants.
        exported.append(
            export_refiner(
                args,
                output_dim=64,
                use_score=False,
                weights_name="k2s_xfeat_pretrained.pth",
                out_stem="k2s_xfeat_refiner",
            )
        )

    if args.matcher in ("superpoint-lightglue-subpx", "all"):
        exported.append(
            export_refiner(
                args,
                output_dim=256,
                use_score=True,
                weights_name="k2s_splg_pretrained.pth",
                out_stem="k2s_splg_refiner",
            )
        )

    for p in exported:
        print(f"Exported: {p}")


if __name__ == "__main__":
    main()
