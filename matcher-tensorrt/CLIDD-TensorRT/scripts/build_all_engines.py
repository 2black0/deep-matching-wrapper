#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

CLIDD_WEIGHTS = Path("/home/ardyseto/Documents/GitHub/deep-matching-wrapper/matcher/clidd/weights")
TRTEXEC = "/home/ardyseto/tensorrt/bin/trtexec"
OUTPUT_DIR = Path("/home/ardyseto/Documents/GitHub/deep-matching-wrapper/matcher-tensorrt/CLIDD-TensorRT/weights")

MODEL_VARIANTS = {
    "M64": {"cfg": "M64", "desc_dim": 64},
    "A48": {"cfg": "A48", "desc_dim": 48},
    "U128": {"cfg": "U128", "desc_dim": 128},
}

RESOLUTIONS = [
    (480, 640),
    (640, 480),
    (800, 600),
    (600, 800),
    (752, 480),
    (1242, 375),
]


def align_to_32(h, w):
    return ((h + 31) // 32) * 32, ((w + 31) // 32) * 32


def export_onnx(variant, orig_h, orig_w, output_dir, topk=2048):
    aligned_h, aligned_w = align_to_32(orig_h, orig_w)
    cfg = MODEL_VARIANTS[variant]["cfg"]

    onnx_path = output_dir / f"clidd_{variant.lower()}_{aligned_w}x{aligned_h}.onnx"

    if onnx_path.exists():
        print(f"ONNX already exists: {onnx_path}")
        return onnx_path, aligned_h, aligned_w

    weights_path = CLIDD_WEIGHTS / f"{cfg}.pth"
    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}")
        return None, aligned_h, aligned_w

    onnx_str = str(onnx_path)
    export_code = f"""
import sys
from pathlib import Path
sys.path.insert(0, "/home/ardyseto/Documents/GitHub/deep-matching-wrapper")

import torch
from matcher.clidd.modules.model import Model
from matcher.clidd.modules.clidd_wrapper import CLIDD

class CLIDDExport(torch.nn.Module):
    def __init__(self, cfg_name, weights_path, top_k=1024, radius=2, score_thresh=-5.0, border=4):
        super().__init__()
        cfg_params = CLIDD.cfgs[cfg_name]
        self.model = Model(**cfg_params)
        
        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.top_k = top_k
        self.radius = radius
        self.score_thresh = float(score_thresh)
        self.border = int(border)
        self.mp = torch.nn.MaxPool2d(kernel_size=self.radius * 2 + 1, stride=1, padding=self.radius)

    def forward(self, x):
        B, C, H, W = x.shape
        size = torch.tensor([W, H], dtype=x.dtype, device=x.device)

        raw_desc, raw_detect = self.model(x)

        is_max = raw_detect == self.mp(raw_detect)

        if self.border > 0:
            b = self.border
            border_mask = torch.ones_like(is_max, dtype=torch.bool)
            border_mask[..., :, :b] = False
            border_mask[..., :, -b:] = False
            border_mask[..., :b, :] = False
            border_mask[..., -b:, :] = False
            is_max = is_max & border_mask

        is_good = raw_detect > self.score_thresh
        valid = (is_max & is_good)

        neg_inf = torch.full_like(raw_detect, -1e8)
        refined = torch.where(valid, raw_detect, neg_inf)

        flat_scores = refined.view(B, -1)
        scores, indices = torch.topk(flat_scores, k=self.top_k, dim=1)

        y = (indices // W).to(torch.float32)
        x_coords = (indices % W).to(torch.float32)
        kpts = torch.stack([x_coords, y], dim=-1)

        norm_kpts = (kpts + 0.5) / size.to(torch.float32) * 2 - 1
        norm_kpts = norm_kpts.unsqueeze(2).to(x.dtype)

        descriptors = self.model.sample(list(raw_desc), norm_kpts)

        return kpts.to(torch.float32), scores.to(torch.float32), descriptors.to(torch.float32)

model = CLIDDExport(
    "{cfg}",
    "{weights_path}",
    top_k={topk},
    radius=2,
    score_thresh=-5.0,
    border=4,
)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dummy_input = torch.randn(1, 3, {aligned_h}, {aligned_w}).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "{onnx_str}",
    input_names=['image'],
    output_names=['keypoints', 'scores', 'descriptors'],
    opset_version=18,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False
)
print("Exported: " + "{onnx_str}")
"""

    result = subprocess.run(
        [sys.executable, "-c", export_code],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"ERROR exporting ONNX {variant} {aligned_h}x{aligned_w}:")
        print(result.stderr)
        return None, aligned_h, aligned_w

    print(f"ONNX exported: {onnx_path}")
    return onnx_path, aligned_h, aligned_w


def build_engine(onnx_path, variant, height, width, precision):
    engine_name = f"clidd_{variant.lower()}_{width}x{height}_{precision}.engine"
    engine_path = OUTPUT_DIR / engine_name

    if engine_path.exists():
        print(f"Engine already exists: {engine_path}")
        return engine_path

    if precision == "fp32":
        cmd = [
            str(TRTEXEC),
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--memPoolSize=workspace:4096M",
        ]
    elif precision == "fp16":
        cmd = [
            str(TRTEXEC),
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16",
            "--memPoolSize=workspace:4096M",
        ]
    else:
        cmd = [
            str(TRTEXEC),
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--int8",
            "--memPoolSize=workspace:4096M",
        ]

    print(f"Building {precision} engine for {variant} {height}x{width}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR building {precision} engine {variant} {height}x{width}:")
        stderr = result.stderr
        print(stderr[-1000:] if len(stderr) > 1000 else stderr)
        if precision == "int8":
            print("Trying fp16 fallback...")
            cmd_fp16 = [
                str(TRTEXEC),
                f"--onnx={onnx_path}",
                f"--saveEngine={engine_path}",
                "--fp16",
                "--int8",
                "--memPoolSize=workspace:4096M",
            ]
            result = subprocess.run(cmd_fp16, capture_output=True, text=True)
            if result.returncode != 0:
                print("Also failed, skipping...")
                return None

    print(f"Engine built: {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines for CLIDD")
    parser.add_argument("--variant", type=str, default="M64", choices=["M64", "A48", "U128"],
                        help="CLIDD variant to build (default: M64)")
    parser.add_argument("--resolution", type=str, default="640x480",
                        help="Resolution to build (default: 640x480)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"],
                        help="Precision to build (default: fp16)")
    parser.add_argument("--topk", type=int, default=2048,
                        help="Top-K keypoints (default: 2048)")
    parser.add_argument("--all-res", action="store_true",
                        help="Build all resolutions")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Building TensorRT Engines for CLIDD-{args.variant}")
    print("=" * 60)

    if args.all_res:
        resolutions = RESOLUTIONS
    else:
        w, h = args.resolution.split("x")
        resolutions = [(int(h), int(w))]

    for orig_h, orig_w in resolutions:
        aligned_h, aligned_w = align_to_32(orig_h, orig_w)

        print(f"\n{'='*60}")
        print(f"Variant: {args.variant}, Original: {orig_h}x{orig_w} -> Aligned: {aligned_h}x{aligned_w}")
        print("=" * 60)

        onnx_path, ah, aw = export_onnx(args.variant, orig_h, orig_w, OUTPUT_DIR, args.topk)
        if onnx_path is not None:
            build_engine(onnx_path, args.variant, ah, aw, args.precision)

    print("\n" + "=" * 60)
    print("DONE! Built engines:")
    print("=" * 60)
    for f in sorted(OUTPUT_DIR.glob("*.engine")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
