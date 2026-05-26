import argparse
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationDataReader, quantize_static

from liftfeat_onnx_utils import WEIGHTS_PATH, LiftFeatExport, p


def export_fp32(weights, out, width, height, opset):
    model = LiftFeatExport(weights).eval()
    dummy = torch.randn(1, 3, height, width, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["kpt_logits", "descriptors_map"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    onnx.checker.check_model(str(out))
    return out


def export_fp16(weights, out, width, height, opset):
    model = LiftFeatExport(weights).eval().half()
    dummy = torch.randn(1, 3, height, width, dtype=torch.float16)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["kpt_logits", "descriptors_map"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    onnx.checker.check_model(str(out))
    return out


class RandomCalib(CalibrationDataReader):
    def __init__(self, input_name, width, height, n=4):
        import numpy as np
        rng = np.random.default_rng(123)
        self.data = [{input_name: rng.random((1, 3, height, width), dtype=np.float32)} for _ in range(n)]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        item = self.data[self.i]
        self.i += 1
        return item


def export_int8_static(fp32_path, int8_path, width, height):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    reader = RandomCalib(sess.get_inputs()[0].name, width, height)
    quantize_static(
        str(fp32_path),
        str(int8_path),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )
    onnx.checker.check_model(str(int8_path))
    return int8_path


def maybe_simplify(path):
    try:
        from onnxsim import simplify
        model = onnx.load(str(path))
        simp, ok = simplify(model, check_n=1)
        if ok:
            onnx.save(simp, str(path))
            p(f"simplified {path}")
    except Exception as e:
        p(f"simplify skipped {path}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS_PATH))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--formats", nargs="+", default=["fp32", "fp16", "int8"], choices=["fp32", "fp16", "int8"])
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"liftfeat_{args.width}x{args.height}"
    fp32 = out_dir / f"{stem}_fp32.onnx"
    fp16 = out_dir / f"{stem}_fp16.onnx"
    int8 = out_dir / f"{stem}_int8.onnx"

    if "fp32" in args.formats or "fp16" in args.formats or "int8" in args.formats:
        p(f"export fp32 -> {fp32}")
        export_fp32(args.weights, fp32, args.width, args.height, args.opset)
        if not args.no_simplify:
            maybe_simplify(fp32)
    if "fp16" in args.formats:
        p(f"convert fp16 -> {fp16}")
        export_fp16(args.weights, fp16, args.width, args.height, args.opset)
        if not args.no_simplify:
            maybe_simplify(fp16)
    if "int8" in args.formats:
        p(f"quantize int8 static -> {int8}")
        export_int8_static(fp32, int8, args.width, args.height)

    p("done")


if __name__ == "__main__":
    main()
