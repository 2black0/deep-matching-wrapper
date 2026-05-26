import argparse
import json
from pathlib import Path

import onnx
import onnxruntime as ort

from liftfeat_onnx_utils import make_input, run_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=[])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    models = [Path(p) for p in args.models]
    if not models:
        models = sorted(Path(__file__).resolve().parent.glob("liftfeat_*x*_*.onnx"))
    image = make_input(args.width, args.height)
    out = []
    for path in models:
        item = {"path": str(path)}
        try:
            m = onnx.load(str(path))
            onnx.checker.check_model(m)
            item["onnx_ok"] = True
            sess, y = run_onnx(path, image)
            item["providers"] = sess.get_providers()
            item["inputs"] = [{"name": i.name, "shape": i.shape, "type": i.type} for i in sess.get_inputs()]
            item["outputs"] = [{"name": o.name, "shape": list(v.shape), "dtype": str(v.dtype), "min": float(v.min()), "max": float(v.max())} for o, v in zip(sess.get_outputs(), y)]
        except Exception as e:
            item["onnx_ok"] = False
            item["error"] = repr(e)
        out.append(item)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
