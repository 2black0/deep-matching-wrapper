import argparse
import json
import statistics as stats
import time
from pathlib import Path

import numpy as np

from liftfeat_onnx_utils import WEIGHTS_PATH, compare_outputs, load_pth_export, make_input, run_onnx, run_pth


def summarize(times):
    return {
        "mean_ms": float(stats.mean(times)),
        "median_ms": float(stats.median(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "std_ms": float(stats.pstdev(times)),
    }


def bench_pth(weights, image, loops):
    t0 = time.perf_counter()
    model, device = load_pth_export(weights)
    load_ms = (time.perf_counter() - t0) * 1000
    run_pth(model, image, device)
    times = []
    last = None
    for _ in range(loops):
        t0 = time.perf_counter()
        last = run_pth(model, image, device)
        if device.type == "cuda":
            import torch
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {"name": "pth", "path": str(weights), "load_ms": load_ms, "latency": summarize(times), "outputs": last}


def bench_onnx(path, image, loops):
    import onnxruntime as ort
    providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(path), providers=providers)
    load_ms = (time.perf_counter() - t0) * 1000
    input_name = sess.get_inputs()[0].name
    image_in = image.astype(np.float16) if sess.get_inputs()[0].type == "tensor(float16)" else image
    sess.run(None, {input_name: image_in})
    times = []
    last = None
    for _ in range(loops):
        t0 = time.perf_counter()
        last = sess.run(None, {input_name: image_in})
        times.append((time.perf_counter() - t0) * 1000)
    last = [x.astype(np.float32) for x in last]
    return {"name": path.stem, "path": str(path), "providers": sess.get_providers(), "load_ms": load_ms, "latency": summarize(times), "outputs": last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS_PATH))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--loops", type=int, default=50)
    ap.add_argument("--models", nargs="*", default=[])
    args = ap.parse_args()

    image = make_input(args.width, args.height)
    pth = bench_pth(args.weights, image, args.loops)
    ref = pth.pop("outputs")
    results = [pth]

    models = [Path(p) for p in args.models] if args.models else sorted(Path(__file__).resolve().parent.glob(f"liftfeat_{args.width}x{args.height}_*.onnx"))
    for model in models:
        try:
            r = bench_onnx(model, image, args.loops)
            out = r.pop("outputs")
            r["accuracy_vs_pth"] = compare_outputs(ref, out)
            results.append(r)
        except Exception as e:
            results.append({"name": model.stem, "path": str(model), "error": repr(e)})

    print(json.dumps({"input_shape": [1, 3, args.height, args.width], "loops": args.loops, "results": results}, indent=2))


if __name__ == "__main__":
    main()
