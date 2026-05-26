import argparse
import json
import time

from liftfeat_onnx_utils import WEIGHTS_PATH, compare_outputs, load_pth_export, make_input, run_pth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS_PATH))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--loops", type=int, default=10)
    args = ap.parse_args()

    model, device = load_pth_export(args.weights)
    image = make_input(args.width, args.height)
    t0 = time.perf_counter()
    ref = run_pth(model, image, device)
    first_ms = (time.perf_counter() - t0) * 1000
    times = []
    for _ in range(args.loops):
        t0 = time.perf_counter()
        got = run_pth(model, image, device)
        if device.type == "cuda":
            import torch
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    print(json.dumps({
        "model": "pth",
        "device": str(device),
        "first_ms": first_ms,
        "loop_ms_mean": sum(times) / len(times),
        "loop_ms_min": min(times),
        "loop_ms_max": max(times),
        "self_compare": compare_outputs(ref, got),
    }, indent=2))


if __name__ == "__main__":
    main()
