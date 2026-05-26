import argparse
import json
import time
from pathlib import Path

from liftfeat_onnx_utils import WEIGHTS_PATH, compare_outputs, load_pth_export, make_input, run_onnx, run_pth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--weights", default=str(WEIGHTS_PATH))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--loops", type=int, default=10)
    args = ap.parse_args()

    image = make_input(args.width, args.height)
    pth, device = load_pth_export(args.weights)
    ref = run_pth(pth, image, device)
    t0 = time.perf_counter()
    sess, got = run_onnx(Path(args.model), image)
    first_ms = (time.perf_counter() - t0) * 1000
    input_name = sess.get_inputs()[0].name
    times = []
    for _ in range(args.loops):
        t0 = time.perf_counter()
        sess.run(None, {input_name: image})
        times.append((time.perf_counter() - t0) * 1000)
    print(json.dumps({
        "model": str(args.model),
        "providers": sess.get_providers(),
        "first_ms": first_ms,
        "loop_ms_mean": sum(times) / len(times),
        "loop_ms_min": min(times),
        "loop_ms_max": max(times),
        "compare_to_pth": compare_outputs(ref, got),
    }, indent=2))


if __name__ == "__main__":
    main()
