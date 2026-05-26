import argparse
import json
from pathlib import Path

import torch

from liftfeat_onnx_utils import WEIGHTS_PATH, LiftFeatExport, make_input, run_pth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS_PATH))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    weights = Path(args.weights)
    sd = torch.load(str(weights), map_location="cpu")
    model = LiftFeatExport(weights)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    image = make_input(args.width, args.height)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    outs = run_pth(model, image, torch.device(device))

    info = {
        "weights": str(weights),
        "num_tensors": len(sd),
        "total_weight_values": int(sum(v.numel() for v in sd.values() if hasattr(v, "numel"))),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "device": device,
        "input_shape": [1, 3, args.height, args.width],
        "outputs": {
            "kpt_logits": {"shape": list(outs[0].shape), "dtype": str(outs[0].dtype), "min": float(outs[0].min()), "max": float(outs[0].max())},
            "descriptors_map": {"shape": list(outs[1].shape), "dtype": str(outs[1].dtype), "min": float(outs[1].min()), "max": float(outs[1].max())},
        },
        "sample_keys": list(sd.keys())[:20],
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
