import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file

from matcher.edm.modules.default_config import get_cfg_defaults
from matcher.edm.modules.edm import EDM
from matcher.edm.modules.misc import lower_config


class EDMTorchScript(nn.Module):
    """TorchScript-friendly EDM deploy wrapper.

    Input:
      - x: (1,2,H,W) float32 in [0,1]
        channel 0: image0 grayscale
        channel 1: image1 grayscale

    Output:
      - y: (topk, 11) float32
    """

    def __init__(self, weights_path: Path, w: int, h: int, topk: int, mconf_thr: float = 0.2, border_rm: int = 2):
        super().__init__()

        cfg = get_cfg_defaults()

        # Apply the same outdoor/megadepth inference settings used by matcher/edm/__init__.py
        cfg.TRAINER.CANONICAL_BS = 32
        cfg.TRAINER.CANONICAL_LR = 2e-3
        cfg.TRAINER.SCALING = 8
        cfg.TRAINER.EPI_ERR_THR = 1e-4
        cfg.EDM.TRAIN_RES_H = 832
        cfg.EDM.TRAIN_RES_W = 832
        cfg.TRAINER.N_VAL_PAIRS_TO_PLOT = 32

        cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
        cfg.EDM.TEST_RES_H = 1152
        cfg.EDM.TEST_RES_W = 1152
        cfg.EDM.NECK.NPE = [832, 832, 1152, 1152]

        # Fixed-shape deploy export parameters
        cfg.EDM.DEPLOY = True
        cfg.EDM.COARSE.TOPK = int(topk)
        cfg.EDM.COARSE.MCONF_THR = float(mconf_thr)
        cfg.EDM.COARSE.BORDER_RM = int(border_rm)
        
        # Ensure NPE is set correctly in the config object before lowering
        cfg.EDM.NECK.NPE = [832, 832, 1152, 1152]

        cfg = lower_config(cfg)
        self.net = EDM(config=cfg["edm"]).eval()

        if not weights_path.exists():
            raise FileNotFoundError(f"Missing EDM weights: {weights_path}")

        # Load SafeTensors and make all tensors contiguous
        print(f"Loading SafeTensors weights from {weights_path}...")
        state = load_file(weights_path)
        
        # FIX #2: Make all tensors contiguous (similar to safetensors-converter)
        print("Making tensors contiguous...")
        state_contiguous = {
            k: v.contiguous() if isinstance(v, torch.Tensor) else v 
            for k, v in state.items()
        }
        
        self.net.load_state_dict(state_contiguous, strict=True)
        print(f"Loaded {len(state_contiguous)} tensors")

        self.w = int(w)
        self.h = int(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Intentionally avoid Python-side shape assertions here.
        # This module is exported via torch.jit.trace (fixed shape), and Python asserts
        # on traced shapes produce noisy TracerWarnings.
        return self.net(x)


def main():
    p = argparse.ArgumentParser(description="EDM TorchScript export for matcher-cpp (deploy mode, fixed shape) - FIXED VERSION")
    p.add_argument("--weights", type=str, default=str(ROOT / "matcher" / "edm" / "weights" / "edm.safetensors"))
    p.add_argument("--w", type=int, default=640)
    p.add_argument("--h", type=int, default=480)
    p.add_argument("--topk", type=int, default=1680)
    p.add_argument("--mconf", type=float, default=0.2)
    p.add_argument("--border-rm", type=int, default=2)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "matcher-cpp" / "edm" / "weights"),
    )
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to export for")
    args = p.parse_args()

    # We export with torch.jit.trace (fixed-shape). The upstream EDM code contains
    # some Python-side conditionals / asserts on tensor sizes that are safe for our
    # fixed-shape export but produce TracerWarnings. Suppress them to keep the
    # export logs clean.
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    weights_path = Path(args.weights)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Support CUDA export for models that will be used on CUDA
    device_suffix = f"_{args.device}" if args.device == "cuda" else ""
    out_path = out_dir / f"edm_fp32_w{int(args.w)}_h{int(args.h)}_topk{int(args.topk)}{device_suffix}_v2.pt"

    print(f"\n{'='*60}")
    print(f"EDM TorchScript Export (FIXED VERSION)")
    print(f"{'='*60}\n")

    # FIX #1 REVISED: For EDM, we need to trace on target device due to device-specific constants
    # Build model on CPU first, then move to target device before tracing
    print("Building EDM model...")
    m = EDMTorchScript(weights_path, w=args.w, h=args.h, topk=args.topk, mconf_thr=args.mconf, border_rm=args.border_rm)
    m.eval()
    
    # Move to target device
    device = torch.device(args.device)
    if args.device == "cuda":
        print(f"Moving model to {args.device}...")
        m = m.to(device)
    
    # Create example input on target device
    print(f"Creating example input ({args.h}x{args.w}) on {args.device}...")
    example = torch.zeros((1, 2, int(args.h), int(args.w)), dtype=torch.float32, device=device)
    
    # FIX #4: Warmup run before tracing
    print("Warming up model...")
    with torch.no_grad():
        _ = m(example)
    
    # Trace on target device (using strict=False for C++ compatibility)
    print("Tracing model (this may take a minute)...")
    ts = torch.jit.trace(m, example, strict=False)
    ts = torch.jit.freeze(ts)
    
    # Skip optimize_for_inference - causes C++ compatibility issues with PyTorch 2.9.1
    
    # Save
    print(f"Saving to {out_path}...")
    ts.save(str(out_path))
    print(f"✓ Saved: {out_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
