#!/usr/bin/env python3
"""
Auto-converter: Creates TorchScript modules from .pt state dict weights.
This allows C++ to directly use matcher/xfeat/weights/xfeat.pt without manual conversion.
"""

import argparse
import sys
import torch
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from matcher.xfeat.modules.xfeat import XFeat
from matcher.xfeat.torchscript.convert_torchscript_xfeat import XFeatTorchScript
from matcher.xfeat.torchscript.convert_torchscript_xfeat_star import XFeatStarTorchScript


def create_sparse_module(weights_path: Path, top_k: int, device: str) -> torch.jit.ScriptModule:
    """Create TorchScript module for XFeat sparse mode."""
    print(f"[Auto-convert] Creating XFeat sparse TorchScript from {weights_path}")
    
    # Load XFeat model with weights
    xfeat = XFeat(weights=str(weights_path), top_k=top_k)
    
    # Create TorchScript wrapper
    module = XFeatTorchScript(weights_path, top_k=top_k, detection_threshold=0.05)
    module.to(device)
    module.eval()
    
    # Trace it
    dummy = torch.zeros((1, 3, 480, 640), dtype=torch.float32, device=device)
    traced = torch.jit.trace(module, (dummy,), strict=False)
    traced = torch.jit.freeze(traced)
    
    return traced


def create_star_module(weights_path: Path, top_k: int, fine_conf: float, device: str) -> torch.jit.ScriptModule:
    """Create TorchScript module for XFeat-Star mode."""
    print(f"[Auto-convert] Creating XFeat-Star TorchScript from {weights_path}")
    
    module = XFeatStarTorchScript(weights_path, top_k=top_k, fine_conf=fine_conf)
    module.to(device)
    module.eval()
    
    # Trace it
    dummy1 = torch.zeros((1, 3, 480, 640), dtype=torch.float32, device=device)
    dummy2 = torch.zeros((1, 3, 480, 640), dtype=torch.float32, device=device)
    
    traced = torch.jit.trace(module, (dummy1, dummy2), strict=False)
    traced = torch.jit.freeze(traced)
    
    return traced


def auto_convert_if_needed(mode: str, top_k: int = 4096, fine_conf: float = 0.25, device: str = "cpu") -> Path:
    """
    Automatically convert .pt weights to TorchScript if needed.
    
    Args:
        mode: "xfeat", "xfeat-star", or "xfeat-lightglue"
        top_k: number of keypoints
        fine_conf: confidence threshold for xfeat-star
        device: "cpu" or "cuda"
    
    Returns:
        Path to TorchScript .pt file
    """
    weights_dir = ROOT / "matcher" / "xfeat" / "weights"
    cpp_weights_dir = ROOT / "matcher-cpp" / "xfeat" / "weights"
    cpp_weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if original weights exist
    source_weights = weights_dir / "xfeat.pt"
    if not source_weights.exists():
        raise FileNotFoundError(f"Source weights not found: {source_weights}")
    
    # Determine output path
    if mode == "xfeat":
        device_suffix = ""
        output_name = f"xfeat_fp32_k{top_k}.pt"
        output_path = cpp_weights_dir / output_name
        
        # Check if already exists and is newer than source
        if output_path.exists() and output_path.stat().st_mtime > source_weights.stat().st_mtime:
            print(f"[Auto-convert] Using existing: {output_path}")
            return output_path
        
        # Create module
        module = create_sparse_module(source_weights, top_k, device)
        module.save(str(output_path))
        print(f"[Auto-convert] Saved: {output_path}")
        
    elif mode == "xfeat-star":
        device_suffix = "_cuda" if device == "cuda" else ""
        output_name = f"xfeat_star_fp32_k{top_k}{device_suffix}.pt"
        output_path = cpp_weights_dir / output_name
        
        # Check if already exists
        if output_path.exists() and output_path.stat().st_mtime > source_weights.stat().st_mtime:
            print(f"[Auto-convert] Using existing: {output_path}")
            return output_path
        
        # Create module
        module = create_star_module(source_weights, top_k, fine_conf, device)
        module.save(str(output_path))
        print(f"[Auto-convert] Saved: {output_path}")
        
    elif mode == "xfeat-lightglue":
        # LightGlue needs both xfeat.pt and xfeat-lighterglue.pt
        lg_weights = weights_dir / "xfeat-lighterglue.pt"
        if not lg_weights.exists():
            print(f"[Auto-convert] Warning: {lg_weights} not found, falling back to sparse mode")
            return auto_convert_if_needed("xfeat", top_k, fine_conf, device)
        
        # For now, just use sparse mode (full LightGlue integration is complex)
        return auto_convert_if_needed("xfeat", top_k, fine_conf, device)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Auto-convert XFeat weights to TorchScript")
    parser.add_argument("--mode", type=str, required=True, choices=["xfeat", "xfeat-star", "xfeat-lightglue"])
    parser.add_argument("--topk", type=int, default=4096)
    parser.add_argument("--fine-conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    
    output_path = auto_convert_if_needed(args.mode, args.topk, args.fine_conf, args.device)
    print(f"\n✓ TorchScript module ready: {output_path}")


if __name__ == "__main__":
    main()
