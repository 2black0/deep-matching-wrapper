import torch
import onnx
import onnxruntime as ort
import numpy as np
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from matcher.liftfeat.onnx.convert_onnx import LiftFeatExport

def compare_liftfeat(size=(640, 480)):
    W, H = size
    current_dir = Path(__file__).parent
    weights_dir = current_dir.parent / "weights"
    
    pth_path = weights_dir / "LiftFeat.pth"
    onnx_fp32 = weights_dir / f"liftfeat_fp32_{W}x{H}.onnx"
    onnx_fp16 = weights_dir / f"liftfeat_fp16_{W}x{H}.onnx"

    print(f"\n=== Comparing LiftFeat Models ({W}x{H}) ===")
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, H, W).to(device)
    dummy_numpy = dummy_input.cpu().numpy()

    # 1. PyTorch Baseline
    print("[*] Running PyTorch Baseline...")
    model_pth = LiftFeatExport(str(pth_path)).to(device).eval()
    with torch.no_grad():
        t_heat, t_desc = model_pth(dummy_input)

    # 2. ONNX FP32
    print("[*] Running ONNX FP32...")
    ort_32 = ort.InferenceSession(str(onnx_fp32), providers=['CPUExecutionProvider'])
    o32_out = ort_32.run(None, {'image': dummy_numpy})

    # 3. ONNX FP16
    print("[*] Running ONNX FP16...")
    ort_16 = ort.InferenceSession(str(onnx_fp16), providers=['CPUExecutionProvider'])
    o16_out = ort_16.run(None, {'image': dummy_numpy.astype(np.float16)})

    def get_mse(a, b): return np.mean((a - b)**2)

    print("\n" + "-"*70)
    print(f"{'Metric':<20} | {'ONNX FP32 vs PTH':<20} | {'ONNX FP16 vs PTH':<20}")
    print("-" * 70)
    
    mse_h32 = get_mse(t_heat.cpu().numpy(), o32_out[0])
    mse_h16 = get_mse(t_heat.cpu().numpy(), o16_out[0])
    print(f"{'Heatmap MSE':<20} | {mse_h32:>18.6e} | {mse_h16:>18.6e}")

    mse_d32 = get_mse(t_desc.cpu().numpy(), o32_out[1])
    mse_d16 = get_mse(t_desc.cpu().numpy(), o16_out[1])
    print(f"{'Descriptor MSE':<20} | {mse_d32:>18.6e} | {mse_d16:>18.6e}")
    print("-" * 70)

if __name__ == "__main__":
    compare_liftfeat()