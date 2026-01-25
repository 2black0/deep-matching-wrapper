import torch
import numpy as np
import sys
from pathlib import Path

import onnxruntime as ort

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from matcher.liftfeat.modules.model import LiftFeatSPModel
from matcher.liftfeat.modules.liftfeat_wrapper import featureboost_config

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

    # 1. PyTorch Baseline (match liftfeat_wrapper.py)
    print("[*] Running PyTorch Baseline...")
    model = LiftFeatSPModel(featureboost_config).to(device).eval()
    state_dict = torch.load(str(pth_path), map_location="cpu")
    model.load_state_dict(state_dict)
    with torch.no_grad():
        des_map, kpt_map, d_feats = model.forward1(dummy_input)
        refined_descs_v = model.forward2(des_map, kpt_map, d_feats)
        refined_descs_map = refined_descs_v.view(1, des_map.shape[2], des_map.shape[3], -1).permute(0, 3, 1, 2)
        refined_descs_map = torch.nn.functional.normalize(refined_descs_map, p=2, dim=1)

        scores_raw = torch.softmax(kpt_map, dim=1)[:, :64]
        t_heat = (
            scores_raw.permute(0, 2, 3, 1)
            .reshape(1, des_map.shape[2], des_map.shape[3], 8, 8)
            .permute(0, 1, 3, 2, 4)
            .reshape(1, 1, des_map.shape[2] * 8, des_map.shape[3] * 8)
        )
        t_desc = refined_descs_map

    def softmax(x, axis=1):
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

    def logits_to_heatmap(kpt_logits):
        scores_raw = softmax(kpt_logits, axis=1)[:, :64]
        B, _, h_feat, w_feat = scores_raw.shape
        heat = (
            scores_raw.transpose(0, 2, 3, 1)
            .reshape(B, h_feat, w_feat, 8, 8)
            .transpose(0, 1, 3, 2, 4)
            .reshape(B, 1, h_feat * 8, w_feat * 8)
        )
        return heat

    # 2. ONNX FP32
    print("[*] Running ONNX FP32...")
    ort_32 = ort.InferenceSession(str(onnx_fp32), providers=['CPUExecutionProvider'])
    o32_kpt, o32_desc = ort_32.run(None, {'image': dummy_numpy})
    o32_heat = logits_to_heatmap(o32_kpt).astype(np.float32)

    # 3. ONNX FP16
    print("[*] Running ONNX FP16...")
    ort_16 = ort.InferenceSession(str(onnx_fp16), providers=['CPUExecutionProvider'])
    o16_kpt, o16_desc = ort_16.run(None, {'image': dummy_numpy.astype(np.float16)})
    o16_heat = logits_to_heatmap(o16_kpt.astype(np.float32)).astype(np.float32)

    def get_mse(a, b): return np.mean((a - b)**2)

    print("\n" + "-"*70)
    print(f"{'Metric':<20} | {'ONNX FP32 vs PTH':<20} | {'ONNX FP16 vs PTH':<20}")
    print("-" * 70)
    
    mse_h32 = get_mse(t_heat.cpu().numpy(), o32_heat)
    mse_h16 = get_mse(t_heat.cpu().numpy(), o16_heat)
    print(f"{'Heatmap MSE':<20} | {mse_h32:>18.6e} | {mse_h16:>18.6e}")

    mse_k32 = get_mse(kpt_map.cpu().numpy(), o32_kpt)
    mse_k16 = get_mse(kpt_map.cpu().numpy(), o16_kpt.astype(np.float32))
    print(f"{'KptLogits MSE':<20} | {mse_k32:>18.6e} | {mse_k16:>18.6e}")

    mse_d32 = get_mse(t_desc.cpu().numpy(), o32_desc)
    mse_d16 = get_mse(t_desc.cpu().numpy(), o16_desc)
    print(f"{'Descriptor MSE':<20} | {mse_d32:>18.6e} | {mse_d16:>18.6e}")
    print("-" * 70)

if __name__ == "__main__":
    compare_liftfeat()
