import torch
import onnx
import onnxruntime as ort
import numpy as np
import argparse
import sys
from pathlib import Path
from scipy.spatial import cKDTree

# Memastikan path modul clidd terdeteksi agar bisa load model asli
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from matcher.clidd.onnx.convert_onnx import CLIDDExport

def get_spatial_metrics(kpts_a, kpts_b, threshold=3.0):
    """
    Menghitung seberapa dekat kumpulan titik B terhadap A secara spasial.
    Precision @threshold: % titik B yang memiliki tetangga di A dalam radius X pixel.
    """
    ka = kpts_a[0] # Shape (1024, 2)
    kb = kpts_b[0]
    
    # Membangun KDTree untuk pencarian tetangga terdekat yang efisien
    tree = cKDTree(ka)
    distances, _ = tree.query(kb)
    
    precision = np.mean(distances <= threshold) * 100
    mean_dist = np.mean(distances)
    return precision, mean_dist

def get_mse(a, b):
    return np.mean((a - b)**2)

def compare_models(cfg_name, size=(640, 480), top_k=1024):
    W, H = size
    cfg_name = cfg_name.upper()
    current_dir = Path(__file__).parent
    weights_dir = current_dir.parent / "weights"
    
    # Paths ke file weights
    pth_path = weights_dir / f"{cfg_name}.pth"
    onnx_fp32 = weights_dir / f"clidd_{cfg_name.lower()}_fp32_{W}x{H}.onnx"
    onnx_fp16 = weights_dir / f"clidd_{cfg_name.lower()}_fp16_{W}x{H}.onnx"

    print(f"\n" + "="*70)
    print(f" COMPARING CLIDD {cfg_name} MODELS ({W}x{H}) ")
    print("="*70)
    
    # 1. Prepare Input
    # Gunakan seed agar random input konsisten untuk perbandingan
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, H, W).to(device)
    dummy_numpy = dummy_input.cpu().numpy()

    # 2. Run PyTorch Baseline (FP32)
    print(f"[*] Executing PyTorch Baseline...")
    try:
        torch_model = CLIDDExport(cfg_name, str(pth_path), top_k=top_k).to(device)
        torch_model.eval()
        with torch.no_grad():
            t_kpts, t_scores, t_desc = torch_model(dummy_input)
        t_kpts_np = t_kpts.cpu().numpy()
        t_scores_np = t_scores.cpu().numpy()
        t_desc_np = t_desc.cpu().numpy()
    except Exception as e:
        print(f"❌ Error loading PyTorch model: {e}")
        return

    # 3. Run ONNX FP32
    print(f"[*] Executing ONNX FP32...")
    try:
        ort_fp32 = ort.InferenceSession(str(onnx_fp32), providers=['CPUExecutionProvider'])
        o32_out = ort_fp32.run(None, {'image': dummy_numpy})
    except Exception as e:
        print(f"❌ Error loading ONNX FP32: {e}")
        o32_out = None

    # 4. Run ONNX FP16
    print(f"[*] Executing ONNX FP16...")
    try:
        dummy_numpy_f16 = dummy_numpy.astype(np.float16)
        ort_fp16 = ort.InferenceSession(str(onnx_fp16), providers=['CPUExecutionProvider'])
        o16_out = ort_fp16.run(None, {'image': dummy_numpy_f16})
    except Exception as e:
        print(f"❌ Error loading ONNX FP16: {e}")
        o16_out = None

    # --- PRINT TABLE RESULTS ---
    print("\n" + "-"*75)
    header = f"{'Metric':<25} | {'ONNX FP32 vs PTH':<22} | {'ONNX FP16 vs PTH':<22}"
    print(header)
    print("-" * 75)
    
    if o32_out and o16_out:
        # Spatial Metrics (Metrik paling jujur untuk keypoints)
        prec32, dist32 = get_spatial_metrics(t_kpts_np, o32_out[0])
        prec16, dist16 = get_spatial_metrics(t_kpts_np, o16_out[0].astype(np.float32))
        
        print(f"{'Spatial Precision @3px':<25} | {prec32:>20.2f}% | {prec16:>20.2f}%")
        print(f"{'Mean Point Dist (px)':<25} | {dist32:>20.4f} px | {dist16:>20.4f} px")
        
        # Scores MSE (Sorted agar adil)
        s_pth = np.sort(t_scores_np[0])
        s_32 = np.sort(o32_out[1][0])
        s_16 = np.sort(o16_out[1][0].astype(np.float32))
        
        mse_s32 = get_mse(s_pth, s_32)
        mse_s16 = get_mse(s_pth, s_16)
        print(f"{'Scores MSE (Sorted)':<25} | {mse_s32:>20.6e}    | {mse_s16:>20.6e}")

        # Descriptor MSE
        # Note: Karena urutan titik berbeda, Descriptor MSE akan tetap tinggi pada FP16
        # kecuali jika kita mencocokkan titik terdekatnya satu per satu.
        mse_d32 = get_mse(t_desc_np, o32_out[2])
        mse_d16 = get_mse(t_desc_np, o16_out[2].astype(np.float32))
        print(f"{'Descriptor MSE (Raw)':<25} | {mse_d32:>20.6e}    | {mse_d16:>20.6e}")
        
    print("-" * 75)
    print("\nAnalysis Note:")
    print("1. FP32 should have near-zero error (MSE < 1e-7).")
    print("2. If FP16 'Spatial Precision' is low (< 50%), the model is likely broken due to overflow.")
    print("3. 'Mean Point Dist' indicates how many pixels keypoints shifted on average.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Model name (A48, U128, etc.)')
    parser.add_argument('--topk', type=int, default=1024)
    args = parser.parse_args()
    
    compare_models(args.weights, top_k=args.topk)