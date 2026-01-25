import torch
import onnxruntime as ort
import numpy as np
import argparse
import sys
from pathlib import Path
from scipy.spatial import cKDTree

# Memastikan path modul clidd terdeteksi agar bisa load model asli
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from matcher.clidd.modules.clidd_wrapper import CLIDD

def get_spatial_metrics(kpts_a, kpts_b, threshold=3.0):
    """
    Menghitung seberapa dekat kumpulan titik B terhadap A secara spasial.
    Precision @threshold: % titik B yang memiliki tetangga di A dalam radius X pixel.
    """
    ka = kpts_a[0]
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

    # 2. Run PyTorch Baseline (match CLIDD wrapper)
    print(f"[*] Executing PyTorch Baseline...")
    try:
        torch_model = CLIDD(cfg=cfg_name, top_k=top_k, radius=2, score=-5, weights_path=pth_path).to(device).eval()
        with torch.no_grad():
            out = torch_model(dummy_input)[0]

        t_kpts_np = out["keypoints"].detach().cpu().numpy()[None, ...]
        t_scores_np = out["scores"].detach().cpu().numpy()[None, ...]
        t_desc_np = out["descriptors"].detach().cpu().numpy()[None, ...]
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
    
    def filter_valid(kpts, scores, descs, score_floor=-1e7):
        # If the exporter had fewer than top_k valid points, it may fill with very negative scores.
        s = scores[0]
        keep = s > score_floor
        return kpts[:, keep], scores[:, keep], descs[:, keep]

    def match_by_spatial_nn(kpts_ref, kpts_test, max_px=1.0):
        ref = kpts_ref[0]
        test = kpts_test[0]
        if len(ref) == 0 or len(test) == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        tree = cKDTree(ref)
        dist, idx = tree.query(test)
        keep = dist <= max_px
        return idx[keep].astype(np.int64), np.nonzero(keep)[0].astype(np.int64)

    if o32_out is not None and o16_out is not None:
        # Spatial Metrics (Metrik paling jujur untuk keypoints)
        o32_k, o32_s, o32_d = o32_out
        o16_k, o16_s, o16_d = o16_out

        t_k, t_s, t_d = filter_valid(t_kpts_np, t_scores_np, t_desc_np)
        o32_k, o32_s, o32_d = filter_valid(o32_k, o32_s, o32_d)
        o16_k, o16_s, o16_d = filter_valid(o16_k.astype(np.float32), o16_s.astype(np.float32), o16_d.astype(np.float32))

        prec32, dist32 = get_spatial_metrics(t_k, o32_k)
        prec16, dist16 = get_spatial_metrics(t_k, o16_k)
        
        print(f"{'Spatial Precision @3px':<25} | {prec32:>20.2f}% | {prec16:>20.2f}%")
        print(f"{'Mean Point Dist (px)':<25} | {dist32:>20.4f} px | {dist16:>20.4f} px")
        
        # Score/descriptor metrics after spatial NN alignment
        r32, q32 = match_by_spatial_nn(t_k, o32_k, max_px=1.0)
        r16, q16 = match_by_spatial_nn(t_k, o16_k, max_px=1.0)

        print(f"{'Aligned Pairs @1px':<25} | {len(r32):>20d}    | {len(r16):>20d}")

        if len(r32) > 0:
            mse_s32 = get_mse(t_s[0, r32], o32_s[0, q32])
            mse_d32 = get_mse(t_d[0, r32], o32_d[0, q32])
        else:
            mse_s32 = float('nan')
            mse_d32 = float('nan')

        if len(r16) > 0:
            mse_s16 = get_mse(t_s[0, r16], o16_s[0, q16])
            mse_d16 = get_mse(t_d[0, r16], o16_d[0, q16])
        else:
            mse_s16 = float('nan')
            mse_d16 = float('nan')

        print(f"{'Scores MSE (Aligned)':<25} | {mse_s32:>20.6e}    | {mse_s16:>20.6e}")
        print(f"{'Desc MSE (Aligned)':<25} | {mse_d32:>20.6e}    | {mse_d16:>20.6e}")
        
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
