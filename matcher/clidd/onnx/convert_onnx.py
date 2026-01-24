import torch
import torch.nn as nn
import argparse
import sys
from pathlib import Path

# Memastikan path modul clidd terdeteksi
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from matcher.clidd.modules.model import Model
from matcher.clidd.modules.clidd_wrapper import CLIDD

class CLIDDExport(nn.Module):
    def __init__(self, cfg_name, weights_path, top_k=1024):
        super().__init__()
        cfg_params = CLIDD.cfgs[cfg_name]
        self.model = Model(**cfg_params)
        
        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.top_k = top_k
        self.radius = 2
        self.mp = nn.MaxPool2d(kernel_size=self.radius * 2 + 1, stride=1, padding=self.radius)

    def forward(self, x):
        B, C, H, W = x.shape
        # Pastikan size memiliki tipe yang sama dengan x untuk menghindari cast error
        size = torch.tensor([W, H], dtype=x.dtype, device=x.device)

        # 1. Jalankan backbone
        dense_features, raw_scores = self.model(x) 

        # 2. Proses deteksi (Scores & Top-K)
        # Kita biarkan tetap dalam dtype x (FP16/FP32) agar konsisten dengan weights
        is_max = (raw_scores == self.mp(raw_scores))
        mask = is_max.to(x.dtype)
        refined_scores = raw_scores * mask
        
        flat_scores = refined_scores.view(B, -1)
        scores, indices = torch.topk(flat_scores, k=self.top_k, dim=1)

        # 3. Koordinat (Keypoints)
        # Kita hitung dalam float32 untuk akurasi posisi, lalu cast kembali ke x.dtype
        y = (indices // W).to(torch.float32)
        x_coords = (indices % W).to(torch.float32)
        kpts = torch.stack([x_coords, y], dim=-1) 

        # 4. Feature Sampling
        # Norm_kpts harus kembali ke dtype x sebelum masuk ke model.sample (Einsum)
        norm_kpts = (kpts + 0.5) / size.to(torch.float32) * 2 - 1
        norm_kpts = norm_kpts.unsqueeze(2).to(x.dtype)

        # model.sample sekarang menerima input yang konsisten dengan bobot model (FP16 atau FP32)
        descriptors = self.model.sample(list(dense_features), norm_kpts)

        # Pastikan kpts dikembalikan ke FP32 untuk konsistensi output wrapper
        return kpts, scores, descriptors.to(torch.float32)

def main():
    parser = argparse.ArgumentParser(description='CLIDD Flexible ONNX Export Script')
    parser.add_argument('--weights', type=str, required=True, help='Model name (e.g., A48, U128)')
    parser.add_argument('--topk', type=int, default=1024, help='Number of keypoints')
    parser.add_argument('--size', nargs=2, type=int, default=[640, 480], help='Width Height')
    parser.add_argument('--dtype', type=str, choices=['FP32', 'FP16'], default='FP32', help='Output data type')
    args = parser.parse_args()

    cfg_name = args.weights.upper()
    W, H = args.size
    
    current_dir = Path(__file__).parent
    weights_file = current_dir.parent / "weights" / f"{cfg_name}.pth"
    output_onnx = current_dir.parent / "weights" / f"clidd_{cfg_name.lower()}_{args.dtype.lower()}_{W}x{H}.onnx"

    print(f"--- Exporting CLIDD {cfg_name} as {args.dtype} ---")
    model = CLIDDExport(cfg_name, str(weights_file), top_k=args.topk)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Terapkan konversi FP16 jika diminta
    if args.dtype == 'FP16':
        model = model.half()
        dummy_input = torch.randn(1, 3, H, W).to(device).half()
    else:
        dummy_input = torch.randn(1, 3, H, W).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx),
        input_names=['image'],
        output_names=['keypoints', 'scores', 'descriptors'],
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        keep_initializers_as_inputs=False
    )

    print(f"Selesai: {output_onnx}")

    try:
        import onnxsim
        import onnx
        print(f"Simplifying {args.dtype} model...")
        onnx_model = onnx.load(str(output_onnx))
        model_simp, check = onnxsim.simplify(onnx_model, check_n=3)
        if check:
            onnx.save(model_simp, str(output_onnx))
            print("Model successfully simplified and verified!")
    except Exception as e:
        print(f"Simplification error: {e}")

if __name__ == "__main__":
    main()