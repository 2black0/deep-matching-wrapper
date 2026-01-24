import torch
import torch.nn as nn
import argparse
import sys
from pathlib import Path

# Setup Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from matcher.liftfeat.modules.model import LiftFeatSPModel
from matcher.liftfeat.modules.liftfeat_wrapper import featureboost_config

class LiftFeatExport(nn.Module):
    def __init__(self, weights_path, top_k=1024):
        super().__init__()
        self.model = LiftFeatSPModel(featureboost_config)
        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _unfold2d_onnx(self, x, ws=8):
        B, C, H, W = x.shape
        # Ops Unfold untuk mengambil patch 8x8 
        x = x.unfold(2, ws, ws).unfold(3, ws, ws)
        return x.contiguous().view(B, C * ws * ws, H // ws, W // ws)

    def forward(self, x):
        B, C_in, H, W = x.shape
        x_gray = x.mean(dim=1, keepdim=True)
        
        # Ekstraksi Dense Maps 
        des_map, kpt_map, d_feats = self.model.forward1(x_gray)

        # Proses Normals (d_feats) 
        normals_feat = self._unfold2d_onnx(d_feats, ws=8)
        
        B, C_d, h_feat, w_feat = des_map.shape
        des_v = des_map.permute(0, 2, 3, 1).reshape(-1, C_d)
        kpts_v = kpt_map.permute(0, 2, 3, 1).reshape(-1, 65)
        norm_v = normals_feat.permute(0, 2, 3, 1).reshape(-1, 192)

        # Feature Boosting 
        refined_descs_v = self.model.feature_boost(des_v, kpts_v, norm_v)
        refined_descs_map = refined_descs_v.view(B, h_feat, w_feat, -1).permute(0, 3, 1, 2)

        # Heatmap Reconstruction (Softmax + Reshape) 
        scores_raw = torch.softmax(kpt_map, dim=1)[:, :64]
        heatmap = scores_raw.view(B, 8, 8, h_feat, w_feat).permute(0, 3, 1, 4, 2).reshape(B, 1, h_feat*8, w_feat*8)

        # Pastikan output kembali ke FP32 untuk kompatibilitas wrapper luar
        return heatmap.float(), refined_descs_map.float()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='matcher/liftfeat/weights/LiftFeat.pth')
    parser.add_argument('--size', nargs=2, type=int, default=[640, 480])
    parser.add_argument('--dtype', type=str, choices=['FP32', 'FP16'], default='FP32')
    args = parser.parse_args()

    W, H = args.size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LiftFeatExport(args.weights).to(device)
    
    if args.dtype == 'FP16':
        model = model.half()
        dummy_input = torch.randn(1, 3, H, W).to(device).half()
    else:
        dummy_input = torch.randn(1, 3, H, W).to(device)

    output_path = Path(args.weights).parent / f"liftfeat_{args.dtype.lower()}_{W}x{H}.onnx"

    print(f"--- Exporting LiftFeat as {args.dtype} ---")
    torch.onnx.export(
        model, dummy_input, str(output_path),
        input_names=['image'], output_names=['heatmap', 'descriptors_map'],
        opset_version=18, do_constant_folding=True
    )

    import onnx
    from onnxsim import simplify
    onnx_model = onnx.load(str(output_path))
    model_simp, check = simplify(onnx_model, check_n=3)
    if check:
        onnx.save(model_simp, str(output_path))
        print("Model simplified successfully.")

if __name__ == "__main__":
    main()