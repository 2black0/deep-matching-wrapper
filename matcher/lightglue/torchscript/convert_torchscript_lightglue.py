import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from matcher.lightglue.modules.lightglue import LightGlue


class LightGlueTorchScript(nn.Module):
    """TorchScript-friendly LightGlue matcher for SuperPoint features.
    
    This exports the core LightGlue transformer matching logic.
    - Disables flash attention for TorchScript compatibility
    - Disables pruning and early stopping (depth_confidence=-1, width_confidence=-1)
    - Outputs log-assignment scores matrix
    
    C++ will handle:
    - Keypoint normalization
    - Match filtering by threshold
    - Extracting final match indices
    
    Input:
      - keypoints0: (1, N0, 2) float32, pixel coordinates
      - descriptors0: (1, N0, 256) float32, L2-normalized descriptors
      - keypoints1: (1, N1, 2) float32, pixel coordinates
      - descriptors1: (1, N1, 256) float32, L2-normalized descriptors
      - image_size0: (1, 2) int64, [width, height]
      - image_size1: (1, 2) int64, [width, height]
    
    Output:
      - scores: (1, N0+1, N1+1) float32, log-assignment matrix
    """
    
    def __init__(self, weights_path: Path):
        super().__init__()
        
        # Configure LightGlue for TorchScript export:
        # - Disable flash attention (not compatible with TorchScript)
        # - Disable pruning/early-stop (dynamic control flow not compatible)
        self.lg = LightGlue(
            features=None,  # type: ignore[arg-type]
            input_dim=256,
            descriptor_dim=256,
            add_scale_ori=False,
            n_layers=9,
            num_heads=4,
            flash=False,  # CRITICAL: Flash attention not compatible with TorchScript
            mp=False,
            depth_confidence=-1,  # Disable pruning
            width_confidence=-1,  # Disable pruning
            filter_threshold=0.1,
        ).eval()
        
        # Load weights with key remapping
        state_dict = torch.load(weights_path, map_location='cpu')
        new_state = {}
        for k, v in state_dict.items():
            k = k.replace("matcher.", "")
            # Remap attention blocks (old format -> new format)
            for i in range(self.lg.conf.n_layers):
                k = k.replace(f"self_attn.{i}.", f"transformers.{i}.self_attn.")
                k = k.replace(f"cross_attn.{i}.", f"transformers.{i}.cross_attn.")
            new_state[k] = v
        self.lg.load_state_dict(new_state, strict=False)
        self.lg.eval()
    
    def normalize_keypoints(self, kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        """Normalize keypoints to [-1, 1] range."""
        # size: (B, 2) as [width, height]
        # Convert to float
        size_f = size.to(dtype=kpts.dtype, device=kpts.device)
        shift = size_f / 2.0
        scale = size_f.max(dim=-1, keepdim=True)[0] / 2.0
        kpts_norm = (kpts - shift.unsqueeze(1)) / scale.unsqueeze(1)
        return kpts_norm
    
    def forward(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size0: torch.Tensor,
        image_size1: torch.Tensor,
    ):
        """
        Match two sets of features using LightGlue.
        
        Args:
            keypoints0: (1, N0, 2) pixel coordinates
            descriptors0: (1, N0, 256) L2-normalized descriptors
            keypoints1: (1, N1, 2) pixel coordinates
            descriptors1: (1, N1, 256) L2-normalized descriptors
            image_size0: (1, 2) [width, height]
            image_size1: (1, 2) [width, height]
            
        Returns:
            scores: (1, N0+1, N1+1) log-assignment matrix
        """
        # Normalize keypoints
        kpts0 = self.normalize_keypoints(keypoints0, image_size0)
        kpts1 = self.normalize_keypoints(keypoints1, image_size1)
        
        # Project descriptors to LightGlue embedding space
        desc0 = descriptors0.contiguous()
        desc1 = descriptors1.contiguous()
        desc0 = self.lg.input_proj(desc0)
        desc1 = self.lg.input_proj(desc1)
        
        # Positional encodings
        enc0 = self.lg.posenc(kpts0)
        enc1 = self.lg.posenc(kpts1)
        
        # Transformer layers (no pruning, no early stopping)
        for i in range(self.lg.conf.n_layers):
            desc0, desc1 = self.lg.transformers[i](desc0, desc1, enc0, enc1)
        
        # Final matching assignment (re-implemented to avoid in-place operations)
        la = self.lg.log_assignment[self.lg.conf.n_layers - 1]
        mdesc0 = la.final_proj(desc0)
        mdesc1 = la.final_proj(desc1)
        d = float(mdesc0.shape[-1])
        mdesc0 = mdesc0 / (d ** 0.25)
        mdesc1 = mdesc1 / (d ** 0.25)
        
        # Compute similarity matrix
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        
        # Compute matchability scores
        z0 = la.matchability(desc0)
        z1 = la.matchability(desc1)
        
        # Build log-assignment matrix
        certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
        scores0 = F.log_softmax(sim, dim=2)
        scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), dim=2).transpose(-1, -2)
        s00 = scores0 + scores1 + certainties
        
        # Add dustbin (unmatched) scores
        last_col = F.logsigmoid(-z0.squeeze(-1))
        last_row = F.logsigmoid(-z1.squeeze(-1))
        top = torch.cat([s00, last_col.unsqueeze(-1)], dim=2)
        bottom_right = sim.new_zeros((sim.shape[0], 1, 1))
        bottom = torch.cat([last_row.unsqueeze(1), bottom_right], dim=2)
        scores = torch.cat([top, bottom], dim=1)
        
        return scores


def main():
    p = argparse.ArgumentParser(description="LightGlue (SuperPoint) TorchScript export for matcher-cpp")
    p.add_argument("--weights", type=str, default="matcher/lightglue/weights/superpoint_lightglue.pth")
    p.add_argument("--max-kpts", type=int, default=2048, 
                   help="Maximum number of keypoints for dummy input (model supports dynamic)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                   help="Device to trace on (cpu or cuda)")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "matcher-cpp" / "lightglue" / "weights"),
    )
    args = p.parse_args()
    
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing LightGlue weights: {weights_path}")
    
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device_suffix = "_cuda" if args.device == "cuda" else ""
    out_path = out_dir / f"superpoint_lightglue_fp32_k{args.max_kpts}{device_suffix}.pt"
    
    m = LightGlueTorchScript(weights_path)
    m.to(device)
    m.eval()
    
    # Create dummy inputs
    n = args.max_kpts
    dummy_kpts0 = torch.rand((1, n, 2), dtype=torch.float32, device=device) * 640.0
    dummy_desc0 = F.normalize(torch.randn(1, n, 256, dtype=torch.float32, device=device), dim=-1)
    dummy_kpts1 = torch.rand((1, n, 2), dtype=torch.float32, device=device) * 640.0
    dummy_desc1 = F.normalize(torch.randn(1, n, 256, dtype=torch.float32, device=device), dim=-1)
    dummy_size0 = torch.tensor([[640, 480]], dtype=torch.int64, device=device)
    dummy_size1 = torch.tensor([[640, 480]], dtype=torch.int64, device=device)
    
    print(f"Tracing LightGlue model on {device} with {n} keypoints...")
    print("Note: The model supports variable number of keypoints at inference time.")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = torch.jit.trace(
            m, 
            (dummy_kpts0, dummy_desc0, dummy_kpts1, dummy_desc1, dummy_size0, dummy_size1),
            strict=False
        )
        ts = torch.jit.freeze(ts)
    
    ts.save(str(out_path))
    print(f"Saved: {out_path}")
    print("Note: Flash attention and pruning are disabled for TorchScript compatibility.")


if __name__ == "__main__":
    main()
