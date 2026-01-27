import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class SuperPointTorchScript(nn.Module):
    """TorchScript-friendly SuperPoint feature extractor (B=1).
    
    This exports only the dense feature extraction part of SuperPoint.
    Post-processing (NMS, border removal, top-k selection, descriptor sampling)
    is handled in C++ for efficiency and flexibility.
    
    Input:
      - x: (1,C,H,W) float32 in [0,1], image (grayscale or RGB)
    
    Output:
      - scores: (1, 1, H, W) float32, dense keypoint probability heatmap
      - descriptors_map: (1, 256, H/8, W/8) float32, L2-normalized descriptor map
    """
    
    def __init__(self, weights_path: Path):
        super().__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        
        # Shared encoder
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        
        # Keypoint detector head
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        
        # Descriptor head
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0)
        
        # Load weights
        state = torch.load(weights_path, map_location='cpu')
        self.load_state_dict(state)
        self.eval()
    
    def forward(self, x: torch.Tensor):
        """
        Extract dense features from image.
        
        Args:
            x: (1,C,H,W) image tensor in [0,1]
            
        Returns:
            scores: (1, 1, H, W) dense keypoint probability heatmap
            descriptors_map: (1, 256, H/8, W/8) L2-normalized descriptor map
        """
        # Convert to grayscale if RGB
        if x.shape[1] == 3:
            r = x[:, 0:1]
            g = x[:, 1:2]
            b = x[:, 2:3]
            x = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        # Shared encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Keypoint detector: dense scores
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        scores = torch.softmax(semi, dim=1)[:, :-1]  # (B, 64, h, w)
        B, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(B, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(B, 1, h * 8, w * 8)
        
        # Descriptor extractor: dense descriptors
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        desc = F.normalize(desc, p=2, dim=1)
        
        return scores, desc


def main():
    p = argparse.ArgumentParser(description="SuperPoint TorchScript export for matcher-cpp (B=1)")
    p.add_argument("--weights", type=str, default="matcher/lightglue/weights/superpoint_v1.pth")
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
        raise FileNotFoundError(f"Missing SuperPoint weights: {weights_path}")
    
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device_suffix = "_cuda" if args.device == "cuda" else ""
    out_path = out_dir / f"superpoint_fp32{device_suffix}.pt"
    
    m = SuperPointTorchScript(weights_path)
    m.to(device)
    m.eval()
    
    # Use trace instead of script for compatibility
    dummy_input = torch.zeros((1, 3, 480, 640), dtype=torch.float32, device=device)
    
    print(f"Tracing SuperPoint model on {device} with example input (1,3,480,640)...")
    print("Note: The model works with dynamic input sizes at inference time.")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = torch.jit.trace(m, dummy_input, strict=False)
        ts = torch.jit.freeze(ts)
    
    ts.save(str(out_path))
    print(f"Saved: {out_path}")
    print("Note: This model outputs dense features. C++ handles NMS and sparse post-processing.")


if __name__ == "__main__":
    main()
