import torch
import torch.nn.functional as F
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from copy import deepcopy

from matcher.base_matcher import BaseMatcher
from .modules import LoFTR, full_default_cfg, opt_default_cfg, reparameter


class EfficientLoFTRMatcher(BaseMatcher):
    """EfficientLoFTR matcher wrapper"""
    
    repo_id = "ariG23498/eloftr"
    weight_filename = "eloftr_outdoors.safetensors"
    
    def __init__(self, device=None, cfg="full", **kwargs):
        super().__init__(device, **kwargs)
        
        # Check if weights exist locally
        weights_dir = Path(__file__).parent / "weights"
        weights_path = weights_dir / self.weight_filename
        
        if not weights_path.exists():
            print(f"Downloading EfficientLoFTR weights from HuggingFace...")
            weights_dir.mkdir(parents=True, exist_ok=True)
            model_path = hf_hub_download(repo_id=self.repo_id, filename=self.weight_filename)
            # Copy to local weights directory
            import shutil
            shutil.copy(model_path, weights_path)
        else:
            print(f"Loading EfficientLoFTR weights from {weights_path}")
        
        # Initialize model
        config = deepcopy(full_default_cfg if cfg == "full" else opt_default_cfg)
        self.matcher = LoFTR(config=config)
        
        # Load weights
        state_dict = load_file(str(weights_path))
        self.matcher.load_state_dict(state_dict)
        
        # Reparameterize and move to device
        self.matcher = reparameter(self.matcher).to(self.device).eval()
        
    def _preprocess(self, img):
        """Convert to grayscale and make divisible by 32"""
        # img is (C, H, W)
        if img.shape[0] == 3:
            # Convert RGB to grayscale
            img = 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]
        elif img.shape[0] == 1:
            pass  # Already grayscale
        else:
            raise ValueError(f"Unexpected number of channels: {img.shape[0]}")
        
        _, h, w = img.shape
        orig_shape = (h, w)
        
        # Resize to be divisible by 32
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32
        
        if new_h != h or new_w != w:
            img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), 
                               mode='bilinear', align_corners=False).squeeze(0)
        
        return img, orig_shape
    
    def _rescale_coords(self, coords, orig_h, orig_w, new_h, new_w):
        """Rescale coordinates from resized image to original image"""
        if coords is None or len(coords) == 0:
            return coords
        
        scale_h = orig_h / new_h
        scale_w = orig_w / new_w
        
        coords_rescaled = coords.clone()
        coords_rescaled[:, 0] *= scale_w
        coords_rescaled[:, 1] *= scale_h
        
        return coords_rescaled
    
    def _forward(self, img0, img1):
        # Preprocess images
        img0_proc, orig_shape0 = self._preprocess(img0)
        img1_proc, orig_shape1 = self._preprocess(img1)
        
        # Add batch dimension
        img0_proc = img0_proc.unsqueeze(0)  # (1, 1, H, W)
        img1_proc = img1_proc.unsqueeze(0)  # (1, 1, H, W)
        
        # Create batch dictionary
        batch = {
            "image0": img0_proc,
            "image1": img1_proc
        }
        
        # Forward pass
        with torch.no_grad():
            self.matcher(batch)
        
        # Extract matches
        mkpts0 = batch.get("mkpts0_f", torch.empty(0, 2))
        mkpts1 = batch.get("mkpts1_f", torch.empty(0, 2))
        
        # Rescale coordinates to original image sizes
        _, _, h0_new, w0_new = img0_proc.shape
        _, _, h1_new, w1_new = img1_proc.shape
        
        mkpts0 = self._rescale_coords(mkpts0, *orig_shape0, h0_new, w0_new)
        mkpts1 = self._rescale_coords(mkpts1, *orig_shape1, h1_new, w1_new)
        
        # EfficientLoFTR doesn't provide all keypoints or descriptors
        # Return None for those fields
        return mkpts0, mkpts1, None, None, None, None
