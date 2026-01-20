from pathlib import Path
import torch
import numpy as np
import torchvision.transforms.functional as TF
from huggingface_hub import hf_hub_download

from matcher.base_matcher import BaseMatcher
from matcher.xfeat.modules.xfeat import XFeat

class XFeatMatcher(BaseMatcher):
    def __init__(self, device="cpu", mode="xfeat", **kwargs):
        """
        Args:
            device (str): Device to run inference on.
            mode (str): One of "xfeat", "xfeat-star", "xfeat-lightglue".
            **kwargs: Additional arguments for BaseMatcher and models.
        """
        super().__init__(device, **kwargs)
        self.mode = mode
        
        # Determine weight paths
        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Main XFeat weights
        xfeat_weights = weights_dir / "xfeat.pt"
        if not xfeat_weights.exists():
            print(f"Downloading XFeat weights to {xfeat_weights}...")
            hf_hub_download(repo_id="image-matching-models/xfeat", filename="xfeat.pt", local_dir=weights_dir)
            
        print(f"loading weights from: {xfeat_weights}")
        # Initialize XFeat
        self.model = XFeat(weights=str(xfeat_weights), top_k=kwargs.get("max_num_keypoints", 4096))
        self.model.net = self.model.net.to(self.device)
        self.model.dev = self.device
        
        # Load LightGlue weights if needed
        if self.mode == "xfeat-lightglue":
            lg_weights = weights_dir / "xfeat-lighterglue.pt"
            if not lg_weights.exists():
                 print(f"Downloading XFeat-LighterGlue weights to {lg_weights}...")
                 torch.hub.download_url_to_file(
                     "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt",
                     lg_weights
                 )
            
            print(f"loading weights from: {lg_weights}")
            
            # Initialize LighterGlue with suppressed output
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                from matcher.xfeat.modules.lighterglue import LighterGlue
                self.model.lighterglue = LighterGlue(weights=str(lg_weights))
                self.model.lighterglue.net = self.model.lighterglue.net.to(self.device)

    def preprocess(self, img: torch.Tensor):
        """
        XFeat requires input dimensions to be divisible by 32.
        Also handles adding batch dimension if needed.
        """
        # Ensure batch dimension
        if img.ndim == 3:
            img = img.unsqueeze(0)
            
        # Parse input (handles internal requirements of XFeat)
        return self.model.parse_input(img)

    def _forward(self, img0: torch.Tensor, img1: torch.Tensor):
        # Preprocess (resize to mod 32 is handled inside XFeat methods or we need to do it)
        # XFeat.parse_input handles shape, but XFeat.preprocess_tensor handles resizing
        # We will let the model methods handle it, as they call preprocess_tensor internally.
        
        # Get original shapes for rescaling
        h0, w0 = img0.shape[-2:]
        h1, w1 = img1.shape[-2:]
        
        # Ensure batch dim for standard compliance if single image
        if img0.ndim == 3: img0 = img0.unsqueeze(0)
        if img1.ndim == 3: img1 = img1.unsqueeze(0)

        img0_dev = img0.to(self.device)
        img1_dev = img1.to(self.device)
        
        if self.mode == "xfeat":
             # Sparse + MNN
             # XFeat.detectAndCompute handles resizing and returns keypoints in original scale (mostly)
             # But XFeat.match_xfeat does everything
             
             # Warning: match_xfeat expects (1,C,H,W) and returns numpy arrays
             mkpts0, mkpts1 = self.model.match_xfeat(img0_dev, img1_dev)
             
             # We also need all keypoints for standard output
             out0 = self.model.detectAndCompute(img0_dev, top_k=self.model.top_k)[0]
             out1 = self.model.detectAndCompute(img1_dev, top_k=self.model.top_k)[0]
             
             all_kpts0 = out0['keypoints'].cpu().numpy()
             all_kpts1 = out1['keypoints'].cpu().numpy()
             all_desc0 = out0['descriptors'].cpu().numpy()
             all_desc1 = out1['descriptors'].cpu().numpy()
             
        elif self.mode == "xfeat-star":
            # Semi-dense + refine
            # match_xfeat_star returns list of matches per batch
            matches_list = self.model.match_xfeat_star(img0_dev, img1_dev)
            
            # Since we support batch=1 for now in this wrapper
            if isinstance(matches_list, list):
                matches = matches_list[0] # (N, 4) x1,y1,x2,y2
            else:
                matches = matches_list # Should be tuple if came from return logic in match_xfeat_star (weird API)
                if isinstance(matches, tuple):
                     mkpts0, mkpts1 = matches
                else:
                     # tensor (N, 4)
                     mkpts0 = matches[:, :2].cpu().numpy()
                     mkpts1 = matches[:, 2:].cpu().numpy()
            
            if isinstance(matches, list): # Handling multiple returns
                 # The code says: return matches if B > 1 else (matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy())
                 pass
            
            # Recalculating efficiently creates redundancy, but XFeat API is designed as end-to-end
            # For "all_kpts", we can run detectAndComputeDense
            out0 = self.model.detectAndComputeDense(img0_dev, top_k=self.model.top_k)
            out1 = self.model.detectAndComputeDense(img1_dev, top_k=self.model.top_k)
            
            all_kpts0 = out0['keypoints'].squeeze(0).cpu().numpy()
            all_kpts1 = out1['keypoints'].squeeze(0).cpu().numpy()
            all_desc0 = out0['descriptors'].squeeze(0).cpu().numpy()
            all_desc1 = out1['descriptors'].squeeze(0).cpu().numpy()

            if not 'mkpts0' in locals():
                 # Handle the return format of match_xfeat_star
                 res = self.model.match_xfeat_star(img0_dev, img1_dev)
                 if isinstance(res, tuple):
                     mkpts0, mkpts1 = res
                 else:
                     mkpts0 = res[0][:, :2].cpu().numpy()
                     mkpts1 = res[0][:, 2:].cpu().numpy()

        elif self.mode == "xfeat-lightglue":
             # Sparse + LightGlue
             out0 = self.model.detectAndCompute(img0_dev, top_k=self.model.top_k)[0]
             out1 = self.model.detectAndCompute(img1_dev, top_k=self.model.top_k)[0]
             
             # Add image size (W, H)
             out0['image_size'] = (w0, h0)
             out1['image_size'] = (w1, h1)
             
             mkpts0, mkpts1, _ = self.model.match_lighterglue(out0, out1)
             
             all_kpts0 = out0['keypoints'].cpu().numpy()
             all_kpts1 = out1['keypoints'].cpu().numpy()
             all_desc0 = out0['descriptors'].cpu().numpy()
             all_desc1 = out1['descriptors'].cpu().numpy()
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        return mkpts0, mkpts1, all_kpts0, all_kpts1, all_desc0, all_desc1
