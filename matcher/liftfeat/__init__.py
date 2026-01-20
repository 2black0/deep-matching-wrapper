
from pathlib import Path
import torch
import numpy as np
from huggingface_hub import hf_hub_download

from matcher.base_matcher import BaseMatcher
from matcher.utils import to_numpy
from matcher.liftfeat.modules.liftfeat_wrapper import LiftFeat

class LiftFeatMatcher(BaseMatcher):
    def __init__(self, device="cpu", detect_threshold=0.05, **kwargs):
        super().__init__(device, **kwargs)
        self.detect_threshold = detect_threshold
        
        # Determine weight paths
        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "LiftFeat.pth"
        
        if not weights_path.exists():
            print(f"Downloading LiftFeat weights to {weights_path}...")
            hf_hub_download(repo_id="image-matching-models/liftfeat", filename="liftfeat.pth", local_dir=weights_dir)
            # rename if needed, but hf_download keeps filename
        
        print(f"loading weights from: {weights_path}")
        
        # LiftFeat wrapper expects string path
        self.model = LiftFeat(weight=str(weights_path), detect_threshold=self.detect_threshold)
        
        # Ensure model is on correct device
        self.model.net = self.model.net.to(self.device)
        self.model.detector = self.model.detector.to(self.device)
        self.model.sampler = self.model.sampler.to(self.device)
        # Update internal device reference if it exists
        if hasattr(self.model, 'device'):
            self.model.device = torch.device(self.device)

    def preprocess(self, img):
        "LiftFeat requires input as raw ndarray (result of cv2.imread)"
        # BaseMatcher loads as Tensor (C, H, W) in [0,1]
        # LiftFeat expects numpy (H, W, C) in [0, 255] usually?
        # Let's check wrapper: image = torch.tensor(image).permute(0, 3, 1, 2) / 255
        # So it expects [0, 255].
        
        if isinstance(img, torch.Tensor):
            # Tensor (C, H, W) [0, 1] -> Numpy (H, W, C) [0, 255]
            img = to_numpy(img).transpose(1, 2, 0)
            img = (img * 255).astype(np.float32)

        return img

    def _forward(self, img0, img1):
        # Preprocess to numpy
        img0_np = self.preprocess(img0)
        img1_np = self.preprocess(img1)
        
        # Run matching
        # match_liftfeat expects numpy inputs
        mkpts0, mkpts1 = self.model.match_liftfeat(img0_np, img1_np)
        
        # LiftFeat only returns matches, no keypoints/descriptors for the full image in this API
        return (
            mkpts0,
            mkpts1,
            None,  # all_kpts0
            None,  # all_kpts1
            None,  # all_desc0
            None,  # all_desc1
        )
