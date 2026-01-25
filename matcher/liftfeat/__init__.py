
from pathlib import Path
import torch
import numpy as np
from huggingface_hub import hf_hub_download

from matcher.base_matcher import BaseMatcher
from matcher.utils import to_numpy
from matcher.liftfeat.modules.liftfeat_wrapper import LiftFeat

class LiftFeatMatcher(BaseMatcher):
    def __init__(self, device="cpu", detect_threshold=0.05, top_k=4096, min_cossim=-1.0, **kwargs):
        super().__init__(device, **kwargs)
        self.detect_threshold = detect_threshold
        self.top_k = int(top_k)
        self.min_cossim = float(min_cossim)
        
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
            # Tensor (C,H,W) in [0,1] -> uint8 (H,W,C) in [0,255]
            img = to_numpy(img).transpose(1, 2, 0)
            img = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)

        return img

    def _forward(self, img0, img1):
        # Preprocess to numpy for the LiftFeat wrapper
        img0_np = self.preprocess(img0)
        img1_np = self.preprocess(img1)

        data0 = self.model.extract(img0_np)
        data1 = self.model.extract(img1_np)

        kpts0 = data0["keypoints"]
        kpts1 = data1["keypoints"]
        desc0 = data0["descriptors"]
        desc1 = data1["descriptors"]
        scores0 = data0["scores"]
        scores1 = data1["scores"]

        # Apply top-k (python wrapper currently does not enforce it)
        if self.top_k > 0 and scores0.numel() > self.top_k:
            _, sel = torch.topk(scores0, k=self.top_k, dim=0, sorted=True)
            kpts0 = kpts0.index_select(0, sel)
            desc0 = desc0.index_select(0, sel)
            scores0 = scores0.index_select(0, sel)

        if self.top_k > 0 and scores1.numel() > self.top_k:
            _, sel = torch.topk(scores1, k=self.top_k, dim=0, sorted=True)
            kpts1 = kpts1.index_select(0, sel)
            desc1 = desc1.index_select(0, sel)
            scores1 = scores1.index_select(0, sel)

        # Mutual NN match (cosine similarity)
        if desc0.numel() == 0 or desc1.numel() == 0:
            empty = desc0.new_zeros((0, 2))
            return empty, empty, kpts0, kpts1, desc0, desc1

        sim = desc0 @ desc1.t()
        sim_t = desc1 @ desc0.t()
        _, match01 = sim.max(dim=1)
        _, match10 = sim_t.max(dim=1)
        idx0 = torch.arange(len(match01), device=match01.device)
        mutual = match10[match01] == idx0

        if self.min_cossim > 0:
            best, _ = sim.max(dim=1)
            good = best > self.min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match01[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match01[mutual]

        mkpts0 = kpts0.index_select(0, idx0)
        mkpts1 = kpts1.index_select(0, idx1)
        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1
