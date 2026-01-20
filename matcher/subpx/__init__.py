
from pathlib import Path
import torch
import numpy as np
from matcher.base_matcher import BaseMatcher
from matcher.utils import to_numpy, to_tensor
from matcher.subpx.modules.keypt2subpx import Keypt2Subpx

# Import local matchers
from matcher.xfeat import XFeatMatcher
from matcher.gim import GIMMatcher
from matcher.lightglue import SuperPointLightGlueMatcher

class Keypt2SubpxMatcher(BaseMatcher):
    # Maps our mode names to Keypt2Subpx detector names and configs
    CONFIGS = {
        "superpoint-lightglue-subpx": {
            "detector": "splg",
            "output_dim": 256,
            "use_score": True,
            "matcher_cls": SuperPointLightGlueMatcher,
            "matcher_kwargs": {},
            "weights_name": "k2s_splg_pretrained.pth"
        },
        "xfeat-subpx": {
            "detector": "xfeat",
            "output_dim": 64,
            "use_score": False,
            "matcher_cls": XFeatMatcher,
            "matcher_kwargs": {"mode": "xfeat"},
            "weights_name": "k2s_xfeat_pretrained.pth"
        },
        "xfeat-lightglue-subpx": {
            "detector": "xfeat",
            "output_dim": 64,
            "use_score": False,
            "matcher_cls": XFeatMatcher,
            "matcher_kwargs": {"mode": "xfeat-lightglue"},
            "weights_name": "k2s_xfeat_pretrained.pth"
        },
    }

    def __init__(self, device="cpu", mode="xfeat-subpx", **kwargs):
        super().__init__(device, **kwargs)
        
        if mode not in self.CONFIGS:
            raise ValueError(f"Unknown mode: {mode}. Supported: {list(self.CONFIGS.keys())}")
            
        self.mode = mode
        self.conf = self.CONFIGS[mode]
        
        # Initialize base matcher
        self.matcher = self.conf["matcher_cls"](device=device, **self.conf["matcher_kwargs"])
        
        # Initialize Keypt2Subpx
        self.keypt2subpx = Keypt2Subpx(
            output_dim=self.conf["output_dim"],
            use_score=self.conf["use_score"]
        ).to(self.device).eval()
        
        # Load weights
        self._load_weights()

    def _load_weights(self):
        filename = self.conf["weights_name"]
        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / filename
        
        if not weights_path.exists():
            print(f"Downloading Keypt2Subpx weights to {weights_path}...")
            detector_code = self.conf["detector"]
            url = f"https://github.com/KimSinjeong/keypt2subpx/raw/master/pretrained/k2s_{detector_code}_pretrained.pth"
            torch.hub.download_url_to_file(url, weights_path)
            
        print(f"loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        self.keypt2subpx.net.load_state_dict(state_dict)
        self.keypt2subpx.eval().to(self.device)

    def get_match_idxs(self, mkpts: np.ndarray | torch.Tensor, kpts: np.ndarray | torch.Tensor) -> np.ndarray:
        idxs = []
        # Ensure numpy
        mkpts_np = to_numpy(mkpts)
        kpts_np = to_numpy(kpts)
        
        # Simple nearest neighbor or exact match search
        # Since mkpts are a subset of kpts (usually), we can find exact matches.
        # But float precision might be an issue.
        # Using a KDTree or just manual search.
        # For efficiency with exact matches:
        
        # Optimization: Create a dict mapping coords to idx
        # Rounding to avoid float issues
        kpts_dict = {tuple(np.round(kp, 3)): i for i, kp in enumerate(kpts_np)}
        
        for mkpt in mkpts_np:
            key = tuple(np.round(mkpt, 3))
            if key in kpts_dict:
                idxs.append(kpts_dict[key])
            else:
                # Fallback: nearest neighbor?
                # If exact match fails, something is wrong or mkpts are refined beyond kpts?
                # But typically mkpts are just selected from kpts.
                # Let's try finding closest
                dists = np.linalg.norm(kpts_np - mkpt, axis=1)
                best_idx = np.argmin(dists)
                if dists[best_idx] < 1e-3: # Strict threshold
                    idxs.append(best_idx)
                else:
                    # Could happen if matcher refines points internally? 
                    # If so, subpx might fail or we should just skip?
                    # Append 0 or handle error.
                    idxs.append(0) 
        
        return np.asarray(idxs)

    def _forward(self, img0, img1):
        # Run base matcher
        # BaseMatcher returns: (mkpts0, mkpts1, kpts0, kpts1, desc0, desc1)
        # All inputs to _forward are already preprocessed/handled by BaseMatcher wrapper
        # BUT self.matcher is a BaseMatcher too, so we can call its _forward directly?
        # NO, we should call it like a callable to ensure its preprocess is used if needed.
        # But BaseMatcher.__call__ handles image loading. Here img0/img1 are already tensors/arrays from OUR __call__.
        # We should call self.matcher._forward(img0, img1).
        
        res = self.matcher._forward(img0, img1)
        mkpts0, mkpts1, kpts0, kpts1, desc0, desc1 = res
        
        # If no matches, return early
        if mkpts0 is None or len(mkpts0) == 0:
             return res

        # Prepare for refinement
        # We need original keypoints' descriptors corresponding to the matches.
        # If GIM/XFeat returns all_kpts and all_desc, we can index.
        # But GIM returns None for kpts/desc if not exposed? 
        # Wait, GIM wrapper provided kpts and desc in the tuple now?
        # My GIM wrapper returns (mkpts0, mkpts1, kpts0, kpts1, desc0, desc1). Yes.
        # My XFeat wrapper returns (mkpts0, mkpts1, all_kpts0, all_kpts1, all_desc0, all_desc1). Yes.
        
        # So we have full kpts and desc.
        
        mkpts0_t = to_tensor(mkpts0, self.device)
        mkpts1_t = to_tensor(mkpts1, self.device)
        
        # We need descriptors for the MATCHED keypoints.
        # Option A: descriptors are already returned for MATCHED keypoints?
        # No, conventions:
        # XFeat/GIM: desc0 is ALL descriptors.
        # But wait, XFeat wrapper returns `features["descriptors"]` which are ALL descriptors.
        # We need to map matched keypoints to their descriptors.
        
        # Find indices of matches in all_kpts
        idx0 = self.get_match_idxs(mkpts0, kpts0)
        idx1 = self.get_match_idxs(mkpts1, kpts1)
        
        # Select descriptors
        # desc0 shape (N, D) or (1, N, D) or (B, N, D) depending on wrapper
        # BaseMatcher usually returns squeezed arrays/tensors if possible?
        # Let's check GIM wrapper again: returns (N, D) tensors.
        # XFeat wrapper: returns (N, D) tensors (I verified squeeeze logic).
        
        mdesc0 = desc0[idx0]
        mdesc1 = desc1[idx1]
        
        # Prepare scores
        scores0, scores1 = None, None
        if self.conf["use_score"]:
            # Only supported for GIM currently which stores last_scoremaps
            if hasattr(self.matcher, "last_scoremaps"):
                scores0 = self.matcher.last_scoremaps.get(0)
                scores1 = self.matcher.last_scoremaps.get(1)
            else:
                 print("Warning: use_score is True but matcher has no last_scoremaps.")
        
        # Ensure scores are on device
        if scores0 is not None: 
            scores0 = scores0.to(self.device).float()
            if scores0.ndim == 2: scores0 = scores0.unsqueeze(0)
        if scores1 is not None: 
            scores1 = scores1.to(self.device).float()
            if scores1.ndim == 2: scores1 = scores1.unsqueeze(0)
        
        mdesc0 = to_tensor(mdesc0, self.device).float()
        mdesc1 = to_tensor(mdesc1, self.device).float()
        
        # Keypt2Subpx expects:
        # keypt: (N, 2)
        # img: (1, C, H, W)? No, (C, H, W) based on code: C, H, W = img1.shape
        # wrappers pass img as (1, C, H, W) (BaseMatcher standard)?
        # Let's check BaseMatcher. _forward inputs are result of load_image -> preprocess.
        # BaseMatcher.load_image returns (C, H, W).
        # Wrapper preprocess usually adds batch dim if model needs it.
        # But we need simple (C, H, W) for extract_patches probably. 
        # Check keypt2subpx.py: img1.shape calls for C, H, W.
        # But extract_patches expects batch dim or no?
        # keypt2subpx.py line 138: img1 < 1.0. 
        # line 141: C, H, W = img1.shape.
        # So it expects 3D tensor.
        
        # Check img0 input. If it has batch dim `(1, C, H, W)`, squeeze it.
        if img0.ndim == 4: img0_sq = img0.squeeze(0)
        else: img0_sq = img0
        
        if img1.ndim == 4: img1_sq = img1.squeeze(0)
        else: img1_sq = img1
            
        params = {
            "keypt1": mkpts0_t,
            "keypt2": mkpts1_t,
            "img1": img0_sq.to(self.device),
            "img2": img1_sq.to(self.device),
            "desc1": mdesc0,
            "desc2": mdesc1,
            "score1": scores0,
            "score2": scores1
        }
        
        # Run Refinement
        with torch.no_grad():
            sub_mkpts0, sub_mkpts1 = self.keypt2subpx(**params)
            
        return (
            sub_mkpts0,
            sub_mkpts1,
            kpts0,
            kpts1,
            desc0,
            desc1
        )
