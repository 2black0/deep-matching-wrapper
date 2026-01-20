
import torch
import numpy as np
from pathlib import Path

from matcher.base_matcher import BaseMatcher
from .modules import CLIDD

class CLIDDMatcher(BaseMatcher):
    def __init__(self, device="cpu", model_name="clidd-e128", **kwargs):
        super().__init__(device, **kwargs)
        
        # Parse model config from name
        # model_name format: clidd-{config}, e.g., clidd-e128 -> E128
        # The user requested specific names: clidd-a48, clidd-e128, etc.
        # configs keys are in CLIDD.cfgs: A48, N64, T64, S64, M64, L64, G128, E128, U128
        
        parts = model_name.split('-')
        if len(parts) < 2:
             raise ValueError(f"Invalid model name for CLIDD: {model_name}. Expected format 'clidd-CONFIG', e.g. 'clidd-e128'")
        
        cfg_suffix = parts[1].upper()
        
        if cfg_suffix not in CLIDD.cfgs:
             raise ValueError(f"Unknown CLIDD config: {cfg_suffix}. Available: {list(CLIDD.cfgs.keys())}")

        self.cfg_name = cfg_suffix
        
        # Weights path
        # Assuming weights are in matcher/clidd/weights/{CONFIG}.pth
        weights_dir = Path(__file__).parent / "weights"
        state_dict_path = weights_dir / f"{cfg_suffix}.pth"
        
        if not state_dict_path.exists():
            print(f"Warning: Weights not found at {state_dict_path}")
            # Try lowercase just in case specific file naming differs, though list_dir showed E128.pth etc.
            # list_dir showed: A48.pth, E128.pth, G128.pth, L64.pth, M64.pth, N64.pth, S64.pth, T64.pth, U128.pth
            # So uppercase is correct.
        
        print(f"Loading CLIDD model: {self.cfg_name} from {state_dict_path}")
        
        # Initialize CLIDD
        # CLIDD __init__(self, cfg, top_k, radius=2, score=-5, weights_path=None)
        # We use default params from clidd.py or reasonable defaults?
        # clidd.py defaults: top_k arg required, radius=2, score=-5
        # We can pass these via kwargs or use defaults.
        
        top_k = kwargs.get("top_k", 2048) # Default top_k to 2048 matches standard
        radius = kwargs.get("nms_radius", 2)
        score_thresh = kwargs.get("detection_threshold", -5) # CLIDD uses very low threshold? Or maybe logit?
        # In clidd.py: detect2 = raw_detect > self.score_thresh. raw_detect comes from model().
        # Model returns score via self.score_head which is Conv/ReLU/Conv/PixelShuffle. No Sigmoid?
        # Let's trust -5 is sensible if it's logits.
        
        self.model = CLIDD(
            cfg=self.cfg_name,
            top_k=top_k, 
            radius=radius,
            score=score_thresh,
            weights_path=state_dict_path
        ).to(self.device).eval()

    def _forward(self, img0, img1):
        # BaseMatcher handles loading and device moving. img0/img1 are (C, H, W) tensors.
        # CLIDD forward expects: (B, C, H, W)
        
        if img0.ndim == 3: img0 = img0.unsqueeze(0)
        if img1.ndim == 3: img1 = img1.unsqueeze(0)
        
        # Extract features
        # forward returns list of dicts [{'keypoints', 'scores', 'descriptors'}]
        res0 = self.model(img0)[0]
        res1 = self.model(img1)[0]
        
        kpts0 = res0['keypoints']
        desc0 = res0['descriptors']
        scores0 = res0['scores'] # Not used by base return but useful
        
        kpts1 = res1['keypoints']
        desc1 = res1['descriptors']
        scores1 = res1['scores']
        
        # Match
        # CLIDD match(desc0, desc1)
        # Returns idxs1, idxs2 (indices of matches)
        idxs0, idxs1 = self.model.match(desc0, desc1)
        
        if len(idxs0) == 0:
             mkpts0 = np.empty((0, 2))
             mkpts1 = np.empty((0, 2))
        else:
             mkpts0 = kpts0[idxs0]
             mkpts1 = kpts1[idxs1]
        
        return (
            mkpts0, 
            mkpts1, 
            kpts0, 
            kpts1, 
            desc0, 
            desc1
        )
