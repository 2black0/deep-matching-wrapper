import sys
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

# Add LightGlue to path - NO LONGER NEEDED with proper package structure
# params_path = Path(__file__).parent / "modules" / "LightGlue"
# if str(params_path) not in sys.path:
#     sys.path.append(str(params_path))

from .modules.lightglue import LightGlue
from .modules.superpoint import SuperPoint
from .modules import rbd

from matcher.base_matcher import BaseMatcher

class SuperPointLightGlueMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, **kwargs):
        super().__init__(device, **kwargs)
        
        self.device = device
        
        # Paths for weights
        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # SuperPoint Weights (Official)
        # LightGlue repo usually handles this internally via torch.hub but we want local management if possible
        # However, LightGlue.SuperPoint() loads from URL by default if path not provided.
        # Let's inspect source or just rely on its default caching if user didn't specify strict local weight manual load.
        # User asked for "weightsnya di matcher/lightglue/weights".
        
        # Check SuperPoint.py in LightGlue: it uses `torch.load(path)`
        # We should download explicitly to our weights dir.
        
        # Repo: gloack/SuperPoint-LightGlue-weights ? Or standard?
        # LightGlue/superpoint.py uses: https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth
        
        sp_weights_path = weights_dir / "superpoint_v1.pth"
        if not sp_weights_path.exists():
            print(f"Downloading SuperPoint weights to {sp_weights_path}...")
            torch.hub.download_url_to_file(
                "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth",
                sp_weights_path
            )
            
        # LightGlue Weights: superpoint_lightglue.pth
        lg_weights_path = weights_dir / "superpoint_lightglue.pth"
        if not lg_weights_path.exists():
             print(f"Downloading LightGlue (SuperPoint) weights to {lg_weights_path}...")
             torch.hub.download_url_to_file(
                "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth",
                lg_weights_path
             )

        print(f"loading weights from: {sp_weights_path}")
        print(f"loading weights from: {lg_weights_path}")
        
        # Initialize models
        # Note: SuperPoint class in LightGlue repo takes `max_num_keypoints` in init (or extract dict?)
        # Let's verify usage. Usually: SuperPoint(max_num_keypoints=...)
        # But wait, looking at imm/im_models/lightglue.py: 
        # self.extractor = SuperPoint(max_num_keypoints=...).eval()
        
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
        self.extractor.load_state_dict(torch.load(sp_weights_path, map_location=device))
        
        self.matcher = LightGlue(features='superpoint').eval().to(device)
        
        # Load and fix state dict keys (older weights vs newer code compatibility)
        state_dict = torch.load(lg_weights_path, map_location=device)
        
        # Check if remapping is needed (if keys start with 'self_attn')
        # The new LightGlue code expects 'transformers.X.self_attn' logic
        # But 'v0.1_arxiv' weights have 'self_attn.X' logic or 'matcher.self_attn.X'
        
        # Standardize keys
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("matcher.", "")
            
            # Remap attention blocks
            for i in range(self.matcher.conf.n_layers):
                 k = k.replace(f"self_attn.{i}.", f"transformers.{i}.self_attn.")
                 k = k.replace(f"cross_attn.{i}.", f"transformers.{i}.cross_attn.")
            
            new_state_dict[k] = v
            
        # The model expects 'confidence_thresholds' but weights might miss it?
        # Actually LightGlue usually registers this buffer. 
        # If strict=False, it might optimize it out if not found? 
        # But let's try strict=False first if minor keys are missing.
        # But transformers structural mismatch MUST be fixed.
        
        self.matcher.load_state_dict(new_state_dict, strict=False)
        
        self.last_scoremaps = {}

    def _forward(self, img0, img1):
        # BaseMatcher handles loading to tensor (C, H, W) normalized 0-1
        
        # LightGlue expects (1, C, H, W)
        if img0.ndim == 3: img0 = img0.unsqueeze(0)
        if img1.ndim == 3: img1 = img1.unsqueeze(0)
        
        # Extract features
        feats0 = self.extractor.extract(img0)
        # Check if extractor exposed scores (modified SuperPoint)
        scores0_map = getattr(self.extractor, "last_scoremaps", None)
        
        feats1 = self.extractor.extract(img1)
        scores1_map = getattr(self.extractor, "last_scoremaps", None)
        
        # Store for consumers
        self.last_scoremaps = {}
        if scores0_map is not None:
             # Handle batch dim? extractor extract usually handles batch. 
             # scores0_map is (B, H, W). We usually feed batch=1.
             self.last_scoremaps[0] = scores0_map[0] if scores0_map.ndim > 2 else scores0_map
        if scores1_map is not None:
             self.last_scoremaps[1] = scores1_map[0] if scores1_map.ndim > 2 else scores1_map
        
        # Match
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        
        # Unpack
        # feats keys: keypoints, descriptors, etc.
        # matches keys: matches, scores
        
        # Remove batch dim [0]
        kpts0 = feats0['keypoints'][0]
        kpts1 = feats1['keypoints'][0]
        desc0 = feats0['descriptors'][0]
        desc1 = feats1['descriptors'][0]
        
        matches = matches01['matches'][0] # indices (N, 2)
        scores = matches01['scores'][0]
        
        # Indices
        param0 = matches[:, 0]
        param1 = matches[:, 1]
        
        mkpts0 = kpts0[param0]
        mkpts1 = kpts1[param1]
        
        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1
