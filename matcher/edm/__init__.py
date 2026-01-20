
import sys
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as tfm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from matcher.base_matcher import BaseMatcher
from matcher.utils import resize_to_divisible

# Setup modules path
edm_modules_path = Path(__file__).parent / "modules"
sys.path.append(str(edm_modules_path))

from matcher.edm.modules.edm import EDM
from matcher.edm.modules.default_config import get_cfg_defaults
from matcher.edm.modules.misc import lower_config

class EDMMatcher(BaseMatcher):
    divisible_size = 32

    def __init__(self, device="cpu", thresh=0.2, **kwargs):
        super().__init__(device, **kwargs)
        self.thresh = thresh
        self.matcher = self.build_matcher()

    def build_matcher(self):
        # Get default configurations
        config = get_cfg_defaults()
        
        # Manually apply outdoor/megadepth settings for inference
        # From original edm_base.py (outdoor)
        config.TRAINER.CANONICAL_BS = 32
        config.TRAINER.CANONICAL_LR = 2e-3
        config.TRAINER.SCALING = 8
        config.TRAINER.EPI_ERR_THR = 1e-4
        config.EDM.TRAIN_RES_H = 832
        config.EDM.TRAIN_RES_W = 832
        config.TRAINER.N_VAL_PAIRS_TO_PLOT = 32

        # From megadepth_test_1500.py (inference settings)
        config.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
        config.EDM.TEST_RES_H = 1152
        config.EDM.TEST_RES_W = 1152
        config.EDM.NECK.NPE = [832, 832, 1152, 1152]

        config.EDM.COARSE.MCONF_THR = self.thresh
        config.EDM.COARSE.BORDER_RM = 2
        config = lower_config(config)

        # Initialize matcher
        matcher = EDM(config=config["edm"])

        # Load weights
        weights_name = "edm.safetensors"
        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / weights_name
        
        if not weights_path.exists():
            print(f"Downloading EDM weights to {weights_path}...")
            # Use huggingface_hub to download directly to our location
            # Or download to cache and copy? hf_hub_download downloads to cache.
            # We can use cache directly.
            cached_path = hf_hub_download(repo_id="image-matching-models/edm", filename=weights_name)
            # Symbolic link or just use cached path? 
            # For "weights directory" consistency, let's copy (or just load from cache if permissible, but user asked for weights in matcher/edm/weights)
            # Actually, let's just use hf_hub_download default behavior but maybe `local_dir`?
            # Creating symlink or copy is better.
            import shutil
            shutil.copy(cached_path, weights_path)
            
        print(f"loading weights from: {weights_path}")
        matcher.load_state_dict(load_file(weights_path))

        return matcher.eval().to(self.device)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        # EDM expects (1, 1, H, W) for single channel? 
        # IMM EDM wrapper uses Grayscale.
        # EDM paper: "The feature extraction network takes grayscale images as input"
        return tfm.Grayscale()(img).unsqueeze(0), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        batch = {"image0": img0, "image1": img1}

        self.matcher(batch)

        mkpts0 = batch["mkpts0_f"]
        mkpts1 = batch["mkpts1_f"]

        # Rescale coordinates to original resolution
        H0, W0 = img0.shape[-2:]
        H1, W1 = img1.shape[-2:]
        
        # mkpts are in resized coordinates.
        # self.rescale_coords is in BaseMatcher? Not default.
        # I need to rescale myself.
        
        def rescale_coords(kpts, h_orig, w_orig, h_new, w_new):
            kpts = kpts.clone()
            kpts[:, 0] *= w_orig / w_new
            kpts[:, 1] *= h_orig / h_new
            return kpts

        mkpts0 = rescale_coords(mkpts0, img0_orig_shape[0], img0_orig_shape[1], H0, W0)
        mkpts1 = rescale_coords(mkpts1, img1_orig_shape[0], img1_orig_shape[1], H1, W1)

        return mkpts0, mkpts1, None, None, None, None
