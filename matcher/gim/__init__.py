
from pathlib import Path
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from kornia.color import rgb_to_grayscale

from matcher.base_matcher import BaseMatcher
from matcher.gim.modules.superpoint import SuperPoint
from matcher.gim.modules.lightglue import LightGlue

class GIMMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, **kwargs):
        super().__init__(device, **kwargs)
        
        # Paths for weights
        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        repo_id = "image-matching-models/gim-lightglue"
        
        # GIM LightGlue weights
        ckpt_path = weights_dir / "gim_lightglue_100h.ckpt"
        if not ckpt_path.exists():
            print(f"Downloading GIM-LightGlue weights to {ckpt_path}...")
            hf_hub_download(repo_id=repo_id, filename="gim_lightglue_100h.ckpt", local_dir=weights_dir)
            
        # SuperPoint v1 weights
        sp_weights_path = weights_dir / "superpoint_v1.pth"
        if not sp_weights_path.exists():
            print(f"Downloading SuperPoint weights to {sp_weights_path}...")
            hf_hub_download(repo_id=repo_id, filename="superpoint_v1.pth", local_dir=weights_dir)
        
        # Initialize SuperPoint
        self.detector = SuperPoint(
            {
                "max_num_keypoints": max_num_keypoints,
                "force_num_keypoints": True,
                "detection_threshold": 0.0,
                "nms_radius": 3,
                "trainable": False,
                # "weights_path": str(sp_weights_path), # We load manually to be safe
            }
        )
        
        # Initialize LightGlue
        self.model = LightGlue(
            {
                "filter_threshold": 0.1,
                "flash": False,
                "checkpointed": True,
            }
        )
        
        # Load weights
        self._load_weights(ckpt_path, sp_weights_path)
        
        self.detector = self.detector.eval().to(self.device)
        self.model = self.model.eval().to(self.device)

    def _load_weights(self, ckpt_path, sp_weights_path):
        # Load SuperPoint
        # Original imm code loads from ckpt? No, let's see.
        # imm code loads detector from ckpt but also sets weights_path to superpoint_v1.pth
        # But verify logic: 
        # state_dict = torch.load(ckpt_path) -> remove model., remove superpoint. -> load to detector
        # THEN state_dict = torch.load(ckpt_path) -> remove superpoint., remove model. -> load to model
        
        # Wait, usually GIM is finetuned LightGlue on SuperPoint.
        # Let's check GIM implementation in imm again.
        
        # In imm/im_models/gim.py:
        # self.ckpt_path = dowload(gim_lightglue_100h.ckpt)
        # self.superpoint_v1_path = download(superpoint_v1.pth)
        # 
        # load_weights:
        # 1. load ckpt_path
        # 2. extract 'superpoint.xxx' -> detector
        # 3. extract 'model.xxx' (LightGlue) -> model
        
        # So superpoint_v1.pth is maybe unused? Or used as initialization?
        # "weights_path": self.superpoint_v1_path in SuperPoint init. 
        # SuperPoint.__init__ calls _init calls load_state_dict if commented out line 204 was active.
        # But in SuperPoint class provided by GIM, _init doesn't load weights.
        
        # So we should follow imm logic exactly: load both from the 100h ckpt?
        # Wait, imm code loads detector state_dict from ckpt_path?
        
        # imm code:
        # for k in list(state_dict.keys()):
        #     if k.startswith("model."): state_dict.pop(k)
        #     if k.startswith("superpoint."): state_dict[k.replace("superpoint.", "", 1)] = state_dict.pop(k)
        # self.detector.load_state_dict(state_dict)
        
        # This implies the ckpt contains both superpoint (finetuned?) and lightglue.
        
        # Let's do exactly as imm.
        
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
            
        # Load detector
        sp_dict = {}
        for k, v in state_dict.items():
            if k.startswith("superpoint."):
                sp_dict[k.replace("superpoint.", "", 1)] = v
            # If there are keys that are NOT model. and NOT superpoint., what are they?
            # Use careful logic.
            
        # If sp_dict is empty, maybe it relies on superpoint_v1.pth?
        # But the imm code does: self.detector.load_state_dict(state_dict) (modified inplace)
        
        # Re-reading imm logic:
        # state_dict loaded.
        # remove "model." keys.
        # rename "superpoint." keys.
        # load to detector.
        
        # RELOAD state_dict (fresh)
        # remove "superpoint." keys
        # rename "model." keys.
        # load to model.
        
        # So yes, both come from the ckpt.
        
        # 1. Detector
        full_dict = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in full_dict:
             full_dict = full_dict["state_dict"]
             
        det_dict = {}
        lg_dict = {}
        
        for k, v in full_dict.items():
            if k.startswith("superpoint."):
                det_dict[k.replace("superpoint.", "", 1)] = v
            elif k.startswith("model."):
                lg_dict[k.replace("model.", "", 1)] = v
                
        if det_dict:
            self.detector.load_state_dict(det_dict, strict=False)
        else:
            # Fallback to superpoint_v1.pth if not in ckpt (unlikely for GIM)
             print("Warning: No SuperPoint weights in GIM ckpt, loading stock weights.")
             print(f"loading weights from: {sp_weights_path}")
             self.detector.load_state_dict(torch.load(sp_weights_path, map_location='cpu'))

        print(f"loading weights from: {ckpt_path}")
        self.model.load_state_dict(lg_dict, strict=False)

    def preprocess(self, img):
        # convert to grayscale and add batch dim if needed
        if img.ndim == 3 and img.shape[0] == 3:
            img = rgb_to_grayscale(img)
            
        if img.ndim == 3:
            img = img.unsqueeze(0)
        elif img.ndim == 2:
            img = img.unsqueeze(0).unsqueeze(0)
            
        return img

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        data = dict(image0=img0, image1=img1)

        scale0 = torch.tensor([1.0, 1.0]).to(self.device)[None]
        scale1 = torch.tensor([1.0, 1.0]).to(self.device)[None]

        size0 = torch.tensor(data["image0"].shape[-2:][::-1])[None].to(self.device)
        size1 = torch.tensor(data["image1"].shape[-2:][::-1])[None].to(self.device)

        data.update(dict(size0=size0, size1=size1))
        data.update(dict(scale0=scale0, scale1=scale1))
        
        # SuperPoint (Detector)
        # We need to create specific input dicts
        pred = {}
        self.last_scoremaps = {}
        
        # Image 0
        input0 = {"image": data["image0"], "image_size": data["size0"]}
        out0 = self.detector(input0)
        self.last_scoremaps[0] = out0["keypoint_scores"]
        pred.update({k + '0': v for k, v in out0.items()})
        
        # Image 1
        input1 = {"image": data["image1"], "image_size": data["size1"]}
        out1 = self.detector(input1)
        self.last_scoremaps[1] = out1["keypoint_scores"]
        pred.update({k + '1': v for k, v in out1.items()})
        
        # LightGlue
        # LightGlue expects "resize0/1" keys similar to size
        lg_input = {**pred, **data, "resize0": data["size0"], "resize1": data["size1"]}
        lg_out = self.model(lg_input)
        pred.update(lg_out)

        # Post-process
        kpts0 = pred["keypoints0"][0] # (N, 2)
        kpts1 = pred["keypoints1"][0] # (N, 2)
        desc0 = pred["descriptors0"][0]
        desc1 = pred["descriptors1"][0]
        
        matches = pred["matches"][0] # (K, 2) indices
        mscores = pred["scores"][0] # (K,)
        
        # mkpts (matched keypoints)
        # matches contains indices into kpts0 and kpts1
        # matches is (K, 2) where col 0 is idx in kpts0, col 1 is idx in kpts1
        
        valid_matches = matches 
        
        mkpts0 = kpts0[valid_matches[:, 0]]
        mkpts1 = kpts1[valid_matches[:, 1]]
        
        return (
            mkpts0,
            mkpts1,
            kpts0,
            kpts1,
            desc0,
            desc1,
        )
