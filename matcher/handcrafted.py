
import cv2
import torch
import numpy as np
from matcher.base_matcher import BaseMatcher

class HandcraftedMatcher(BaseMatcher):
    def __init__(self, device="cpu", method="orb-nn", **kwargs):
        super().__init__(device, **kwargs)
        self.method = method
        self.max_keypoints = kwargs.get("max_keypoints", 2048)
        
        if "orb" in method:
            self.detector = cv2.ORB_create(nfeatures=self.max_keypoints)
        elif "sift" in method:
            self.detector = cv2.SIFT_create(nfeatures=self.max_keypoints)
            
        if "lightglue" in method:
             from matcher.gim.modules.lightglue import LightGlue
             from pathlib import Path
             
             weights_name = "sift_lightglue"
             weights_path = Path(__file__).parent / "gim/weights" / f"{weights_name}.pth"
             weights_path.parent.mkdir(parents=True, exist_ok=True)
             
             if not weights_path.exists():
                 print(f"Downloading {weights_name} weights...")
                 url = f"https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/{weights_name}.pth"
                 torch.hub.download_url_to_file(url, weights_path)
                 
             print(f"loading weights from: {weights_path}")
             self.lightglue = LightGlue({
                 "weights": str(weights_path), 
                 "input_dim": 128, 
                 "descriptor_dim": 256, 
                 "filter_threshold": 0.1
             }).eval().to(self.device)

    def _forward(self, img0, img1):
        # img0, img1 are tensors (B, C, H, W) or (C, H, W).
        
        # Convert to numpy uint8 grayscale for OpenCV detector
        def to_cv2(img):
            if img.ndim == 4: img = img.squeeze(0)
            img = img.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            if img.shape[2] == 1:
                img = img.squeeze(2)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return img
            
        npy0 = to_cv2(img0)
        npy1 = to_cv2(img1)
        
        kp0, des0 = self.detector.detectAndCompute(npy0, None)
        kp1, des1 = self.detector.detectAndCompute(npy1, None)
        
        # Format keypoints: (N, 2)
        kpts0 = np.array([k.pt for k in kp0], dtype=np.float32)
        kpts1 = np.array([k.pt for k in kp1], dtype=np.float32)
        
        if des0 is None: des0 = np.zeros((0, self.detector.descriptorSize()), dtype=np.float32)
        if des1 is None: des1 = np.zeros((0, self.detector.descriptorSize()), dtype=np.float32)
        
        # Match
        if "nn" in self.method:
            # Nearest Neighbor with cross check or ratio test
            # Standard: BFMatcher
            if "orb" in self.method:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                
            matches = bf.match(des0, des1)
            # matches is DMatch list
            
            # Sorted by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) > 0:
                mkpts0 = np.array([kpts0[m.queryIdx] for m in matches])
                mkpts1 = np.array([kpts1[m.trainIdx] for m in matches])
            else:
                mkpts0 = np.zeros((0, 2))
                mkpts1 = np.zeros((0, 2))
            
        elif "lightglue" in self.method:
             # Prepare for LightGlue
             if len(kpts0) == 0 or len(kpts1) == 0:
                 mkpts0 = np.zeros((0, 2))
                 mkpts1 = np.zeros((0, 2))
             else:
                 with torch.inference_mode():
                     # LightGlue expects tensors:
                     # keypoints: (1, N, 2)
                     # descriptors: (1, N, D)
                     kpts0_t = torch.from_numpy(kpts0).float().unsqueeze(0).to(self.device)
                     kpts1_t = torch.from_numpy(kpts1).float().unsqueeze(0).to(self.device)
                     des0_t = torch.from_numpy(des0).float().unsqueeze(0).to(self.device)
                     des1_t = torch.from_numpy(des1).float().unsqueeze(0).to(self.device)
                     
                     data = {
                         "keypoints0": kpts0_t,
                         "keypoints1": kpts1_t,
                         "descriptors0": des0_t,
                         "descriptors1": des1_t,
                         "image_size0": torch.tensor(npy0.shape[::-1]).unsqueeze(0).to(self.device), # W, H
                         "image_size1": torch.tensor(npy1.shape[::-1]).unsqueeze(0).to(self.device)
                     }
                     
                     res = self.lightglue(data)
                     
                     m0 = res['matches0'][0] # indices of matches in 0
                     valid = m0 > -1
                     m_indices_0 = torch.where(valid)[0]
                     m_indices_1 = m0[valid]
                     
                     mkpts0 = kpts0_t[0][m_indices_0].cpu().numpy()
                     mkpts1 = kpts1_t[0][m_indices_1].cpu().numpy()
             
        return mkpts0, mkpts1, kpts0, kpts1, des0, des1

