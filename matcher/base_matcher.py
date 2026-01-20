
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as tfm
from pathlib import Path

from matcher.utils import to_normalized_coords, to_px_coords, to_numpy

class BaseMatcher(torch.nn.Module):
    """
    This serves as a base class for all matchers.
    """

    def __init__(self, device: str = None, **kwargs):
        super().__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.skip_ransac: bool = False

        # OpenCV default ransac params
        self.ransac_iters: int = kwargs.get("ransac_iters", 2000)
        self.ransac_conf: float = kwargs.get("ransac_conf", 0.95)
        self.ransac_reproj_thresh: float = kwargs.get("ransac_reproj_thresh", 3)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def load_image(path: str | Path, resize: int | tuple = None, rot_angle: float = 0) -> torch.Tensor:
        """load image from filesystem and return as tensor. Optionally rotate and resize."""
        if isinstance(resize, int):
            resize = (resize, resize)
        img = tfm.ToTensor()(Image.open(path).convert("RGB"))
        if resize is not None:
            img = tfm.Resize(resize, antialias=True)(img)
        img = tfm.functional.rotate(img, rot_angle)
        return img

    def rescale_coords(self, pts, h_orig, w_orig, h_new, w_new):
        """Rescale kpts coordinates from one img size to another"""
        return to_px_coords(to_normalized_coords(pts, h_new, w_new), h_orig, w_orig)

    def compute_ransac(self, matched_kpts0, matched_kpts1):
        """Process matches into inliers and the respective Homography using RANSAC."""
        if len(matched_kpts0) < 4 or self.skip_ransac:
            return None, np.empty([0, 2]), np.empty([0, 2])

        H, inliers_mask = cv2.findHomography(
            matched_kpts0,
            matched_kpts1,
            cv2.USAC_MAGSAC,
            self.ransac_reproj_thresh,
            self.ransac_conf,
            self.ransac_iters,
        )
        # handle case where H is None (not enough inliers found)
        if H is None:
             return None, np.empty([0, 2]), np.empty([0, 2])
             
        inliers_mask = inliers_mask[:, 0].astype(bool)
        inlier_kpts0 = matched_kpts0[inliers_mask]
        inlier_kpts1 = matched_kpts1[inliers_mask]

        return H, inlier_kpts0, inlier_kpts1

    @torch.inference_mode()
    def forward(self, img0, img1):
        """Run matching pipeline on two images."""
        # Take as input a pair of images (not a batch)
        if isinstance(img0, (str, Path)):
            img0 = BaseMatcher.load_image(img0)
        if isinstance(img1, (str, Path)):
            img1 = BaseMatcher.load_image(img1)

        assert isinstance(img0, torch.Tensor)
        assert isinstance(img1, torch.Tensor)

        img0 = img0.to(self.device)
        img1 = img1.to(self.device)

        # self._forward() is implemented by the children modules
        matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1 = self._forward(img0, img1)

        # Check that returned objects are of accepted types
        self.check_types(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1)

        # Convert torch tensors to numpy
        matched_kpts0, matched_kpts1 = to_numpy(matched_kpts0), to_numpy(matched_kpts1)
        all_kpts0, all_kpts1 = to_numpy(all_kpts0), to_numpy(all_kpts1)
        all_desc0, all_desc1 = to_numpy(all_desc0), to_numpy(all_desc1)

        # Handle None/empty
        matched_kpts0 = self.get_empty_array_if_none(matched_kpts0)
        matched_kpts1 = self.get_empty_array_if_none(matched_kpts1)
        all_kpts0 = self.get_empty_array_if_none(all_kpts0)
        all_kpts1 = self.get_empty_array_if_none(all_kpts1)
        all_desc0 = self.get_empty_array_if_none(all_desc0)
        all_desc1 = self.get_empty_array_if_none(all_desc1)

        # Check shapes
        self.check_shapes(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1)

        # Compute RANSAC
        H, inlier_kpts0, inlier_kpts1 = self.compute_ransac(matched_kpts0, matched_kpts1)

        return {
            "num_inliers": len(inlier_kpts0),
            "H": H,
            "all_kpts0": all_kpts0,
            "all_kpts1": all_kpts1,
            "all_desc0": all_desc0,
            "all_desc1": all_desc1,
            "matched_kpts0": matched_kpts0,
            "matched_kpts1": matched_kpts1,
            "inlier_kpts0": inlier_kpts0,
            "inlier_kpts1": inlier_kpts1,
        }

    @staticmethod
    def get_empty_array_if_none(array):
        if array is None or array.size == 0:
            return np.empty([0, 2])
        return array

    @staticmethod
    def check_types(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1):
        def is_array_or_tensor_or_none(data) -> bool:
            return data is None or isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)

        assert is_array_or_tensor_or_none(matched_kpts0)
        assert is_array_or_tensor_or_none(matched_kpts1)
        assert is_array_or_tensor_or_none(all_kpts0)
        assert is_array_or_tensor_or_none(all_kpts1)
        assert is_array_or_tensor_or_none(all_desc0)
        assert is_array_or_tensor_or_none(all_desc1)

    @staticmethod
    def check_shapes(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1):
        def check_kpts_shape(np_array) -> bool:
            return np_array.ndim == 2 and np_array.shape[1] == 2

        assert check_kpts_shape(matched_kpts0), f"matched_kpts0 {matched_kpts0.shape}"
        assert check_kpts_shape(matched_kpts1), f"matched_kpts1 {matched_kpts1.shape}"
        assert check_kpts_shape(all_kpts0), f"all_kpts0 {all_kpts0.shape}"
        assert check_kpts_shape(all_kpts1), f"all_kpts1 {all_kpts1.shape}"
        assert matched_kpts0.shape == matched_kpts1.shape
        
        # Descriptors check
        if all_desc0.size > 0:
             assert all_desc0.ndim == 2
        
        if all_desc1.size > 0:
             assert all_desc1.ndim == 2


# ============================================================================
# Matcher Registry
# ============================================================================

# List of all available matchers
AVAILABLE_MATCHERS = [
    # XFeat variants
    "xfeat", "xfeat-star", "xfeat-lightglue",
    # LiftFeat
    "liftfeat",
    # GIM (SuperPoint+LightGlue finetuned)
    "gim-lightglue",
    # EDM
    "edm",
    # Handcrafted features
    "orb-nn", "sift-nn", "sift-lightglue",
    # SuperPoint + LightGlue
    "superpoint-lightglue",
    # Subpixel refinement variants
    "xfeat-subpx", "xfeat-lightglue-subpx", "superpoint-lightglue-subpx",
    # CLIDD variants
    "clidd-a48", "clidd-n64", "clidd-t64", "clidd-s64", 
    "clidd-m64", "clidd-l64", "clidd-g128", "clidd-e128", "clidd-u128",
    # EfficientLoFTR
    "eloftr",
]

# Alias for compatibility with benchmark scripts
available_models = AVAILABLE_MATCHERS


def get_matcher(name: str, device: str = None, **kwargs):
    """
    Factory function to get a matcher instance by name.
    
    Args:
        name: Name of the matcher (case-insensitive)
        device: Device to run on ('cuda' or 'cpu')
        **kwargs: Additional arguments passed to matcher constructor
        
    Returns:
        BaseMatcher instance
        
    Raises:
        ValueError: If matcher name is not recognized
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    name = name.lower()
    
    # XFeat variants
    if "xfeat" in name:
        from matcher.xfeat import XFeatMatcher
        
        if name == "xfeat":
            return XFeatMatcher(device=device, mode='xfeat', **kwargs)
        elif name == "xfeat-star":
            return XFeatMatcher(device=device, mode='xfeat-star', **kwargs)
        elif name == "xfeat-lightglue":
            return XFeatMatcher(device=device, mode='xfeat-lightglue', **kwargs)
        elif name == "xfeat-subpx":
            from matcher.subpx import Keypt2SubpxMatcher
            return Keypt2SubpxMatcher(device=device, mode='xfeat-subpx', **kwargs)
        elif name == "xfeat-lightglue-subpx":
            from matcher.subpx import Keypt2SubpxMatcher
            return Keypt2SubpxMatcher(device=device, mode='xfeat-lightglue-subpx', **kwargs)
    
    # LiftFeat
    elif "liftfeat" in name:
        from matcher.liftfeat import LiftFeatMatcher
        return LiftFeatMatcher(device=device, **kwargs)
    
    # GIM (SuperPoint+LightGlue)
    elif "gim" in name:
        from matcher.gim import GIMMatcher
        return GIMMatcher(device=device, **kwargs)
    
    # SuperPoint + LightGlue
    elif "superpoint-lightglue" in name and "subpx" not in name:
        from matcher.lightglue import SuperPointLightGlueMatcher
        return SuperPointLightGlueMatcher(device=device, **kwargs)
    
    # Subpixel refinement (SuperPoint variant)
    elif "superpoint-lightglue" in name and "subpx" in name:
        from matcher.subpx import Keypt2SubpxMatcher
        return Keypt2SubpxMatcher(device=device, mode='superpoint-lightglue-subpx', **kwargs)
    
    # EDM
    elif "edm" in name:
        from matcher.edm import EDMMatcher
        return EDMMatcher(device=device, **kwargs)
    
    # CLIDD variants
    elif "clidd" in name:
        from matcher.clidd import CLIDDMatcher
        return CLIDDMatcher(device=device, model_name=name, **kwargs)
    
    # EfficientLoFTR
    elif "eloftr" in name or "efficient-loftr" in name:
        from matcher.eloftr import EfficientLoFTRMatcher
        return EfficientLoFTRMatcher(device=device, **kwargs)
    
    # Handcrafted features
    elif "orb" in name or "sift" in name:
        from matcher.handcrafted import HandcraftedMatcher
        return HandcraftedMatcher(device=device, method=name, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown matcher: '{name}'. "
            f"Available matchers: {', '.join(AVAILABLE_MATCHERS)}"
        )
