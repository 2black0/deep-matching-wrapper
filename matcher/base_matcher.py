
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
