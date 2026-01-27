from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from base_matcher import BaseMatcher
from _ort import create_session


class EDMONNXMatcher(BaseMatcher):
    """EDM (Efficient Dense Matching) ONNX matcher.
    
    EDM outputs dense coarse + fine matches directly from concatenated grayscale images.
    The ONNX model takes (1, 2, H, W) input where:
      - Channel 0: image0 grayscale
      - Channel 1: image1 grayscale
    
    Output is (N, 11) where each row contains:
      [x0, y0, x1, y1, offset_x0, offset_y0, offset_x1, offset_y1, conf, mconf, ...]
      
    We extract x0, y0, x1, y1 as the matched keypoints after applying fine offsets.
    """

    def __init__(
        self,
        device: str | None = None,
        dtype: str = "fp32",
        size: tuple[int, int] = (640, 480),
        weights_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.dtype = str(dtype).lower()
        self.size = (int(size[0]), int(size[1]))

        W, H = self.size

        weights_dir = Path(__file__).resolve().parent / "weights" / "edm"
        if weights_path is None:
            weights_path_p = weights_dir / f"edm_{self.dtype}_{W}x{H}.onnx"
        else:
            weights_path_p = Path(weights_path)
        if not weights_path_p.exists():
            raise FileNotFoundError(f"Missing EDM ONNX weights: {weights_path_p}")
        self.weights_path = weights_path_p

        try:
            self.session = create_session(str(self.weights_path), self.device)
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for matcher-onnx. Install with 'pip install onnxruntime' "
                "(CPU) or 'pip install onnxruntime-gpu' (CUDA)."
            ) from e

    def _forward(self, img0: np.ndarray, img1: np.ndarray):
        """Forward pass through EDM ONNX model.
        
        Args:
            img0: (3, H, W) RGB float32 in [0, 1]
            img1: (3, H, W) RGB float32 in [0, 1]
            
        Returns:
            mkpts0, mkpts1, kpts0, kpts1, desc0, desc1
        """
        W, H = self.size

        def preprocess(img_chw: np.ndarray):
            """Convert RGB to grayscale and resize to target size."""
            # Convert CHW RGB to HWC
            img_hwc = img_chw.transpose(1, 2, 0)
            oh, ow = img_hwc.shape[:2]
            
            # Convert to grayscale (ITU-R BT.601 formula)
            gray = 0.299 * img_hwc[:, :, 0] + 0.587 * img_hwc[:, :, 1] + 0.114 * img_hwc[:, :, 2]
            
            # Resize to target size
            gray_resized = cv2.resize(gray, (W, H), interpolation=cv2.INTER_LINEAR)
            
            return gray_resized, (ow, oh)

        gray0, (ow0, oh0) = preprocess(img0)
        gray1, (ow1, oh1) = preprocess(img1)

        # Stack into (1, 2, H, W)
        img_pair = np.stack([gray0, gray1], axis=0)[None, ...]  # (1, 2, H, W)
        
        if self.dtype == "fp16":
            img_pair = img_pair.astype(np.float16)
        else:
            img_pair = img_pair.astype(np.float32)

        # Run inference
        output = self.session.run(None, {"input": img_pair})[0]  # (N, 11)
        
        if output.shape[0] == 0:
            # No matches
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                None,
                None,
                None,
                None,
            )

        # Extract coordinates from output
        # Format: [x0_coarse, y0_coarse, x1_coarse, y1_coarse, 
        #          offset_x0, offset_y0, offset_x1, offset_y1, conf, mconf, ...]
        x0_coarse = output[:, 0]
        y0_coarse = output[:, 1]
        x1_coarse = output[:, 2]
        y1_coarse = output[:, 3]
        offset_x0 = output[:, 4]
        offset_y0 = output[:, 5]
        offset_x1 = output[:, 6]
        offset_y1 = output[:, 7]

        # Apply fine offsets to get final coordinates
        x0 = x0_coarse + offset_x0
        y0 = y0_coarse + offset_y0
        x1 = x1_coarse + offset_x1
        y1 = y1_coarse + offset_y1

        mkpts0 = np.stack([x0, y0], axis=1).astype(np.float32)
        mkpts1 = np.stack([x1, y1], axis=1).astype(np.float32)

        # Rescale coordinates back to original image size
        mkpts0[:, 0] *= ow0 / float(W)
        mkpts0[:, 1] *= oh0 / float(H)
        mkpts1[:, 0] *= ow1 / float(W)
        mkpts1[:, 1] *= oh1 / float(H)

        # EDM doesn't provide separate keypoint extraction, only dense matches
        return mkpts0, mkpts1, None, None, None, None
