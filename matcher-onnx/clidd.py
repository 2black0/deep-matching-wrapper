from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from base_matcher import BaseMatcher
from _ort import ort_providers


def _clidd_match(desc0: np.ndarray, desc1: np.ndarray, beta: float = 20.0, min_score: float = 0.01):
    """Numpy port of matcher/clidd/modules/clidd_wrapper.py:CLIDD.match."""
    if desc0.shape[0] == 0 or desc1.shape[0] == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    sim = desc0 @ desc1.T
    dist = np.exp((sim - 1.0) * float(beta))
    sum1 = dist.sum(axis=1, keepdims=True)
    sum2 = dist.sum(axis=0, keepdims=True)
    dist = (dist * dist) / (sum1 * sum2 + 1e-12)

    nn12 = dist.argmax(axis=1)
    nn21 = dist.argmax(axis=0)
    ids1 = np.arange(dist.shape[0])
    mutual = ids1 == nn21[nn12]

    scores = dist[ids1, nn12]
    keep = mutual & (scores > float(min_score))
    return ids1[keep].astype(np.int64), nn12[keep].astype(np.int64)


class CLIDDONNXMatcher(BaseMatcher):
    def __init__(
        self,
        device: str | None = None,
        model_name: str = "clidd-u128",
        dtype: str = "fp32",
        size: tuple[int, int] = (640, 480),
        score_thresh: float = -5.0,
        beta: float = 20.0,
        min_match_score: float = 0.01,
        weights_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.model_name = model_name.lower()
        self.dtype = str(dtype).lower()
        self.size = (int(size[0]), int(size[1]))
        self.score_thresh = float(score_thresh)
        self.beta = float(beta)
        self.min_match_score = float(min_match_score)

        cfg = self.model_name.split("-", 1)[1].upper()  # clidd-u128 -> U128
        W, H = self.size

        weights_dir = Path(__file__).resolve().parent / "weights" / "clidd"
        if weights_path is None:
            weights_path_p = weights_dir / f"clidd_{cfg.lower()}_{self.dtype}_{W}x{H}.onnx"
        else:
            weights_path_p = Path(weights_path)
        if not weights_path_p.exists():
            raise FileNotFoundError(f"Missing CLIDD ONNX weights: {weights_path_p}")
        self.weights_path = weights_path_p

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for matcher-onnx. Install with 'pip install onnxruntime' "
                "(CPU) or 'pip install onnxruntime-gpu' (CUDA)."
            ) from e

        self.session = ort.InferenceSession(str(self.weights_path), providers=ort_providers(self.device))

    def _forward(self, img0: np.ndarray, img1: np.ndarray):
        W, H = self.size

        def run(img_chw: np.ndarray):
            img_hwc = img_chw.transpose(1, 2, 0)
            oh, ow = img_hwc.shape[:2]
            resized = cv2.resize(img_hwc, (W, H))
            inp = resized.transpose(2, 0, 1)[None, ...]
            if self.dtype == "fp16":
                inp = inp.astype(np.float16)
            else:
                inp = inp.astype(np.float32)
            k, s, d = self.session.run(None, {"image": inp})
            k = k[0].astype(np.float32)
            s = s[0].astype(np.float32)
            d = d[0].astype(np.float32)

            keep = s > max(self.score_thresh, -1e7)
            kpts = k[keep]
            scores = s[keep]
            desc = d[keep]

            # Model keypoints are in resized coords; rescale back.
            rw, rh = ow / float(W), oh / float(H)
            kpts = kpts * np.array([rw, rh], dtype=np.float32)
            return kpts, scores, desc

        kpts0, _, desc0 = run(img0)
        kpts1, _, desc1 = run(img1)

        idx0, idx1 = _clidd_match(desc0, desc1, beta=self.beta, min_score=self.min_match_score)
        mkpts0 = kpts0[idx0] if len(idx0) else np.empty((0, 2), dtype=np.float32)
        mkpts1 = kpts1[idx1] if len(idx1) else np.empty((0, 2), dtype=np.float32)

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1
