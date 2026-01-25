from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from base_matcher import BaseMatcher
from _ort import create_session


def _softmax(x: np.ndarray, axis: int = 1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _logits_to_heatmap(kpt_logits: np.ndarray):
    # Matches matcher/liftfeat/onnx/demo_onnx.py
    scores_raw = _softmax(kpt_logits, axis=1)[:, :64]
    B, _, h_feat, w_feat = scores_raw.shape
    heat = (
        scores_raw.transpose(0, 2, 3, 1)
        .reshape(B, h_feat, w_feat, 8, 8)
        .transpose(0, 1, 3, 2, 4)
        .reshape(B, 1, h_feat * 8, w_feat * 8)
    )
    return heat.astype(np.float32)


def _simple_nms(heatmap_hw: np.ndarray, threshold: float = 0.01, kernel_size: int = 5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    local_max = cv2.dilate(heatmap_hw.astype(np.float32), kernel, iterations=1)
    eps = 1e-6
    return (heatmap_hw >= (local_max - eps)) & (heatmap_hw > float(threshold))


def _remap_sample(feat_chw: np.ndarray, x: np.ndarray, y: np.ndarray, H: int, W: int, interpolation):
    C, h_f, w_f = feat_chw.shape
    map_x = (x.astype(np.float32) * (w_f / float(W - 1)) - 0.5).reshape(-1, 1)
    map_y = (y.astype(np.float32) * (h_f / float(H - 1)) - 0.5).reshape(-1, 1)
    out = np.empty((len(x), C), dtype=np.float32)
    for i in range(C):
        sampled_i = cv2.remap(
            feat_chw[i].astype(np.float32),
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out[:, i] = sampled_i.reshape(-1)
    return out


def _extract_keypoints(heat: np.ndarray, desc_map: np.ndarray, top_k: int, threshold: float):
    heat_hw = heat[0, 0]
    H, W = heat_hw.shape

    peaks = _simple_nms(heat_hw, threshold=threshold, kernel_size=5)
    y, x = np.where(peaks)
    if len(x) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 64), dtype=np.float32),
        )

    kpts = np.stack([x, y], axis=1).astype(np.float32)

    dm = desc_map[0].astype(np.float32)
    denom = np.linalg.norm(dm, axis=0, keepdims=True)
    dm = dm / (denom + 1e-8)

    scores = _remap_sample(heat_hw[None, ...], x, y, H, W, interpolation=cv2.INTER_CUBIC)[:, 0]
    if top_k is not None and top_k > 0 and len(scores) > top_k:
        sel = np.argsort(scores)[::-1][:top_k]
        x, y = x[sel], y[sel]
        kpts = kpts[sel]
        scores = scores[sel]

    descs = _remap_sample(dm, x, y, H, W, interpolation=cv2.INTER_CUBIC)
    norm = np.linalg.norm(descs, axis=1, keepdims=True)
    descs = descs / (norm + 1e-8)
    return kpts, scores.astype(np.float32), descs.astype(np.float32)


def _match_mnn(desc0: np.ndarray, desc1: np.ndarray, min_cossim: float = -1.0):
    if len(desc0) == 0 or len(desc1) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    sims = desc0 @ desc1.T
    m01 = sims.argmax(axis=1)
    m10 = sims.argmax(axis=0)
    idx0 = np.arange(len(m01))
    mutual = m10[m01] == idx0
    if min_cossim > 0:
        good = sims[idx0, m01] > float(min_cossim)
        keep = mutual & good
    else:
        keep = mutual
    return idx0[keep].astype(np.int64), m01[keep].astype(np.int64)


class LiftFeatONNXMatcher(BaseMatcher):
    def __init__(
        self,
        device: str | None = None,
        dtype: str = "fp32",
        size: tuple[int, int] = (640, 480),
        top_k: int = 4096,
        detect_threshold: float = 0.005,
        min_cossim: float = -1.0,
        weights_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.dtype = str(dtype).lower()
        self.size = (int(size[0]), int(size[1]))
        self.top_k = int(top_k)
        self.detect_threshold = float(detect_threshold)
        self.min_cossim = float(min_cossim)

        weights_dir = Path(__file__).resolve().parent / "weights" / "liftfeat"
        if weights_path is None:
            W, H = self.size
            weights_path_p = weights_dir / f"liftfeat_{self.dtype}_{W}x{H}.onnx"
        else:
            weights_path_p = Path(weights_path)
        if not weights_path_p.exists():
            raise FileNotFoundError(f"Missing LiftFeat ONNX weights: {weights_path_p}")
        self.weights_path = weights_path_p

        try:
            self.session = create_session(str(self.weights_path), self.device)
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for matcher-onnx. Install with 'pip install onnxruntime' "
                "(CPU) or 'pip install onnxruntime-gpu' (CUDA)."
            ) from e

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
            kpt_logits, desc_map = self.session.run(None, {"image": inp})
            kpt_logits = kpt_logits.astype(np.float32)
            desc_map = desc_map.astype(np.float32)
            heat = _logits_to_heatmap(kpt_logits)
            kpts_r, _, descs = _extract_keypoints(heat, desc_map, top_k=self.top_k, threshold=self.detect_threshold)

            # Rescale keypoints back to original image coords.
            rw, rh = ow / float(W), oh / float(H)
            kpts = kpts_r * np.array([rw, rh], dtype=np.float32)
            return kpts, descs

        kpts0, desc0 = run(img0)
        kpts1, desc1 = run(img1)

        idx0, idx1 = _match_mnn(desc0, desc1, min_cossim=self.min_cossim)
        mkpts0 = kpts0[idx0] if len(idx0) else np.empty((0, 2), dtype=np.float32)
        mkpts1 = kpts1[idx1] if len(idx1) else np.empty((0, 2), dtype=np.float32)
        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1
