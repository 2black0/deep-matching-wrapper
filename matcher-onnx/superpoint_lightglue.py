from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from base_matcher import BaseMatcher
from _ort import ort_providers


def _dilate(x: np.ndarray, radius: int):
    k = 2 * radius + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(x.astype(np.float32), kernel, iterations=1)


def _simple_nms(scores_hw: np.ndarray, nms_radius: int):
    if nms_radius <= 0:
        return scores_hw.astype(np.float32)
    zeros = np.zeros_like(scores_hw, dtype=np.float32)
    local_max = _dilate(scores_hw, nms_radius)
    eps = 1e-6
    max_mask = scores_hw >= (local_max - eps)
    for _ in range(2):
        supp = _dilate(max_mask.astype(np.float32), nms_radius) > 0
        supp_scores = np.where(supp, zeros, scores_hw)
        new_local_max = _dilate(supp_scores, nms_radius)
        new_max_mask = supp_scores >= (new_local_max - eps)
        max_mask = max_mask | (new_max_mask & (~supp))
    return np.where(max_mask, scores_hw, zeros).astype(np.float32)


def _sample_descriptors(keypoints_xy: np.ndarray, desc_map: np.ndarray, image_hw: tuple[int, int], s: int = 8):
    H, W = image_hw
    C, h, w = desc_map.shape
    if len(keypoints_xy) == 0:
        return np.zeros((0, C), dtype=np.float32)

    k = keypoints_xy.astype(np.float32).copy()
    k[:, 0] = k[:, 0] - s / 2.0 + 0.5
    k[:, 1] = k[:, 1] - s / 2.0 + 0.5
    denom_x = (w * s - s / 2.0 - 0.5)
    denom_y = (h * s - s / 2.0 - 0.5)
    u = k[:, 0] / float(denom_x) * 2.0 - 1.0
    v = k[:, 1] / float(denom_y) * 2.0 - 1.0
    map_x = ((u + 1.0) * 0.5 * (w - 1)).reshape(-1, 1).astype(np.float32)
    map_y = ((v + 1.0) * 0.5 * (h - 1)).reshape(-1, 1).astype(np.float32)

    out = np.empty((len(k), C), dtype=np.float32)
    for i in range(C):
        sampled_i = cv2.remap(
            desc_map[i].astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out[:, i] = sampled_i.reshape(-1)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / (norm + 1e-8)


def _filter_matches(log_assignment: np.ndarray, th: float = 0.1):
    scores = log_assignment[:, :-1, :-1]
    max0_val = scores.max(axis=2)
    max1_val = scores.max(axis=1)
    m0 = scores.argmax(axis=2)
    m1 = scores.argmax(axis=1)

    idx0 = np.arange(m0.shape[1])[None, :]
    idx1 = np.arange(m1.shape[1])[None, :]
    mutual0 = idx0 == np.take_along_axis(m1, m0, axis=1)
    mutual1 = idx1 == np.take_along_axis(m0, m1, axis=1)

    ms0 = np.exp(max0_val)
    zero = np.array(0.0, dtype=ms0.dtype)
    mscores0 = np.where(mutual0, ms0, zero)
    mscores1 = np.where(mutual1, np.take_along_axis(mscores0, m1, axis=1), zero)
    valid0 = mutual0 & (mscores0 > float(th))
    valid1 = mutual1 & np.take_along_axis(valid0, m1, axis=1)

    m0_out = np.where(valid0, m0, -1)
    m1_out = np.where(valid1, m1, -1)
    return m0_out, m1_out


class SuperPointLightGlueONNXMatcher(BaseMatcher):
    def __init__(
        self,
        device: str | None = None,
        dtype: str = "fp32",
        size: tuple[int, int] = (640, 480),
        top_k: int = 1024,
        nms_radius: int = 4,
        detect_threshold: float = 0.0005,
        remove_borders: int = 4,
        min_conf: float = 0.1,
        sp_weights_path: str | Path | None = None,
        lg_weights_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.dtype = str(dtype).lower()
        self.size = (int(size[0]), int(size[1]))
        self.top_k = int(top_k)
        self.nms_radius = int(nms_radius)
        self.detect_threshold = float(detect_threshold)
        self.remove_borders = int(remove_borders)
        self.min_conf = float(min_conf)

        weights_dir = Path(__file__).resolve().parent / "weights" / "lightglue"
        W, H = self.size

        if sp_weights_path is None:
            sp_weights_path = weights_dir / f"superpoint_backbone_{self.dtype}_{W}x{H}.onnx"
        if lg_weights_path is None:
            lg_weights_path = weights_dir / f"superpoint_lightglue_{self.dtype}_k{self.top_k}.onnx"

        self.sp_weights_path = Path(sp_weights_path)
        self.lg_weights_path = Path(lg_weights_path)
        if not self.sp_weights_path.exists():
            raise FileNotFoundError(f"Missing SuperPoint ONNX weights: {self.sp_weights_path}")
        if not self.lg_weights_path.exists():
            raise FileNotFoundError(f"Missing LightGlue ONNX weights: {self.lg_weights_path}")

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for matcher-onnx. Install with 'pip install onnxruntime' "
                "(CPU) or 'pip install onnxruntime-gpu' (CUDA)."
            ) from e

        providers = ort_providers(self.device)
        self.sp_sess = ort.InferenceSession(str(self.sp_weights_path), providers=providers)
        self.lg_sess = ort.InferenceSession(str(self.lg_weights_path), providers=providers)

    def _forward(self, img0: np.ndarray, img1: np.ndarray):
        W, H = self.size

        def run_backbone(img_chw: np.ndarray):
            img_hwc = img_chw.transpose(1, 2, 0)
            oh, ow = img_hwc.shape[:2]
            resized = cv2.resize(img_hwc, (W, H))
            inp = resized.transpose(2, 0, 1)[None, ...]
            if self.dtype == "fp16":
                inp = inp.astype(np.float16)
            else:
                inp = inp.astype(np.float32)
            scores, desc_map = self.sp_sess.run(None, {"image": inp})
            scores = scores.astype(np.float32)[0, 0]
            desc_map = desc_map.astype(np.float32)[0]
            return scores, desc_map, (ow, oh)

        s0, d0m, orig0 = run_backbone(img0)
        s1, d1m, orig1 = run_backbone(img1)

        s0_nms = _simple_nms(s0, nms_radius=self.nms_radius)
        s1_nms = _simple_nms(s1, nms_radius=self.nms_radius)
        b = self.remove_borders
        if b > 0:
            s0_nms[:b, :] = -1
            s0_nms[-b:, :] = -1
            s0_nms[:, :b] = -1
            s0_nms[:, -b:] = -1
            s1_nms[:b, :] = -1
            s1_nms[-b:, :] = -1
            s1_nms[:, :b] = -1
            s1_nms[:, -b:] = -1

        y0, x0 = np.where(s0_nms > self.detect_threshold)
        y1, x1 = np.where(s1_nms > self.detect_threshold)
        scores0 = s0_nms[y0, x0].astype(np.float32)
        scores1 = s1_nms[y1, x1].astype(np.float32)

        order0 = np.argsort(scores0)[::-1][: min(self.top_k, len(scores0))]
        order1 = np.argsort(scores1)[::-1][: min(self.top_k, len(scores1))]
        x0, y0 = x0[order0], y0[order0]
        x1, y1 = x1[order1], y1[order1]

        kpts0_resized = np.stack([x0.astype(np.float32), y0.astype(np.float32)], axis=1)
        kpts1_resized = np.stack([x1.astype(np.float32), y1.astype(np.float32)], axis=1)
        desc0 = _sample_descriptors(kpts0_resized, d0m, image_hw=(H, W), s=8)
        desc1 = _sample_descriptors(kpts1_resized, d1m, image_hw=(H, W), s=8)

        ow0, oh0 = orig0
        ow1, oh1 = orig1
        rw0, rh0 = ow0 / float(W), oh0 / float(H)
        rw1, rh1 = ow1 / float(W), oh1 / float(H)
        kpts0 = kpts0_resized * np.array([rw0, rh0], dtype=np.float32)
        kpts1 = kpts1_resized * np.array([rw1, rh1], dtype=np.float32)

        K = self.top_k
        kp0 = np.zeros((1, K, 2), dtype=np.float32)
        kp1 = np.zeros((1, K, 2), dtype=np.float32)
        ds0 = np.zeros((1, K, 256), dtype=np.float32)
        ds1 = np.zeros((1, K, 256), dtype=np.float32)
        n0, n1 = len(kpts0), len(kpts1)
        kp0[0, :n0] = kpts0
        kp1[0, :n1] = kpts1
        ds0[0, :n0] = desc0
        ds1[0, :n1] = desc1

        if self.dtype == "fp16":
            kp0_i = kp0.astype(np.float16)
            kp1_i = kp1.astype(np.float16)
            ds0_i = ds0.astype(np.float16)
            ds1_i = ds1.astype(np.float16)
        else:
            kp0_i, kp1_i, ds0_i, ds1_i = kp0, kp1, ds0, ds1

        sz0 = np.array([[ow0, oh0]], dtype=np.int64)
        sz1 = np.array([[ow1, oh1]], dtype=np.int64)
        (log_assignment,) = self.lg_sess.run(
            None,
            {
                "keypoints0": kp0_i,
                "descriptors0": ds0_i,
                "keypoints1": kp1_i,
                "descriptors1": ds1_i,
                "image_size0": sz0,
                "image_size1": sz1,
            },
        )
        log_assignment = log_assignment.astype(np.float32)
        m0, _ = _filter_matches(log_assignment, th=self.min_conf)
        idx0 = np.where(m0[0] > -1)[0].astype(np.int64)
        idx1 = m0[0, idx0].astype(np.int64)

        keep = (idx0 < n0) & (idx1 < n1)
        idx0, idx1 = idx0[keep], idx1[keep]
        mkpts0 = kpts0[idx0] if len(idx0) else np.empty((0, 2), dtype=np.float32)
        mkpts1 = kpts1[idx1] if len(idx1) else np.empty((0, 2), dtype=np.float32)

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1
