from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from base_matcher import BaseMatcher
from _ort import create_session


def _weights_dir() -> Path:
    return Path(__file__).resolve().parent / "weights" / "subpx"


def _match_indices(mkpts: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    if len(mkpts) == 0 or len(kpts) == 0:
        return np.zeros((0,), dtype=np.int64)
    d = {tuple(np.round(p, 3)): i for i, p in enumerate(kpts)}
    out = np.empty((len(mkpts),), dtype=np.int64)
    for i, p in enumerate(mkpts):
        key = tuple(np.round(p, 3))
        j = d.get(key)
        if j is not None:
            out[i] = j
            continue
        dist = np.linalg.norm(kpts - p[None, :], axis=1)
        out[i] = int(np.argmin(dist))
    return out


def _to_gray(img_chw: np.ndarray) -> np.ndarray:
    if img_chw.shape[0] == 1:
        return img_chw[0].astype(np.float32)
    r, g, b = img_chw[0], img_chw[1], img_chw[2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def _extract_patches_gray(img_chw: np.ndarray, kpts_xy: np.ndarray, patch_radius: int = 5) -> np.ndarray:
    """Patch extraction compatible with matcher/subpx/modules/keypt2subpx.py.

    - Uses grayscale image
    - Pads by patch_radius
    - Uses floor(keypoint) as top-left corner in padded image
    """

    gray = _to_gray(img_chw)
    ps = 2 * patch_radius + 1
    padded = np.pad(gray, ((patch_radius, patch_radius), (patch_radius, patch_radius)), mode="constant")
    Hp, Wp = padded.shape

    if len(kpts_xy) == 0:
        return np.zeros((1, 0, 1, ps, ps), dtype=np.float32)

    corners = np.floor(kpts_xy).astype(np.int64)
    corners[:, 0] = np.clip(corners[:, 0], 0, Wp - ps)
    corners[:, 1] = np.clip(corners[:, 1], 0, Hp - ps)

    patches = np.empty((len(corners), ps, ps), dtype=np.float32)
    for i, (x0, y0) in enumerate(corners):
        patches[i] = padded[y0 : y0 + ps, x0 : x0 + ps]

    return patches[:, None, :, :][None, ...]  # (1, N, 1, ps, ps)


class SubPXONNXMatcher(BaseMatcher):
    def __init__(
        self,
        device: str | None = None,
        mode: str = "xfeat-subpx",
        dtype: str = "fp32",
        size: tuple[int, int] = (640, 480),
        top_k: int = 1024,
        patch_radius: int = 5,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.mode = str(mode).lower()
        self.dtype = str(dtype).lower()
        self.size = (int(size[0]), int(size[1]))
        self.top_k = int(top_k)
        self.patch_radius = int(patch_radius)

        weights_dir = _weights_dir()
        if self.mode in ("xfeat-subpx", "xfeat-lightglue-subpx"):
            ref_name = f"k2s_xfeat_refiner_{self.dtype}.onnx"
            self.use_score = False
            self.desc_dim = 64

            from xfeat import XFeatONNXMatcher

            base_mode = "xfeat" if self.mode == "xfeat-subpx" else "lightglue"
            self.base = XFeatONNXMatcher(
                device=self.device,
                mode=base_mode,
                dtype=self.dtype,
                size=self.size,
                top_k=self.top_k,
            )
        elif self.mode == "superpoint-lightglue-subpx":
            ref_name = f"k2s_splg_refiner_{self.dtype}.onnx"
            self.use_score = True
            self.desc_dim = 256

            from superpoint_lightglue import SuperPointLightGlueONNXMatcher

            self.base = SuperPointLightGlueONNXMatcher(
                device=self.device,
                dtype=self.dtype,
                size=self.size,
                top_k=self.top_k,
            )
        else:
            raise ValueError(f"Unknown subpx matcher: {mode}")

        self.refiner_path = weights_dir / ref_name
        if not self.refiner_path.exists():
            raise FileNotFoundError(
                f"Missing SubPX refiner ONNX: {self.refiner_path}. "
                f"Export with: matcher/subpx/onnx/convert-onnx.py --matcher all --dtype {self.dtype.upper()}"
            )

        self.ref_sess = create_session(str(self.refiner_path), self.device)

        # For the SP variant, we need dense score maps. We recompute them from the
        # exported superpoint backbone ONNX (already used by the base matcher).
        self.sp_score_sess = None
        if self.use_score:
            w, h = self.size
            sp_path = Path(__file__).resolve().parent / "weights" / "lightglue" / f"superpoint_backbone_{self.dtype}_{w}x{h}.onnx"
            if not sp_path.exists():
                raise FileNotFoundError(f"Missing SuperPoint backbone ONNX for subpx: {sp_path}")
            self.sp_score_sess = create_session(str(sp_path), self.device)

    def _forward(self, img0: np.ndarray, img1: np.ndarray):
        # base matcher works in original image coords
        res = self.base(img0, img1)
        mkpts0 = res["matched_kpts0"].astype(np.float32)
        mkpts1 = res["matched_kpts1"].astype(np.float32)
        kpts0 = res["all_kpts0"].astype(np.float32)
        kpts1 = res["all_kpts1"].astype(np.float32)
        desc0 = res["all_desc0"].astype(np.float32)
        desc1 = res["all_desc1"].astype(np.float32)

        if len(mkpts0) == 0:
            return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1

        idx0 = _match_indices(mkpts0, kpts0)
        idx1 = _match_indices(mkpts1, kpts1)
        mdesc0 = desc0[idx0]
        mdesc1 = desc1[idx1]
        desc_mean = ((mdesc0 + mdesc1) / 2.0)[None, ...].astype(np.float32)

        patch0 = _extract_patches_gray(img0, mkpts0, patch_radius=self.patch_radius)
        patch1 = _extract_patches_gray(img1, mkpts1, patch_radius=self.patch_radius)

        if self.dtype == "fp16":
            patch0_i = patch0.astype(np.float16)
            patch1_i = patch1.astype(np.float16)
            desc_mean_i = desc_mean.astype(np.float16)
        else:
            patch0_i, patch1_i, desc_mean_i = patch0, patch1, desc_mean

        if self.use_score:
            assert self.sp_score_sess is not None
            w_base, h_base = self.size
            # Run score backbone at base size, then resize score map to original.
            def score_map(img_chw: np.ndarray):
                img_hwc = img_chw.transpose(1, 2, 0)
                oh, ow = img_hwc.shape[:2]
                resized = cv2.resize(img_hwc, (w_base, h_base))
                inp = resized.transpose(2, 0, 1)[None, ...]
                if self.dtype == "fp16":
                    inp = inp.astype(np.float16)
                else:
                    inp = inp.astype(np.float32)
                scores, _ = self.sp_score_sess.run(None, {"image": inp})
                scores = scores.astype(np.float32)[0, 0]
                scores = cv2.resize(scores, (ow, oh), interpolation=cv2.INTER_LINEAR)
                return scores

            s0 = score_map(img0)
            s1 = score_map(img1)
            score0 = _extract_patches_gray(s0[None, ...], mkpts0, patch_radius=self.patch_radius)
            score1 = _extract_patches_gray(s1[None, ...], mkpts1, patch_radius=self.patch_radius)

            if self.dtype == "fp16":
                score0_i = score0.astype(np.float16)
                score1_i = score1.astype(np.float16)
            else:
                score0_i, score1_i = score0, score1

            (d0,) = self.ref_sess.run(None, {"patch": patch0_i, "scorepatch": score0_i, "desc_mean": desc_mean_i})
            (d1,) = self.ref_sess.run(None, {"patch": patch1_i, "scorepatch": score1_i, "desc_mean": desc_mean_i})
        else:
            (d0,) = self.ref_sess.run(None, {"patch": patch0_i, "desc_mean": desc_mean_i})
            (d1,) = self.ref_sess.run(None, {"patch": patch1_i, "desc_mean": desc_mean_i})

        d0 = d0.astype(np.float32)[0]
        d1 = d1.astype(np.float32)[0]
        sub_mkpts0 = (mkpts0 + d0).astype(np.float32)
        sub_mkpts1 = (mkpts1 + d1).astype(np.float32)

        return sub_mkpts0, sub_mkpts1, kpts0, kpts1, desc0, desc1
