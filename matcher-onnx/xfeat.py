from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np

from base_matcher import BaseMatcher
from _ort import ort_providers


def _softmax(x: np.ndarray, axis: int):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _get_kpts_heatmap(kpt_logits: np.ndarray, softmax_temp: float = 1.0):
    scores = _softmax(kpt_logits * float(softmax_temp), axis=1)[:, :64]
    B, _, h, w = scores.shape
    heat = (
        scores.transpose(0, 2, 3, 1)
        .reshape(B, h, w, 8, 8)
        .transpose(0, 1, 3, 2, 4)
        .reshape(B, 1, h * 8, w * 8)
    )
    return heat.astype(np.float32)


def _nms_peaks(heatmap_hw: np.ndarray, threshold: float = 0.05, kernel_size: int = 5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    local_max = cv2.dilate(heatmap_hw.astype(np.float32), kernel, iterations=1)
    eps = 1e-6
    peaks = (heatmap_hw >= (local_max - eps)) & (heatmap_hw > float(threshold))
    y, x = np.where(peaks)
    return x.astype(np.int64), y.astype(np.int64)


def _remap_sample_chw(feat_chw: np.ndarray, x: np.ndarray, y: np.ndarray, H: int, W: int, interpolation):
    C, h_f, w_f = feat_chw.shape
    if len(x) == 0:
        return np.zeros((0, C), dtype=np.float32)
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


def _extract_sparse(
    desc_map: np.ndarray,
    kpt_logits: np.ndarray,
    rel_map: np.ndarray,
    orig_size_wh: tuple[int, int],
    resized_size_wh: tuple[int, int],
    top_k: int,
    det_thresh: float,
    softmax_temp: float,
):
    W, H = resized_size_wh
    heat = _get_kpts_heatmap(kpt_logits, softmax_temp=softmax_temp)[0, 0]
    x, y = _nms_peaks(heat, threshold=det_thresh, kernel_size=5)
    if len(x) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 64), dtype=np.float32),
        )

    s_nearest = heat[y, x]
    rel = _remap_sample_chw(rel_map[0].astype(np.float32), x, y, H, W, interpolation=cv2.INTER_LINEAR)[:, 0]
    scores = (s_nearest * rel).astype(np.float32)

    order = np.argsort(scores)[::-1]
    if top_k is not None and top_k > 0:
        order = order[: min(int(top_k), len(order))]
    x, y, scores = x[order], y[order], scores[order]

    descs = _remap_sample_chw(desc_map[0].astype(np.float32), x, y, H, W, interpolation=cv2.INTER_CUBIC)
    norm = np.linalg.norm(descs, axis=1, keepdims=True)
    descs = descs / (norm + 1e-8)

    kpts_r = np.stack([x.astype(np.float32), y.astype(np.float32)], axis=1)
    ow, oh = orig_size_wh
    rw, rh = ow / float(W), oh / float(H)
    kpts = kpts_r * np.array([rw, rh], dtype=np.float32)

    valid = scores > 0
    return kpts[valid], scores[valid], descs[valid]


def _match_mnn(desc0: np.ndarray, desc1: np.ndarray, min_cossim: float = 0.82):
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


def _filter_matches(log_assignment: np.ndarray, th: float = 0.1):
    scores = log_assignment[:, :-1, :-1]
    max0_val = scores.max(axis=2)
    m0 = scores.argmax(axis=2)
    m1 = scores.argmax(axis=1)

    idx0 = np.arange(m0.shape[1])[None, :]
    mutual0 = idx0 == np.take_along_axis(m1, m0, axis=1)
    ms0 = np.exp(max0_val)
    valid0 = mutual0 & (ms0 > float(th))
    m0_out = np.where(valid0, m0, -1)
    return m0_out


class XFeatONNXMatcher(BaseMatcher):
    def __init__(
        self,
        device: str | None = None,
        mode: str = "xfeat",
        dtype: str = "fp32",
        size: tuple[int, int] = (640, 480),
        top_k: int = 1024,
        detect_threshold: float = 0.05,
        softmax_temp: float = 1.0,
        min_cossim: float = 0.82,
        min_conf: float = 0.1,
        fine_conf: float = 0.25,
        star_scales: tuple[float, float] = (0.6, 1.3),
        star_multiscale: bool = True,
        backbone_path: str | Path | None = None,
        lightglue_path: str | Path | None = None,
        finematcher_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.mode = str(mode).lower()  # "xfeat" | "lightglue" | "star"
        self.dtype = str(dtype).lower()
        self.size = (int(size[0]), int(size[1]))
        self.top_k = int(top_k)
        self.detect_threshold = float(detect_threshold)
        self.softmax_temp = float(softmax_temp)
        self.min_cossim = float(min_cossim)
        self.min_conf = float(min_conf)
        self.fine_conf = float(fine_conf)
        self.star_scales = (float(star_scales[0]), float(star_scales[1]))
        self.star_multiscale = bool(star_multiscale)

        weights_dir = Path(__file__).resolve().parent / "weights" / "xfeat"
        W, H = self.size

        def backbone_for_size(w: int, h: int) -> Path:
            return weights_dir / f"xfeat_backbone_{self.dtype}_{w}x{h}.onnx"

        def backbone_star_for_size(w: int, h: int) -> Path:
            return weights_dir / f"xfeat_backbone_star_{self.dtype}_{w}x{h}.onnx"

        if self.mode == "star":
            # Use the same sizes as the PyTorch multi-scale pipeline (after floor-to-32).
            def floor32(x: int) -> int:
                return max(32, (int(x) // 32) * 32)

            self.star_levels: list[dict] = []
            for s in (self.star_scales if self.star_multiscale else (1.0,)):
                w_in = int(round(W * float(s)))
                h_in = int(round(H * float(s)))
                w32 = floor32(w_in)
                h32 = floor32(h_in)
                rw = w_in / float(w32)
                rh = h_in / float(h32)
                self.star_levels.append(
                    {
                        "scale": float(s),
                        "w_in": w_in,
                        "h_in": h_in,
                        "w32": w32,
                        "h32": h32,
                        "rw": float(rw),
                        "rh": float(rh),
                    }
                )

            # In star mode we may need multiple backbones.
            self.backbone_paths = []
            self.star_sizes = tuple((lvl["w32"], lvl["h32"]) for lvl in self.star_levels)
            for (ww, hh) in self.star_sizes:
                # Prefer the star-specific (unnormalized) backbone, but allow fallback.
                p = backbone_star_for_size(ww, hh)
                if backbone_path is not None:
                    p = Path(backbone_path)
                if not p.exists() and backbone_path is None:
                    p = backbone_for_size(ww, hh)
                if not p.exists():
                    cmds = []
                    for (w_need, h_need) in self.star_sizes:
                        cmds.append(
                            f"matcher/xfeat/onnx/convert-onnx.py --matcher xfeat-star --dtype {self.dtype.upper()} --size {w_need} {h_need}"
                        )
                    raise FileNotFoundError(
                        f"Missing XFeat backbone ONNX for xfeat-star at {ww}x{hh}: {p}. "
                        f"Required backbones for size {W}x{H} with scales {self.star_scales}: {self.star_sizes}. "
                        f"Export with: " + " ; ".join(cmds)
                    )
                self.backbone_paths.append(p)
            self.backbone_path = self.backbone_paths[0]
        else:
            if backbone_path is None:
                backbone_path = backbone_for_size(W, H)
            self.backbone_path = Path(backbone_path)
            if not self.backbone_path.exists():
                raise FileNotFoundError(f"Missing XFeat backbone ONNX: {self.backbone_path}")

        if self.mode == "lightglue":
            if lightglue_path is None:
                lightglue_path = weights_dir / f"xfeat_lighterglue_{self.dtype}_k{self.top_k}.onnx"
            self.lightglue_path = Path(lightglue_path)
            if not self.lightglue_path.exists():
                raise FileNotFoundError(f"Missing XFeat LighterGlue ONNX: {self.lightglue_path}")
        else:
            self.lightglue_path = None

        if self.mode == "star":
            if finematcher_path is None:
                finematcher_path = weights_dir / f"xfeat_finematcher_{self.dtype}.onnx"
            self.finematcher_path = Path(finematcher_path)
            if not self.finematcher_path.exists():
                raise FileNotFoundError(f"Missing XFeat fine matcher ONNX: {self.finematcher_path}")
        else:
            self.finematcher_path = None

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for matcher-onnx. Install with 'pip install onnxruntime' "
                "(CPU) or 'pip install onnxruntime-gpu' (CUDA)."
            ) from e

        providers = ort_providers(self.device)
        if self.mode == "star":
            self.backbone_sess = None
            self.backbone_sess_by_size: dict[tuple[int, int], ort.InferenceSession] = {}
            for p in getattr(self, "backbone_paths", [self.backbone_path]):
                m = re.search(r"(\d+)x(\d+)(?:\.onnx)?$", p.name)
                if not m:
                    raise ValueError(f"Cannot infer backbone size from filename: {p.name}")
                key = (int(m.group(1)), int(m.group(2)))
                self.backbone_sess_by_size[key] = ort.InferenceSession(str(p), providers=providers)
            self.lg_sess = None
            self.fm_sess = ort.InferenceSession(str(self.finematcher_path), providers=providers)
        else:
            self.backbone_sess = ort.InferenceSession(str(self.backbone_path), providers=providers)
            self.lg_sess = (
                ort.InferenceSession(str(self.lightglue_path), providers=providers) if self.lightglue_path else None
            )
            self.fm_sess = None

    def _forward(self, img0: np.ndarray, img1: np.ndarray):
        W, H = self.size

        if self.mode == "star":
            return self._forward_star(img0, img1)

        def run_backbone(img_chw: np.ndarray):
            img_hwc = img_chw.transpose(1, 2, 0)
            oh, ow = img_hwc.shape[:2]
            resized = cv2.resize(img_hwc, (W, H))
            inp = resized.transpose(2, 0, 1)[None, ...]
            if self.dtype == "fp16":
                inp = inp.astype(np.float16)
            else:
                inp = inp.astype(np.float32)
            d_map, k_logits, r_map = self.backbone_sess.run(None, {"image": inp})
            d_map = d_map.astype(np.float32)
            k_logits = k_logits.astype(np.float32)
            r_map = r_map.astype(np.float32)
            kpts, _, desc = _extract_sparse(
                d_map,
                k_logits,
                r_map,
                orig_size_wh=(ow, oh),
                resized_size_wh=(W, H),
                top_k=self.top_k,
                det_thresh=self.detect_threshold,
                softmax_temp=self.softmax_temp,
            )
            return kpts, desc, (ow, oh)

        kpts0, desc0, orig0 = run_backbone(img0)
        kpts1, desc1, orig1 = run_backbone(img1)

        if self.mode == "xfeat":
            idx0, idx1 = _match_mnn(desc0, desc1, min_cossim=self.min_cossim)
        else:
            assert self.lg_sess is not None
            n = self.top_k
            n0, n1 = len(kpts0), len(kpts1)
            kp0 = np.zeros((1, n, 2), dtype=np.float32)
            kp1 = np.zeros((1, n, 2), dtype=np.float32)
            ds0 = np.zeros((1, n, 64), dtype=np.float32)
            ds1 = np.zeros((1, n, 64), dtype=np.float32)
            kp0[0, : min(n0, n)] = kpts0[: min(n0, n)]
            kp1[0, : min(n1, n)] = kpts1[: min(n1, n)]
            ds0[0, : min(n0, n)] = desc0[: min(n0, n)]
            ds1[0, : min(n1, n)] = desc1[: min(n1, n)]

            ow0, oh0 = orig0
            ow1, oh1 = orig1
            sz0 = np.array([[ow0, oh0]], dtype=np.int64)
            sz1 = np.array([[ow1, oh1]], dtype=np.int64)

            if self.dtype == "fp16":
                kp0_i = kp0.astype(np.float16)
                kp1_i = kp1.astype(np.float16)
                ds0_i = ds0.astype(np.float16)
                ds1_i = ds1.astype(np.float16)
            else:
                kp0_i, kp1_i, ds0_i, ds1_i = kp0, kp1, ds0, ds1

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
            m0 = _filter_matches(log_assignment, th=self.min_conf)
            idx0 = np.where(m0[0] > -1)[0].astype(np.int64)
            idx1 = m0[0, idx0].astype(np.int64)

            keep = (idx0 < n0) & (idx1 < n1)
            idx0, idx1 = idx0[keep], idx1[keep]

        mkpts0 = kpts0[idx0] if len(idx0) else np.empty((0, 2), dtype=np.float32)
        mkpts1 = kpts1[idx1] if len(idx1) else np.empty((0, 2), dtype=np.float32)
        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1

    def _forward_star(self, img0: np.ndarray, img1: np.ndarray):
        # Implements the same high-level pipeline as XFeat.match_xfeat_star:
        # coarse (dense) extraction -> MNN matching -> fine refinement.

        base_w, base_h = self.size
        # Output coordinates are in the configured base size; rescale to original at the end.
        img0_hwc0 = img0.transpose(1, 2, 0)
        img1_hwc0 = img1.transpose(1, 2, 0)
        oh0, ow0 = img0_hwc0.shape[:2]
        oh1, ow1 = img1_hwc0.shape[:2]
        sx0, sy0 = ow0 / float(base_w), oh0 / float(base_h)
        sx1, sy1 = ow1 / float(base_w), oh1 / float(base_h)

        def run_backbone_at(img_chw: np.ndarray, w: int, h: int):
            img_hwc = img_chw.transpose(1, 2, 0)
            resized = cv2.resize(img_hwc, (w, h))
            inp = resized.transpose(2, 0, 1)[None, ...]
            if self.dtype == "fp16":
                inp = inp.astype(np.float16)
            else:
                inp = inp.astype(np.float32)
            sess = self.backbone_sess_by_size[(w, h)]
            d_map, _, rel = sess.run(None, {"image": inp})
            d_map = d_map.astype(np.float32)[0]  # (64,h/8,w/8)
            rel = rel.astype(np.float32)[0, 0]  # (h/8,w/8)
            return d_map, rel

        def extract_dense(img_chw: np.ndarray, w: int, h: int, topk: int, rw: float, rh: float):
            desc_map, rel = run_backbone_at(img_chw, w, h)
            hh, ww = rel.shape

            # Grid of (x,y) at feature resolution -> pixel coords (*8)
            y, x = np.meshgrid(np.arange(hh, dtype=np.int64), np.arange(ww, dtype=np.int64), indexing="ij")
            xy = np.stack([x.reshape(-1), y.reshape(-1)], axis=1).astype(np.float32) * 8.0
            scores = rel.reshape(-1)

            k = min(int(topk), len(scores))
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(scores[idx])[::-1]]

            kpts = xy[idx]
            # Match torch preprocess_tensor() scale correction.
            kpts = kpts * np.array([float(rw), float(rh)], dtype=np.float32)
            desc_flat = desc_map.transpose(1, 2, 0).reshape(-1, 64)
            desc = desc_flat[idx].astype(np.float32)
            return kpts.astype(np.float32), desc

        # Multi-scale coarse extraction (20% + 80%)
        if self.star_multiscale and len(getattr(self, "star_levels", ())) >= 2:
            l1, l2 = self.star_levels[0], self.star_levels[1]
            w1, h1, rw1, rh1, s1 = l1["w32"], l1["h32"], l1["rw"], l1["rh"], l1["scale"]
            w2, h2, rw2, rh2, s2 = l2["w32"], l2["h32"], l2["rw"], l2["rh"], l2["scale"]
            k1 = max(1, int(round(self.top_k * 0.20)))
            k2 = max(1, int(self.top_k - k1))
            kpts0_1, desc0_1 = extract_dense(img0, w1, h1, k1, rw1, rh1)
            kpts0_2, desc0_2 = extract_dense(img0, w2, h2, k2, rw2, rh2)
            kpts1_1, desc1_1 = extract_dense(img1, w1, h1, k1, rw1, rh1)
            kpts1_2, desc1_2 = extract_dense(img1, w2, h2, k2, rw2, rh2)

            sc0 = np.concatenate(
                [
                    np.full((len(kpts0_1),), 1.0 / float(s1), dtype=np.float32),
                    np.full((len(kpts0_2),), 1.0 / float(s2), dtype=np.float32),
                ],
                axis=0,
            )
            kpts0 = np.concatenate([kpts0_1 / float(s1), kpts0_2 / float(s2)], axis=0)
            kpts1 = np.concatenate([kpts1_1 / float(s1), kpts1_2 / float(s2)], axis=0)
            desc0 = np.concatenate([desc0_1, desc0_2], axis=0)
            desc1 = np.concatenate([desc1_1, desc1_2], axis=0)
        else:
            base_w, base_h = self.size
            l0 = (
                self.star_levels[0]
                if getattr(self, "star_levels", None)
                else {"w32": base_w, "h32": base_h, "rw": 1.0, "rh": 1.0, "scale": 1.0}
            )
            w, h, rw, rh, s = l0["w32"], l0["h32"], l0["rw"], l0["rh"], l0["scale"]
            kpts0, desc0 = extract_dense(img0, int(w), int(h), self.top_k, float(rw), float(rh))
            kpts1, desc1 = extract_dense(img1, int(w), int(h), self.top_k, float(rw), float(rh))
            sc0 = np.ones((len(kpts0),), dtype=np.float32)

        idx0, idx1 = _batch_match(desc0, desc1, min_cossim=-1.0)
        if len(idx0) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                kpts0,
                kpts1,
                desc0,
                desc1,
            )

        mkpts0 = kpts0[idx0].copy()
        mkpts1 = kpts1[idx1].copy()

        pairs = np.concatenate([desc0[idx0], desc1[idx1]], axis=1)[None, ...].astype(np.float32)
        pairs_i = pairs.astype(np.float16) if self.dtype == "fp16" else pairs
        (offset_logits,) = self.fm_sess.run(None, {"pairs": pairs_i})
        offset_logits = offset_logits.astype(np.float32)[0]
        offsets, conf = _subpix_softmax2d(offset_logits, temp=3.0)
        mkpts0 = mkpts0 + offsets * sc0[idx0][:, None]

        keep = conf > float(self.fine_conf)
        mkpts0 = mkpts0[keep]
        mkpts1 = mkpts1[keep]

        # Rescale coordinates to original image sizes.
        kpts0 = (kpts0 * np.array([sx0, sy0], dtype=np.float32)).astype(np.float32)
        kpts1 = (kpts1 * np.array([sx1, sy1], dtype=np.float32)).astype(np.float32)
        mkpts0 = (mkpts0 * np.array([sx0, sy0], dtype=np.float32)).astype(np.float32)
        mkpts1 = (mkpts1 * np.array([sx1, sy1], dtype=np.float32)).astype(np.float32)

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1


def _softmax_last(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)


def _subpix_softmax2d(logits_n64: np.ndarray, temp: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    # logits_n64 -> (N, 64) interpreted as 8x8.
    if logits_n64.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    hm = logits_n64.reshape(-1, 8, 8).astype(np.float32)
    p = _softmax_last((hm * float(temp)).reshape(-1, 64)).reshape(-1, 8, 8)
    y, x = np.meshgrid(np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32), indexing="ij")
    x = x - 4.0
    y = y - 4.0
    dx = (p * x[None, ...]).sum(axis=(1, 2))
    dy = (p * y[None, ...]).sum(axis=(1, 2))
    conf = p.reshape(-1, 64).max(axis=1)
    return np.stack([dx, dy], axis=1).astype(np.float32), conf.astype(np.float32)


def _batch_match(desc0: np.ndarray, desc1: np.ndarray, min_cossim: float = -1.0):
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
