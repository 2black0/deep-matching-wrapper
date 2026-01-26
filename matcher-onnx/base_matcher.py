import cv2
import numpy as np
from pathlib import Path


class BaseMatcher:
    """Base class for ONNXRuntime-backed matchers.

    Matches the high-level output contract of `matcher/base_matcher.py`.
    Child classes implement `_forward(img0, img1)`.
    """

    def __init__(self, device: str | None = None, **kwargs):
        if device is None:
            device = "cuda"  # best-effort; each matcher falls back to CPU if needed
        self.device = device

        self.skip_ransac: bool = False
        self.ransac_iters: int = int(kwargs.get("ransac_iters", 2000))
        self.ransac_conf: float = float(kwargs.get("ransac_conf", 0.95))
        self.ransac_reproj_thresh: float = float(kwargs.get("ransac_reproj_thresh", 3))

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        """Load image as CHW float32 RGB in [0,1]."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return x

    def compute_ransac(self, matched_kpts0: np.ndarray, matched_kpts1: np.ndarray):
        if matched_kpts0 is None or matched_kpts1 is None:
            return None, np.empty([0, 2]), np.empty([0, 2])
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
        if H is None or inliers_mask is None:
            return None, np.empty([0, 2]), np.empty([0, 2])

        inliers_mask = inliers_mask[:, 0].astype(bool)
        return H, matched_kpts0[inliers_mask], matched_kpts1[inliers_mask]

    def __call__(self, img0, img1):
        # Keep the same UX as the torch version: accept path / numpy.
        if isinstance(img0, (str, Path)):
            img0 = self.load_image(img0)
        if isinstance(img1, (str, Path)):
            img1 = self.load_image(img1)

        # Accept HWC uint8/float arrays too.
        if isinstance(img0, np.ndarray) and img0.ndim == 3 and img0.shape[0] != 3:
            img0 = img0.transpose(2, 0, 1).astype(np.float32) / (255.0 if img0.dtype != np.float32 else 1.0)
        if isinstance(img1, np.ndarray) and img1.ndim == 3 and img1.shape[0] != 3:
            img1 = img1.transpose(2, 0, 1).astype(np.float32) / (255.0 if img1.dtype != np.float32 else 1.0)

        assert isinstance(img0, np.ndarray) and img0.ndim == 3 and img0.shape[0] == 3
        assert isinstance(img1, np.ndarray) and img1.ndim == 3 and img1.shape[0] == 3

        mkpts0, mkpts1, kpts0, kpts1, desc0, desc1 = self._forward(img0, img1)

        mkpts0 = np.empty((0, 2), dtype=np.float32) if mkpts0 is None else mkpts0
        mkpts1 = np.empty((0, 2), dtype=np.float32) if mkpts1 is None else mkpts1
        kpts0 = np.empty((0, 2), dtype=np.float32) if kpts0 is None else kpts0
        kpts1 = np.empty((0, 2), dtype=np.float32) if kpts1 is None else kpts1
        desc0 = np.empty((0, 1), dtype=np.float32) if desc0 is None else desc0
        desc1 = np.empty((0, 1), dtype=np.float32) if desc1 is None else desc1

        # Basic sanity.
        if mkpts0.size > 0:
            assert mkpts0.ndim == 2 and mkpts0.shape[1] == 2
            assert mkpts1.ndim == 2 and mkpts1.shape[1] == 2
            assert mkpts0.shape == mkpts1.shape
        if kpts0.size > 0:
            assert kpts0.ndim == 2 and kpts0.shape[1] == 2
        if kpts1.size > 0:
            assert kpts1.ndim == 2 and kpts1.shape[1] == 2

        H, inlier_kpts0, inlier_kpts1 = self.compute_ransac(mkpts0, mkpts1)

        return {
            "num_inliers": int(len(inlier_kpts0)),
            "H": H,
            "all_kpts0": kpts0,
            "all_kpts1": kpts1,
            "all_desc0": desc0,
            "all_desc1": desc1,
            "matched_kpts0": mkpts0,
            "matched_kpts1": mkpts1,
            "inlier_kpts0": inlier_kpts0,
            "inlier_kpts1": inlier_kpts1,
        }


def _discover_available_matchers() -> list[str]:
    """Discover available ONNX matchers from `matcher-onnx/weights/`."""

    root = Path(__file__).resolve().parent
    weights_root = root / "weights"
    out: list[str] = []

    # XFeat
    xfeat_dir = weights_root / "xfeat"
    if xfeat_dir.exists():
        if any(xfeat_dir.glob("xfeat_backbone_*.onnx")):
            out.append("xfeat")
        # Star uses a separate unnormalized backbone (preferred) + fine matcher.
        has_backbone_star = any(xfeat_dir.glob("xfeat_backbone_star_*.onnx"))
        has_finematcher = any(xfeat_dir.glob("xfeat_finematcher_*.onnx"))
        if has_finematcher and (has_backbone_star or any(xfeat_dir.glob("xfeat_backbone_*.onnx"))):
            out.append("xfeat-star")
        if any(xfeat_dir.glob("xfeat_lighterglue_*.onnx")):
            out.append("xfeat-lightglue")

    # LiftFeat
    liftfeat_dir = weights_root / "liftfeat"
    if liftfeat_dir.exists() and any(liftfeat_dir.glob("liftfeat_*.onnx")):
        out.append("liftfeat")

    # SuperPoint + LightGlue
    lg_dir = weights_root / "lightglue"
    if lg_dir.exists() and any(lg_dir.glob("superpoint_backbone_*.onnx")) and any(
        lg_dir.glob("superpoint_lightglue_*.onnx")
    ):
        out.append("superpoint-lightglue")

    # CLIDD variants
    clidd_dir = weights_root / "clidd"
    if clidd_dir.exists():
        cfgs: set[str] = set()
        for p in clidd_dir.glob("clidd_*_fp*_*.onnx"):
            name = p.name
            # clidd_u128_fp32_640x480.onnx -> u128
            mid = name[len("clidd_") :]
            cfg = mid.split("_fp", 1)[0]
            if cfg:
                cfgs.add(cfg.lower())
        for cfg in sorted(cfgs):
            out.append(f"clidd-{cfg}")

    # SubPX (Keypt2Subpx) refiners
    subpx_dir = weights_root / "subpx"
    if subpx_dir.exists():
        has_k2s_xfeat = any(subpx_dir.glob("k2s_xfeat_refiner_*.onnx"))
        has_k2s_splg = any(subpx_dir.glob("k2s_splg_refiner_*.onnx"))

        if has_k2s_xfeat and "xfeat" in out:
            out.append("xfeat-subpx")
        if has_k2s_xfeat and "xfeat-lightglue" in out:
            out.append("xfeat-lightglue-subpx")
        if has_k2s_splg and "superpoint-lightglue" in out:
            out.append("superpoint-lightglue-subpx")

    # Stable ordering
    primary = [
        "xfeat",
        "xfeat-star",
        "xfeat-lightglue",
        "xfeat-subpx",
        "xfeat-lightglue-subpx",
        "superpoint-lightglue",
        "superpoint-lightglue-subpx",
        "liftfeat",
    ]
    ordered = [m for m in primary if m in out] + [m for m in out if m.startswith("clidd-")]
    return ordered


AVAILABLE_MATCHERS = _discover_available_matchers()


def get_matcher(name: str, device: str | None = None, **kwargs):
    name = name.lower()
    if name == "xfeat":
        from xfeat import XFeatONNXMatcher

        return XFeatONNXMatcher(device=device, mode="xfeat", **kwargs)
    if name == "xfeat-star":
        from xfeat import XFeatONNXMatcher

        return XFeatONNXMatcher(device=device, mode="star", **kwargs)
    if name == "xfeat-lightglue":
        from xfeat import XFeatONNXMatcher

        return XFeatONNXMatcher(device=device, mode="lightglue", **kwargs)
    if name == "xfeat-subpx":
        from subpx import SubPXONNXMatcher

        return SubPXONNXMatcher(device=device, mode="xfeat-subpx", **kwargs)
    if name == "xfeat-lightglue-subpx":
        from subpx import SubPXONNXMatcher

        return SubPXONNXMatcher(device=device, mode="xfeat-lightglue-subpx", **kwargs)
    if name == "liftfeat":
        from liftfeat import LiftFeatONNXMatcher

        return LiftFeatONNXMatcher(device=device, **kwargs)
    if name == "superpoint-lightglue":
        from superpoint_lightglue import SuperPointLightGlueONNXMatcher

        return SuperPointLightGlueONNXMatcher(device=device, **kwargs)
    if name == "superpoint-lightglue-subpx":
        from subpx import SubPXONNXMatcher

        return SubPXONNXMatcher(device=device, mode="superpoint-lightglue-subpx", **kwargs)
    if name.startswith("clidd-"):
        from clidd import CLIDDONNXMatcher

        return CLIDDONNXMatcher(device=device, model_name=name, **kwargs)

    raise ValueError(f"Unknown ONNX matcher: '{name}'. Available: {', '.join(AVAILABLE_MATCHERS)}")
