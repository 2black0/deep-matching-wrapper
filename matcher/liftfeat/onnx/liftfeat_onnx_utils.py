import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from matcher.liftfeat.modules.model import LiftFeatSPModel
from matcher.liftfeat.modules.liftfeat_wrapper import featureboost_config

ONNX_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = ONNX_DIR.parent / "weights" / "LiftFeat.pth"


class LiftFeatExport(nn.Module):
    """ONNX-friendly LiftFeat dense export.

    Input: image tensor [1, 3, H, W] or [1, 1, H, W], float in [0, 1].
    Output:
      kpt_logits: [1, 65, H/8, W/8]
      descriptors_map: [1, 64, H/8, W/8], L2-normalized along channel.

    Sparse NMS, keypoint sampling, descriptor sampling stay outside ONNX because
    original wrapper does post-processing with dynamic keypoint count.
    """

    def __init__(self, weights_path=WEIGHTS_PATH):
        super().__init__()
        self.model = LiftFeatSPModel(featureboost_config)
        state_dict = torch.load(str(weights_path), map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _unfold2d_onnx(self, x, ws=8):
        b, c, h, w = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(b, c, h // ws, w // ws, ws * ws)
        return x.permute(0, 1, 4, 2, 3).reshape(b, -1, h // ws, w // ws)

    def forward(self, image):
        des_map, kpt_map, d_feats = self.model.forward1(image)
        normals_feat = self._unfold2d_onnx(d_feats, ws=8)
        b, c, h, w = des_map.shape
        des_v = des_map.permute(0, 2, 3, 1).reshape(-1, c)
        kpts_v = kpt_map.permute(0, 2, 3, 1).reshape(-1, 65)
        norm_v = normals_feat.permute(0, 2, 3, 1).reshape(-1, 192)
        refined = self.model.feature_boost(des_v, kpts_v, norm_v)
        desc_map = refined.view(b, h, w, -1).permute(0, 3, 1, 2)
        desc_map = F.normalize(desc_map, p=2, dim=1)
        return kpt_map.float(), desc_map.float()


def make_input(width=640, height=480, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    x = rng.random((1, 3, height, width), dtype=np.float32)
    return x.astype(dtype)


def load_pth_export(weights_path=WEIGHTS_PATH, device=None, half=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LiftFeatExport(weights_path).eval().to(device)
    if half:
        model = model.half()
    return model, torch.device(device)


def run_pth(model, image_np, device):
    x = torch.from_numpy(image_np).to(device)
    if next(model.parameters()).dtype == torch.float16:
        x = x.half()
    with torch.inference_mode():
        y = model(x)
    return [t.detach().float().cpu().numpy() for t in y]


def run_onnx(path, image_np, providers=None):
    import onnxruntime as ort
    providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in providers if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(path), providers=providers)
    input_name = sess.get_inputs()[0].name
    if sess.get_inputs()[0].type == "tensor(float16)":
        image_np = image_np.astype(np.float16)
    out = sess.run(None, {input_name: image_np})
    return sess, [o.astype(np.float32) for o in out]


def compare_outputs(ref, got):
    rows = []
    for name, a, b in zip(["kpt_logits", "descriptors_map"], ref, got):
        diff = np.abs(a - b)
        rows.append({
            "name": name,
            "shape_ref": list(a.shape),
            "shape_got": list(b.shape),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "cosine_mean": float(np.mean(np.sum(a.reshape(a.shape[0], a.shape[1], -1) * b.reshape(b.shape[0], b.shape[1], -1), axis=1) / (np.linalg.norm(a.reshape(a.shape[0], a.shape[1], -1), axis=1) * np.linalg.norm(b.reshape(b.shape[0], b.shape[1], -1), axis=1) + 1e-12))),
        })
    return rows


def p(msg):
    print(msg, flush=True)
