import sys, time, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from matcher.clidd.modules.model import Model
from matcher.clidd.modules.clidd_wrapper import CLIDD

ONNX_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = ONNX_DIR.parent / "weights"
CFG_NAMES = list(CLIDD.cfgs.keys())

class CLIDDExport(nn.Module):
    def __init__(self, cfg_name, weights_path, top_k=1024, radius=2, score_thresh=-5.0, border=4):
        super().__init__()
        self.cfg_name = cfg_name.upper()
        self.model = Model(**CLIDD.cfgs[self.cfg_name])
        self.model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
        self.model.eval()
        self.top_k = int(top_k)
        self.radius = int(radius)
        self.score_thresh = float(score_thresh)
        self.border = int(border)
        self.mp = nn.MaxPool2d(self.radius * 2 + 1, 1, self.radius) if self.radius > 0 else None

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        raw_desc, raw_detect = self.model(x)
        if self.radius > 0:
            is_max = raw_detect == self.mp(raw_detect)
        else:
            is_max = torch.ones_like(raw_detect, dtype=torch.bool)
        if self.border > 0:
            bd = self.border
            mask = torch.ones_like(is_max, dtype=torch.bool)
            mask[..., :, :bd] = False
            mask[..., :, -bd:] = False
            mask[..., :bd, :] = False
            mask[..., -bd:, :] = False
            is_max = is_max & mask
        valid = is_max & (raw_detect > self.score_thresh)
        neg = torch.full_like(raw_detect, -1e8)
        refined = torch.where(valid, raw_detect, neg)
        scores, idx = torch.topk(refined.view(b, -1), k=self.top_k, dim=1)
        y = (idx // w).to(torch.float32)
        xcoord = (idx % w).to(torch.float32)
        kpts = torch.stack([xcoord, y], dim=-1)
        size = torch.tensor([w, h], dtype=torch.float32, device=x.device)
        norm_kpts = ((kpts + 0.5) / size * 2 - 1).unsqueeze(2).to(x.dtype)
        desc = self.model.sample(list(raw_desc), norm_kpts)
        return kpts.float(), scores.float(), desc.float()

def weights_for(cfg):
    return WEIGHTS_DIR / f"{cfg.upper()}.pth"

def model_path(cfg, dtype, width, height, out_dir=ONNX_DIR, topk=None):
    k = "" if topk is None else f"_k{int(topk)}"
    return Path(out_dir) / f"clidd_{cfg.lower()}_{dtype.lower()}_{width}x{height}{k}.onnx"

def make_input(width=640, height=480, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return rng.random((1,3,height,width), dtype=np.float32).astype(dtype)

def load_pth(cfg, width=640, height=480, topk=1024, device=None, half=False):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    m = CLIDDExport(cfg, weights_for(cfg), top_k=topk).eval().to(device)
    if half:
        m = m.half()
    return m, device

def run_pth(model, image_np, device):
    x = torch.from_numpy(image_np).to(device)
    if next(model.parameters()).dtype == torch.float16:
        x = x.half()
    with torch.inference_mode():
        out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return [o.detach().float().cpu().numpy() for o in out]

def create_ort_session(path, cuda=True):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    providers = []
    if cuda and "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(path), providers=providers)

def run_onnx(path, image_np, cuda=True):
    sess = create_ort_session(path, cuda=cuda)
    inp = sess.get_inputs()[0]
    x = image_np.astype(np.float16) if inp.type == "tensor(float16)" else image_np.astype(np.float32)
    out = sess.run(None, {inp.name: x})
    return sess, [o.astype(np.float32) for o in out]

def compare_outputs(ref, got):
    names = ["keypoints", "scores", "descriptors"]
    rows=[]
    for n,a,b in zip(names, ref, got):
        diff=np.abs(a-b)
        row={"name":n,"shape_ref":list(a.shape),"shape_got":list(b.shape),"max_abs":float(diff.max()),"mean_abs":float(diff.mean()),"rmse":float(np.sqrt(np.mean((a-b)**2)))}
        if n == "descriptors":
            aa=a.reshape(-1,a.shape[-1]); bb=b.reshape(-1,b.shape[-1])
            row["cosine_mean"] = float(np.mean(np.sum(aa*bb,axis=1)/(np.linalg.norm(aa,axis=1)*np.linalg.norm(bb,axis=1)+1e-12)))
        rows.append(row)
    return rows

def p(x):
    print(x, flush=True)
