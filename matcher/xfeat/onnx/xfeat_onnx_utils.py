import sys, time, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from matcher.xfeat.modules.model import XFeatModel
from matcher.xfeat.modules.interpolator import InterpolateSparse2d

ONNX_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = ONNX_DIR.parent / 'weights'
WEIGHTS = WEIGHTS_DIR / 'xfeat.pt'

class XFeatRawExport(nn.Module):
    def __init__(self, weights=WEIGHTS, normalize=True):
        super().__init__(); self.net=XFeatModel(); self.net.load_state_dict(torch.load(str(weights), map_location='cpu')); self.net.eval(); self.normalize=normalize
    def forward(self, image):
        desc,kpt,rel=self.net(image)
        if self.normalize: desc=F.normalize(desc, dim=1)
        return desc.float(), kpt.float(), rel.float()

class XFeatE2EExport(nn.Module):
    def __init__(self, weights=WEIGHTS, topk=1024, threshold=0.05):
        super().__init__(); self.raw=XFeatRawExport(weights, normalize=True); self.topk=int(topk); self.threshold=float(threshold); self.mp=nn.MaxPool2d(5,1,2); self.nearest=InterpolateSparse2d('nearest'); self.bilinear=InterpolateSparse2d('bilinear'); self.bicubic=InterpolateSparse2d('bicubic')
    def heatmap(self,kpt):
        s=F.softmax(kpt,1)[:,:64]; B,C,H,W=s.shape
        return s.permute(0,2,3,1).reshape(B,H,W,8,8).permute(0,1,3,2,4).reshape(B,1,H*8,W*8)
    def forward(self, image):
        B,C,H,W=image.shape
        desc,kpt,rel=self.raw(image)
        kh=self.heatmap(kpt)
        local=kh==self.mp(kh)
        valid=local & (kh>self.threshold)
        refined=torch.where(valid, kh, torch.full_like(kh, -1.0))
        scores,idx=torch.topk(refined.view(B,-1), k=self.topk, dim=1)
        y=(idx//W).float(); x=(idx%W).float(); kpts=torch.stack([x,y], dim=-1)
        rel_scores=self.bilinear(rel,kpts,H,W).squeeze(-1)
        scores=scores*rel_scores
        order=torch.argsort(-scores, dim=1)
        kpts=torch.gather(kpts,1,order.unsqueeze(-1).expand(-1,-1,2))
        scores=torch.gather(scores,1,order)
        feats=self.bicubic(desc,kpts,H,W)
        feats=F.normalize(feats,dim=-1)
        return kpts.float(), scores.float(), feats.float()

def make_input(width=640,height=480,seed=0,dtype=np.float32):
    rng=np.random.default_rng(seed); return rng.random((1,3,height,width), dtype=np.float32).astype(dtype)

def load_pth(topk=1024, device=None):
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    m=XFeatE2EExport(WEIGHTS, topk=topk).eval().to(device); return m,device

def run_pth(model,image_np,device):
    x=torch.from_numpy(image_np).to(device)
    with torch.inference_mode():
        y=model(x)
        if device.type=='cuda': torch.cuda.synchronize()
    return [t.detach().float().cpu().numpy() for t in y]

def create_ort_session(path,cuda=True):
    import onnxruntime as ort
    avail=ort.get_available_providers(); providers=[]
    if cuda and 'CUDAExecutionProvider' in avail: providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    return ort.InferenceSession(str(path), providers=providers)

def run_onnx(path,image_np,cuda=True):
    sess=create_ort_session(path,cuda=cuda); inp=sess.get_inputs()[0]; x=image_np.astype(np.float16) if inp.type=='tensor(float16)' else image_np.astype(np.float32); out=sess.run(None,{inp.name:x}); return sess,[o.astype(np.float32) for o in out]

def raw_postprocess(raw, topk=1024, threshold=0.05, device='cuda'):
    desc=torch.from_numpy(raw[0]).to(device); kpt=torch.from_numpy(raw[1]).to(device); rel=torch.from_numpy(raw[2]).to(device)
    B,_,h8,w8=desc.shape; H=h8*8; W=w8*8
    s=F.softmax(kpt,1)[:,:64]
    kh=s.permute(0,2,3,1).reshape(B,h8,w8,8,8).permute(0,1,3,2,4).reshape(B,1,H,W)
    mp=nn.MaxPool2d(5,1,2).to(device); valid=(kh==mp(kh)) & (kh>threshold)
    refined=torch.where(valid, kh, torch.full_like(kh,-1.0))
    scores,idx=torch.topk(refined.view(B,-1), k=topk, dim=1)
    y=(idx//W).float(); x=(idx%W).float(); kpts=torch.stack([x,y],-1)
    bil=InterpolateSparse2d('bilinear').to(device); bic=InterpolateSparse2d('bicubic').to(device)
    scores=scores*bil(rel,kpts,H,W).squeeze(-1)
    order=torch.argsort(-scores,dim=1); kpts=torch.gather(kpts,1,order.unsqueeze(-1).expand(-1,-1,2)); scores=torch.gather(scores,1,order)
    feats=F.normalize(bic(desc,kpts,H,W),dim=-1)
    if device=='cuda': torch.cuda.synchronize()
    return [kpts.cpu().numpy().astype(np.float32), scores.cpu().numpy().astype(np.float32), feats.cpu().numpy().astype(np.float32)]

def compare_outputs(ref,got):
    rows=[]
    for n,a,b in zip(['keypoints','scores','descriptors'],ref,got):
        d=np.abs(a-b); row={'name':n,'shape_ref':list(a.shape),'shape_got':list(b.shape),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'rmse':float(np.sqrt(np.mean((a-b)**2)))}
        if n=='descriptors':
            aa=a.reshape(-1,a.shape[-1]); bb=b.reshape(-1,b.shape[-1]); row['cosine_mean']=float(np.mean(np.sum(aa*bb,1)/(np.linalg.norm(aa,axis=1)*np.linalg.norm(bb,axis=1)+1e-12)))
        rows.append(row)
    return rows
