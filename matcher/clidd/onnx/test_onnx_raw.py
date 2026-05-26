import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from clidd_onnx_utils import make_input, weights_for, compare_outputs, load_pth, run_pth, create_ort_session
from matcher.clidd.modules.model import Model
from matcher.clidd.modules.clidd_wrapper import CLIDD


def nms_topk_sample(raw_desc_np, raw_detect_np, cfg, topk=1024, radius=2, score_thresh=-5.0, border=4, device='cuda'):
    model = Model(**CLIDD.cfgs[cfg]).eval().to(device)
    model.load_state_dict(torch.load(str(weights_for(cfg)), map_location='cpu'))
    raw_desc = [torch.from_numpy(x).to(device) for x in raw_desc_np]
    raw_detect = torch.from_numpy(raw_detect_np).to(device)
    B, _, H, W = raw_detect.shape
    mp = nn.MaxPool2d(radius * 2 + 1, 1, radius).to(device)
    is_max = raw_detect == mp(raw_detect)
    if border > 0:
        mask = torch.ones_like(is_max, dtype=torch.bool)
        mask[..., :, :border] = False; mask[..., :, -border:] = False
        mask[..., :border, :] = False; mask[..., -border:, :] = False
        is_max = is_max & mask
    valid = is_max & (raw_detect > score_thresh)
    refined = torch.where(valid, raw_detect, torch.full_like(raw_detect, -1e8))
    scores, idx = torch.topk(refined.view(B, -1), k=topk, dim=1)
    y = (idx // W).float(); x = (idx % W).float()
    kpts = torch.stack([x, y], dim=-1)
    size = torch.tensor([W, H], dtype=torch.float32, device=device)
    norm = ((kpts + 0.5) / size * 2 - 1).unsqueeze(2)
    with torch.inference_mode():
        desc = model.sample(raw_desc, norm)
        if device == 'cuda': torch.cuda.synchronize()
    return [kpts.cpu().numpy().astype(np.float32), scores.cpu().numpy().astype(np.float32), desc.cpu().numpy().astype(np.float32)]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--cfg', default='A48')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--topk', type=int, default=1024)
    ap.add_argument('--loops', type=int, default=10)
    args=ap.parse_args()
    image=make_input(args.width,args.height)
    pth,dev=load_pth(args.cfg,args.width,args.height,args.topk)
    ref=run_pth(pth,image,dev)
    sess=create_ort_session(Path(args.model), cuda=True)
    inp=sess.get_inputs()[0]
    raw=sess.run(None,{inp.name:image})
    got=nms_topk_sample(raw[:3], raw[3], args.cfg, args.topk, device=str(dev))
    times=[]
    for _ in range(args.loops):
        t0=time.perf_counter(); raw=sess.run(None,{inp.name:image}); got=nms_topk_sample(raw[:3], raw[3], args.cfg, args.topk, device=str(dev)); times.append((time.perf_counter()-t0)*1000)
    print(json.dumps({'model':args.model,'providers':sess.get_providers(),'loop_ms_mean':sum(times)/len(times),'loop_ms_min':min(times),'loop_ms_max':max(times),'compare_to_pth':compare_outputs(ref,got)}, indent=2))

if __name__=='__main__': main()
