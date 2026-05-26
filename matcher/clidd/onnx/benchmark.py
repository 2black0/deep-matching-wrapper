import argparse,json,statistics as stats,time
from pathlib import Path
import numpy as np
from clidd_onnx_utils import CFG_NAMES, ONNX_DIR, weights_for, load_pth, make_input, run_pth, create_ort_session, compare_outputs
from test_onnx_raw import nms_topk_sample

def summary(t): return {'mean_ms':float(stats.mean(t)),'median_ms':float(stats.median(t)),'min_ms':float(min(t)),'max_ms':float(max(t)),'p95_ms':float(np.percentile(t,95)),'std_ms':float(stats.pstdev(t))}

def bench_pth(cfg,image,topk,loops):
    t0=time.perf_counter(); m,d=load_pth(cfg,topk=topk); load=(time.perf_counter()-t0)*1000; run_pth(m,image,d); times=[]; last=None
    for _ in range(loops):
        t0=time.perf_counter(); last=run_pth(m,image,d); times.append((time.perf_counter()-t0)*1000)
    return {'name':f'{cfg}_pth','kind':'pth','path':str(weights_for(cfg)),'device':str(d),'load_ms':load,'latency':summary(times),'outputs':last}

def bench_e2e_onnx(path,image,loops):
    t0=time.perf_counter(); sess=create_ort_session(path,cuda=True); load=(time.perf_counter()-t0)*1000; inp=sess.get_inputs()[0]
    x=image.astype(np.float16) if inp.type=='tensor(float16)' else image.astype(np.float32); sess.run(None,{inp.name:x}); times=[]; last=None
    for _ in range(loops):
        t0=time.perf_counter(); last=sess.run(None,{inp.name:x}); times.append((time.perf_counter()-t0)*1000)
    return {'name':path.stem,'kind':'onnx_e2e','path':str(path),'providers':sess.get_providers(),'load_ms':load,'latency':summary(times),'outputs':[o.astype(np.float32) for o in last]}

def bench_raw_onnx(path,cfg,image,topk,loops):
    import torch
    t0=time.perf_counter(); sess=create_ort_session(path,cuda=True); load=(time.perf_counter()-t0)*1000; inp=sess.get_inputs()[0]
    x=image.astype(np.float32); raw=sess.run(None,{inp.name:x}); nms_topk_sample(raw[:3],raw[3],cfg,topk,device='cuda' if torch.cuda.is_available() else 'cpu')
    times=[]; last=None
    for _ in range(loops):
        t0=time.perf_counter(); raw=sess.run(None,{inp.name:x}); last=nms_topk_sample(raw[:3],raw[3],cfg,topk,device='cuda' if torch.cuda.is_available() else 'cpu'); times.append((time.perf_counter()-t0)*1000)
    return {'name':path.stem,'kind':'onnx_raw_plus_post','path':str(path),'providers':sess.get_providers(),'load_ms':load,'latency':summary(times),'outputs':last}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cfg',default='A48',choices=['all']+CFG_NAMES); ap.add_argument('--width',type=int,default=640); ap.add_argument('--height',type=int,default=480); ap.add_argument('--topk',type=int,default=1024); ap.add_argument('--loops',type=int,default=50); ap.add_argument('--models',nargs='*',default=[]); ap.add_argument('--raw-models',nargs='*',default=[])
    args=ap.parse_args(); image=make_input(args.width,args.height); cfgs=CFG_NAMES if args.cfg=='all' else [args.cfg]; results=[]
    for cfg in cfgs:
        p=bench_pth(cfg,image,args.topk,args.loops); ref=p.pop('outputs'); results.append(p)
        e2es=[Path(m) for m in args.models] if args.models else sorted(ONNX_DIR.glob(f'clidd_{cfg.lower()}_fp32_{args.width}x{args.height}_k{args.topk}.onnx'))
        raws=[Path(m) for m in args.raw_models] if args.raw_models else sorted(ONNX_DIR.glob(f'clidd_{cfg.lower()}_raw_fp32_{args.width}x{args.height}.onnx'))
        for m in e2es:
            if cfg.lower() not in m.stem: continue
            try:
                r=bench_e2e_onnx(m,image,args.loops); out=r.pop('outputs'); r['accuracy_vs_pth']=compare_outputs(ref,out); results.append(r)
            except Exception as e: results.append({'name':m.stem,'kind':'onnx_e2e','path':str(m),'error':repr(e)})
        for m in raws:
            if cfg.lower() not in m.stem: continue
            try:
                r=bench_raw_onnx(m,cfg,image,args.topk,args.loops); out=r.pop('outputs'); r['accuracy_vs_pth']=compare_outputs(ref,out); results.append(r)
            except Exception as e: results.append({'name':m.stem,'kind':'onnx_raw_plus_post','path':str(m),'error':repr(e)})
    print(json.dumps({'input_shape':[1,3,args.height,args.width],'topk':args.topk,'loops':args.loops,'results':results},indent=2))
if __name__=='__main__': main()
