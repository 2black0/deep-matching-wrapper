import argparse,json,statistics as stats,time
from pathlib import Path
import numpy as np
from xfeat_onnx_utils import ONNX_DIR, load_pth, make_input, run_pth, create_ort_session, raw_postprocess, compare_outputs

def summary(t): return {'mean_ms':float(stats.mean(t)),'median_ms':float(stats.median(t)),'min_ms':float(min(t)),'max_ms':float(max(t)),'p95_ms':float(np.percentile(t,95)),'std_ms':float(stats.pstdev(t))}

def bench_pth(image,topk,loops):
 t0=time.perf_counter(); m,d=load_pth(topk); load=(time.perf_counter()-t0)*1000; run_pth(m,image,d); times=[]; last=None
 for _ in range(loops): t0=time.perf_counter(); last=run_pth(m,image,d); times.append((time.perf_counter()-t0)*1000)
 return {'name':f'xfeat_pth_k{topk}','kind':'pth','device':str(d),'load_ms':load,'latency':summary(times),'outputs':last}

def bench_e2e(path,image,loops):
 t0=time.perf_counter(); sess=create_ort_session(path,True); load=(time.perf_counter()-t0)*1000; inp=sess.get_inputs()[0]; x=image.astype(np.float16) if inp.type=='tensor(float16)' else image.astype(np.float32); sess.run(None,{inp.name:x}); times=[]; last=None
 for _ in range(loops): t0=time.perf_counter(); last=sess.run(None,{inp.name:x}); times.append((time.perf_counter()-t0)*1000)
 return {'name':path.stem,'kind':'onnx_e2e','providers':sess.get_providers(),'load_ms':load,'latency':summary(times),'outputs':[o.astype(np.float32) for o in last]}

def bench_raw(path,image,topk,loops,device='cuda'):
 t0=time.perf_counter(); sess=create_ort_session(path,True); load=(time.perf_counter()-t0)*1000; inp=sess.get_inputs()[0]; x=image.astype(np.float16) if inp.type=='tensor(float16)' else image.astype(np.float32); raw=sess.run(None,{inp.name:x}); raw_postprocess([r.astype(np.float32) for r in raw],topk,device=device); times=[]; last=None
 for _ in range(loops): t0=time.perf_counter(); raw=sess.run(None,{inp.name:x}); last=raw_postprocess([r.astype(np.float32) for r in raw],topk,device=device); times.append((time.perf_counter()-t0)*1000)
 return {'name':path.stem+f'_post_k{topk}','kind':'onnx_raw_plus_post','providers':sess.get_providers(),'load_ms':load,'latency':summary(times),'outputs':last}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--width',type=int,default=640); ap.add_argument('--height',type=int,default=480); ap.add_argument('--topks',nargs='+',type=int,default=[1024,2048]); ap.add_argument('--loops',type=int,default=50)
 args=ap.parse_args(); image=make_input(args.width,args.height); results=[]
 for tk in args.topks:
  p=bench_pth(image,tk,args.loops); ref=p.pop('outputs'); results.append(p)
  for m in sorted(ONNX_DIR.glob(f'xfeat_e2e_*_{args.width}x{args.height}_k{tk}.onnx')):
   try: r=bench_e2e(m,image,args.loops); out=r.pop('outputs'); r['accuracy_vs_pth']=compare_outputs(ref,out); results.append(r)
   except Exception as e: results.append({'name':m.stem,'kind':'onnx_e2e','error':repr(e)})
  for m in sorted(ONNX_DIR.glob(f'xfeat_raw_*_{args.width}x{args.height}.onnx')):
   try: r=bench_raw(m,image,tk,args.loops,device='cuda'); out=r.pop('outputs'); r['accuracy_vs_pth']=compare_outputs(ref,out); results.append(r)
   except Exception as e: results.append({'name':m.stem+f'_post_k{tk}','kind':'onnx_raw_plus_post','error':repr(e)})
 print(json.dumps({'input_shape':[1,3,args.height,args.width],'topks':args.topks,'loops':args.loops,'results':results},indent=2))
if __name__=='__main__': main()
