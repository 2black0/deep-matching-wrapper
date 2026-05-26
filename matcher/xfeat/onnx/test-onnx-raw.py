import argparse,json,time,torch
from pathlib import Path
from xfeat_onnx_utils import load_pth, make_input, run_pth, create_ort_session, raw_postprocess, compare_outputs

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('--width',type=int,default=640); ap.add_argument('--height',type=int,default=480); ap.add_argument('--topk',type=int,default=1024); ap.add_argument('--loops',type=int,default=10)
 args=ap.parse_args(); image=make_input(args.width,args.height); pth,d=load_pth(args.topk); ref=run_pth(pth,image,d); sess=create_ort_session(Path(args.model),cuda=True); inp=sess.get_inputs()[0]; x=image.astype('float16') if inp.type=='tensor(float16)' else image
 raw=sess.run(None,{inp.name:x}); got=raw_postprocess([r.astype('float32') for r in raw],args.topk,device=str(d)); times=[]
 for _ in range(args.loops):
  t0=time.perf_counter(); raw=sess.run(None,{inp.name:x}); got=raw_postprocess([r.astype('float32') for r in raw],args.topk,device=str(d)); times.append((time.perf_counter()-t0)*1000)
 print(json.dumps({'model':args.model,'providers':sess.get_providers(),'loop_ms_mean':sum(times)/len(times),'loop_ms_min':min(times),'loop_ms_max':max(times),'compare_to_pth':compare_outputs(ref,got)},indent=2))
if __name__=='__main__': main()
