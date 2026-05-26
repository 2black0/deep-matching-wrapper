import argparse,json,time
from clidd_onnx_utils import load_pth, make_input, run_pth, compare_outputs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cfg',default='A48'); ap.add_argument('--width',type=int,default=640); ap.add_argument('--height',type=int,default=480); ap.add_argument('--topk',type=int,default=1024); ap.add_argument('--loops',type=int,default=10)
    args=ap.parse_args(); image=make_input(args.width,args.height); model,device=load_pth(args.cfg,args.width,args.height,args.topk)
    t0=time.perf_counter(); ref=run_pth(model,image,device); first=(time.perf_counter()-t0)*1000
    times=[]; got=ref
    for _ in range(args.loops):
        t0=time.perf_counter(); got=run_pth(model,image,device); times.append((time.perf_counter()-t0)*1000)
    print(json.dumps({'cfg':args.cfg,'model':'pth','device':str(device),'first_ms':first,'loop_ms_mean':sum(times)/len(times),'loop_ms_min':min(times),'loop_ms_max':max(times),'self_compare':compare_outputs(ref,got)},indent=2))
if __name__=='__main__': main()
