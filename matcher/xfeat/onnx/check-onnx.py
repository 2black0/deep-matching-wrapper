import argparse,json
from pathlib import Path
import onnx
from xfeat_onnx_utils import ONNX_DIR, make_input, run_onnx

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('models',nargs='*'); ap.add_argument('--width',type=int,default=640); ap.add_argument('--height',type=int,default=480)
 args=ap.parse_args(); models=[Path(m) for m in args.models] if args.models else sorted(ONNX_DIR.glob('xfeat_*.onnx')); image=make_input(args.width,args.height); out=[]
 for m in models:
  item={'path':str(m)}
  try:
   onnx.checker.check_model(str(m)); item['onnx_ok']=True
   sess,y=run_onnx(m,image,cuda=True); item['providers']=sess.get_providers(); item['inputs']=[{'name':i.name,'shape':i.shape,'type':i.type} for i in sess.get_inputs()]; item['outputs']=[{'name':o.name,'shape':list(v.shape),'dtype':str(v.dtype),'min':float(v.min()),'max':float(v.max())} for o,v in zip(sess.get_outputs(),y)]
  except Exception as e: item['onnx_ok']=False; item['error']=repr(e)
  out.append(item)
 print(json.dumps(out,indent=2))
if __name__=='__main__': main()
