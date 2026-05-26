import argparse
from pathlib import Path
import torch, onnx
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationDataReader, quantize_static
from xfeat_onnx_utils import ONNX_DIR, WEIGHTS, XFeatE2EExport, XFeatRawExport

class RandomCalib(CalibrationDataReader):
    def __init__(self, input_name, width, height, n=4):
        import numpy as np
        rng=np.random.default_rng(123); self.data=[{input_name:rng.random((1,3,height,width), dtype=np.float32)} for _ in range(n)]; self.i=0
    def get_next(self):
        if self.i>=len(self.data): return None
        x=self.data[self.i]; self.i+=1; return x

def out_name(kind,dtype,w,h,topk=None):
    if kind=='raw': return ONNX_DIR / f'xfeat_raw_{dtype}_{w}x{h}.onnx'
    return ONNX_DIR / f'xfeat_e2e_{dtype}_{w}x{h}_k{topk}.onnx'

def export(kind,dtype,w,h,topk,opset=18):
    out=out_name(kind,dtype,w,h,topk)
    model=XFeatRawExport(WEIGHTS) if kind=='raw' else XFeatE2EExport(WEIGHTS, topk=topk)
    model.eval(); dummy=torch.randn(1,3,h,w)
    if dtype=='fp16': model=model.half(); dummy=dummy.half()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=model.to(device); dummy=dummy.to(device)
    print('export', out, flush=True)
    outs=['descriptors_map','kpt_logits','reliability'] if kind=='raw' else ['keypoints','scores','descriptors']
    torch.onnx.export(model,dummy,str(out),input_names=['image'],output_names=outs,opset_version=opset,do_constant_folding=True,export_params=True,keep_initializers_as_inputs=False)
    onnx.checker.check_model(str(out)); return out

def quantize(fp32,out,w,h):
    import onnxruntime as ort
    sess=ort.InferenceSession(str(fp32), providers=['CPUExecutionProvider']); reader=RandomCalib(sess.get_inputs()[0].name,w,h)
    print('quant int8', out, flush=True)
    quantize_static(str(fp32), str(out), reader, quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=False)
    onnx.checker.check_model(str(out))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--width',type=int,default=640); ap.add_argument('--height',type=int,default=480); ap.add_argument('--topks',nargs='+',type=int,default=[1024,2048]); ap.add_argument('--formats',nargs='+',default=['fp32','fp16','int8'],choices=['fp32','fp16','int8']); ap.add_argument('--kinds',nargs='+',default=['e2e','raw'],choices=['e2e','raw']); ap.add_argument('--opset',type=int,default=18)
    args=ap.parse_args(); ONNX_DIR.mkdir(parents=True, exist_ok=True)
    for kind in args.kinds:
        topks=args.topks if kind=='e2e' else [None]
        for tk in topks:
            fp32=None
            if 'fp32' in args.formats or 'int8' in args.formats: fp32=export(kind,'fp32',args.width,args.height,tk,args.opset)
            if 'fp16' in args.formats: export(kind,'fp16',args.width,args.height,tk,args.opset)
            if 'int8' in args.formats: quantize(fp32,out_name(kind,'int8',args.width,args.height,tk),args.width,args.height)
    print('done')
if __name__=='__main__': main()
