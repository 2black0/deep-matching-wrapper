import argparse
from pathlib import Path
import onnx, torch
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationDataReader, quantize_static
from clidd_onnx_utils import CFG_NAMES, ONNX_DIR, CLIDDExport, weights_for, model_path, p
from matcher.clidd.modules.model import Model
from matcher.clidd.modules.clidd_wrapper import CLIDD

class RandomCalib(CalibrationDataReader):
    def __init__(self, input_name, width, height, n=4):
        import numpy as np
        rng=np.random.default_rng(123)
        self.data=[{input_name:rng.random((1,3,height,width), dtype=np.float32)} for _ in range(n)]
        self.i=0
    def get_next(self):
        if self.i>=len(self.data): return None
        x=self.data[self.i]; self.i+=1; return x

class CLIDDRawExport(torch.nn.Module):
    def __init__(self, cfg, weights_path):
        super().__init__()
        self.model = Model(**CLIDD.cfgs[cfg])
        self.model.load_state_dict(torch.load(str(weights_path), map_location='cpu'))
        self.model.eval()
    def forward(self, x):
        raw_desc, raw_detect = self.model(x)
        return raw_desc[0].float(), raw_desc[1].float(), raw_desc[2].float(), raw_detect.float()

def raw_model_path(cfg, dtype, width, height, out_dir=ONNX_DIR):
    return Path(out_dir) / f"clidd_{cfg.lower()}_raw_{dtype.lower()}_{width}x{height}.onnx"

def export(cfg, dtype, width, height, topk, out_dir, opset, raw=False):
    out = raw_model_path(cfg,dtype,width,height,out_dir) if raw else model_path(cfg,dtype,width,height,out_dir,topk=topk)
    half=dtype.lower()=='fp16'
    model=(CLIDDRawExport(cfg, weights_for(cfg)) if raw else CLIDDExport(cfg, weights_for(cfg), top_k=topk)).eval()
    dummy=torch.randn(1,3,height,width)
    if half:
        model=model.half(); dummy=dummy.half()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device); dummy=dummy.to(device)
    p(f"export {cfg} {'raw ' if raw else ''}{dtype} -> {out}")
    names=['desc1','desc2','desc3','raw_detect'] if raw else ['keypoints','scores','descriptors']
    torch.onnx.export(model,dummy,str(out),input_names=['image'],output_names=names,opset_version=opset,do_constant_folding=True,export_params=True,keep_initializers_as_inputs=False)
    onnx.checker.check_model(str(out))
    return out

def quant_int8(fp32, out, width, height):
    import onnxruntime as ort
    sess=ort.InferenceSession(str(fp32), providers=['CPUExecutionProvider'])
    reader=RandomCalib(sess.get_inputs()[0].name,width,height)
    p(f'quant int8 -> {out}')
    quantize_static(str(fp32), str(out), reader, quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=False)
    onnx.checker.check_model(str(out))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cfg', default='A48', choices=['all']+CFG_NAMES)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--topk', type=int, default=1024)
    ap.add_argument('--formats', nargs='+', default=['fp32','fp16','int8'], choices=['fp32','fp16','int8'])
    ap.add_argument('--raw', action='store_true', help='export dense raw outputs: desc1 desc2 desc3 raw_detect')
    ap.add_argument('--out-dir', default=str(ONNX_DIR))
    ap.add_argument('--opset', type=int, default=18)
    args=ap.parse_args()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfgs=CFG_NAMES if args.cfg=='all' else [args.cfg]
    for cfg in cfgs:
        fp32=None
        if any(f in args.formats for f in ['fp32','int8']):
            fp32=export(cfg,'fp32',args.width,args.height,args.topk,out_dir,args.opset,raw=args.raw)
        if 'fp16' in args.formats:
            export(cfg,'fp16',args.width,args.height,args.topk,out_dir,args.opset,raw=args.raw)
        if 'int8' in args.formats:
            quant_int8(fp32, model_path(cfg,'int8',args.width,args.height,out_dir,topk=args.topk), args.width, args.height)
    p('done')
if __name__=='__main__': main()
