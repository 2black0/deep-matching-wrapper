import argparse, json
import torch
from clidd_onnx_utils import CFG_NAMES, WEIGHTS_DIR, load_pth, make_input, run_pth


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cfg', default='all', choices=['all']+CFG_NAMES)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--topk', type=int, default=1024)
    args=ap.parse_args()
    cfgs = CFG_NAMES if args.cfg == 'all' else [args.cfg]
    image=make_input(args.width,args.height)
    out=[]
    for cfg in cfgs:
        model,device=load_pth(cfg,args.width,args.height,args.topk)
        y=run_pth(model,image,device)
        sd=torch.load(str(WEIGHTS_DIR/f'{cfg}.pth'), map_location='cpu')
        out.append({
            'cfg':cfg,'weights':str(WEIGHTS_DIR/f'{cfg}.pth'),'device':str(device),
            'num_tensors':len(sd),'total_weight_values':int(sum(v.numel() for v in sd.values())),
            'input_shape':[1,3,args.height,args.width],
            'outputs':{
                'keypoints':{'shape':list(y[0].shape),'min':float(y[0].min()),'max':float(y[0].max())},
                'scores':{'shape':list(y[1].shape),'min':float(y[1].min()),'max':float(y[1].max())},
                'descriptors':{'shape':list(y[2].shape),'min':float(y[2].min()),'max':float(y[2].max())},
            }
        })
    print(json.dumps(out, indent=2))

if __name__=='__main__': main()
