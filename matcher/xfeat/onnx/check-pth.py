import argparse,json,torch
from xfeat_onnx_utils import WEIGHTS, load_pth, make_input, run_pth

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--width',type=int,default=640); ap.add_argument('--height',type=int,default=480); ap.add_argument('--topk',type=int,default=1024)
 args=ap.parse_args(); image=make_input(args.width,args.height); m,d=load_pth(args.topk); y=run_pth(m,image,d); sd=torch.load(str(WEIGHTS), map_location='cpu')
 print(json.dumps({'weights':str(WEIGHTS),'device':str(d),'num_tensors':len(sd),'total_weight_values':int(sum(v.numel() for v in sd.values())),'input_shape':[1,3,args.height,args.width],'topk':args.topk,'outputs':{'keypoints':list(y[0].shape),'scores':list(y[1].shape),'descriptors':list(y[2].shape)}},indent=2))
if __name__=='__main__': main()
