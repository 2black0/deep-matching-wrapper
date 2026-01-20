
import numpy as np
import torch
import torchvision.transforms as tfm
from pathlib import Path
from PIL import Image

def to_numpy(x):
    """convert item or container of items to numpy"""
    if isinstance(x, list):
        return np.array([to_numpy(i) for i in x])
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_numpy(v)
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if x is None:
        return
    raise NotImplementedError(f"to_numpy not implemented for data type {type(x)}")

def to_normalized_coords(pts, height, width):
    """normalize kpt coords from px space to [0,1]"""
    # assume pts are in x,y order
    assert pts.shape[-1] == 2, f"input to `to_normalized_coords` should be shape (N, 2), input is shape {pts.shape}"
    pts = to_numpy(pts).astype(float)
    pts[:, 0] /= width
    pts[:, 1] /= height
    return pts

def to_px_coords(pts, height, width):
    """unnormalized kpt coords from [0,1] to px space"""
    assert pts.shape[-1] == 2, f"input to `to_px_coords` should be shape (N, 2), input is shape {pts.shape}"
    pts = to_numpy(pts)
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts

def resize_to_divisible(img, divisible_by=14):
    """Resize to be divisible by a factor."""
    h, w = img.shape[-2:]
    divisible_h = round(h / divisible_by) * divisible_by
    divisible_w = round(w / divisible_by) * divisible_by
    img = tfm.functional.resize(img, [divisible_h, divisible_w], antialias=True)
    return img

def to_tensor(x, device=None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if device is not None:
        x = x.to(device)
    return x
