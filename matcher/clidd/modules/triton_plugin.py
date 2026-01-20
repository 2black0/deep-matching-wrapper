
import torch
import torch.nn.functional as F

def deformable_sample_project(input, grid, weight, bias, *, is_input_nhwc: bool = False, align_corners: bool = False):
    """
    Pure PyTorch implementation of the deformable sample & project operation.
    Replaces the Triton kernel for compatibility and CPU support.
    """
    
    # Input handling
    # input: (B, H, W, C) if nhwc else (B, C, H, W)
    if is_input_nhwc:
        input = input.permute(0, 3, 1, 2) # to (B, C, H, W)
    
    # grid: (B, N, M, 2)
    # weight: (C_out, C_in, 1, M)
    # bias: (C_out) or None
    
    B, C_in, H, W = input.shape
    B_grid, N, M, _ = grid.shape
    
    # Verify shapes match expected logic
    C_out, C_in_w, _, M_w = weight.shape
    assert C_in == C_in_w, f"Input channels {C_in} != Weight input channels {C_in_w}"
    assert M == M_w, f"Grid heads {M} != Weight heads {M_w}"
    
    # Grid Sample
    # F.grid_sample samples from input at locations specified by grid.
    # grid values should be in range [-1, 1]. The logic in the original Triton kernel
    # confirms that it converts from [-1, 1] to pixel coords, so grid_sample is appropriate.
    # grid shape (B, N, M, 2) treats N dim as Height_out and M dim as Width_out
    
    # samples: (B, C_in, N, M)
    samples = F.grid_sample(input, grid, align_corners=align_corners, mode='bilinear', padding_mode='zeros')
    
    # Projection
    # We want to compute: Out[b, n, co] = sum_m sum_ci ( samples[b, ci, n, m] * weight[co, ci, 0, m] )
    
    # Prepare weights: (C_out, C_in, M)
    w = weight.squeeze(2) 
    
    # Einsum for the batched projection
    # b: batch
    # c: input channels (C_in)
    # n: num points
    # m: num heads
    # o: output channels (C_out)
    
    # samples: bcnm
    # w: ocm
    # result: bno
    
    out = torch.einsum('bcnm,ocm->bno', samples, w)
    
    if bias is not None:
        out = out + bias.view(1, 1, -1)
        
    return out