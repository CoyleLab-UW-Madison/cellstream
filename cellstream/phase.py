"""
cellstream.phase

Phase analysis utilities, including winding number computation for topological charge.
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def winding_number(phase_img, n=5, mode="replicate", row_blocks=1, device='cpu'):
    """
    Compute winding number (topological charge) in local windows.
    """
    assert n % 2 == 1, "Window size n must be odd"
    pad = n // 2

    # Standardize to 4D (N, C, H, W)
    if phase_img.ndim == 2:
        img_4d = phase_img.unsqueeze(0).unsqueeze(0)
    elif phase_img.ndim == 3:
        img_4d = phase_img.unsqueeze(0)
    else:
        img_4d = phase_img

    img_4d = img_4d.to(device)
    N, C, H, W = img_4d.shape

    # Preallocate output
    winding = torch.empty((N, C, H, W), dtype=img_4d.dtype, device=device)

    # Pad image
    img_p = F.pad(img_4d, (pad, pad, pad, pad), mode=mode)

    # Perimeter indices for winding number calculation
    idx = []
    idx += [(0, j) for j in range(n)]
    idx += [(i, n-1) for i in range(1, n-1)]
    idx += [(n-1, j) for j in reversed(range(n))]
    idx += [(i, 0) for i in reversed(range(1, n-1))]
    idx = torch.tensor(idx, device=device)

    # Process in row blocks to save memory
    for row_start in tqdm(range(0, H, row_blocks)):
        row_end = min(row_start + row_blocks, H)
        rows = row_end - row_start

        # Slice padded image to get overlapping patches
        patch_slice = img_p[:, :, row_start:row_start + rows + n - 1, :]
        shape = (N, C, rows, W, n, n)
        strides = (
            *patch_slice.stride()[:2], 
            patch_slice.stride(2), 
            patch_slice.stride(3), 
            patch_slice.stride(2), 
            patch_slice.stride(3)
        )
        patches = patch_slice.as_strided(shape, strides)
        
        # Extract perimeter and compute phase differences
        perim = patches[..., idx[:, 0], idx[:, 1]]
        diffs = torch.diff(perim, dim=-1, append=perim[..., :1])
        # Wrap phase differences to [-pi, pi]
        diffs = (diffs + torch.pi) % (2 * torch.pi) - torch.pi

        winding[:, :, row_start:row_end, :] = diffs.sum(dim=-1) / (2 * torch.pi)

    # Restore original shape
    if phase_img.ndim == 2:
        return winding.squeeze(0).squeeze(0)
    elif phase_img.ndim == 3:
        return winding.squeeze(0)
    return winding
