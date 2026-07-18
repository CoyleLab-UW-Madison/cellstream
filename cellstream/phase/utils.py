"""
cellstream.phase.utils

Phase analysis utilities: winding number (topological charge) computation
and unified phase-field feature generation (defects, velocity, FTLE).
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

def winding_number(phase_img, n=5, mode="replicate", row_blocks=1, device='cpu'):
    """
    Compute winding number (topological charge) in local windows.
    """
    assert n % 2 == 1, "Window size n must be odd"
    pad = n // 2

    original_shape = phase_img.shape
    if phase_img.ndim < 2:
        raise ValueError("Phase image must have at least 2 spatial dimensions (H, W)")

    H, W = original_shape[-2:]
    
    if phase_img.ndim == 2:
        img_4d = phase_img.unsqueeze(0).unsqueeze(0)
    elif phase_img.ndim == 3:
        img_4d = phase_img.unsqueeze(0)
    else:
        C = original_shape[-3]
        N = torch.tensor(original_shape[:-3]).prod().item() if len(original_shape) > 3 else 1
        img_4d = phase_img.reshape(N, C, H, W)

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
    
    if row_blocks == 'auto':
        from ..utils import get_auto_batch_size
        row_blocks = get_auto_batch_size(
            (N * W,), # Batch size represents number of rows (each row has N*W pixels)
            dtype=img_4d.dtype, 
            device=device,
            bytes_per_element_multiplier=C * n * 20 # 20n footprint due to intermediate diff tensors
        )
        row_blocks = max(1, row_blocks)

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
    return winding.reshape(original_shape)


def generate_phase_features(
    phase: torch.Tensor,
    mask: torch.Tensor = None,
    device: str = None,
    ftle_integration_time: int = 20,
    smooth_sigma: float = 1.0,
    defect_window_size: int = 5,
):
    """
    Generate all relevant features from a phase field (defects, velocity, FTLE).
    
    This is the phase-module analogue of ``fft.generate_fft_features`` — it takes
    a raw phase tensor and returns a dictionary of derived feature arrays.  The
    heavier Zarr I/O orchestration lives in ``phase.process``.
    
    Args:
        phase: (T, Y, X) tensor of phase values.
        mask: Optional (T, Y, X) or (Y, X) mask.
        device: 'cuda', 'cpu', or None (auto-detect).
        ftle_integration_time: Number of frames for FTLE integration.
        smooth_sigma: Gaussian smoothing sigma for phase velocity.
        defect_window_size: Kernel size for winding number computation (must be odd).
        
    Returns:
        dict with keys:
            - 'winding_number': (T, Y, X) float32 array
            - 'velocity':       (T, 2, Y, X) float32 array
            - 'speed':          (T, Y, X) float32 array
            - 'ftle_forward':   (T, Y, X) float32 array
            - 'ftle_backward':  (T, Y, X) float32 array
    """
    from .analytic import phase_velocity, compute_ftle

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    phase = phase.to(device)
    if mask is not None:
        mask = mask.to(device)
        
    features = {}
    
    # 1. Defects (Winding Number Field)
    wn = winding_number(phase, n=defect_window_size, device=device)
    features['winding_number'] = wn.cpu().numpy()
    
    # 2. Phase Velocity
    v, speed, _ = phase_velocity(phase, smooth_sigma=smooth_sigma, device=device)
    features['velocity'] = v.cpu().numpy()
    features['speed'] = speed.cpu().numpy()
    
    # 3. FTLE (Forward and Backward)
    ftle_fwd = compute_ftle(
        v, 
        integration_time=ftle_integration_time, 
        device=device, 
        mask=mask, 
        backward=False
    )
    features['ftle_forward'] = ftle_fwd.cpu().numpy()
    
    ftle_bwd = compute_ftle(
        v, 
        integration_time=ftle_integration_time, 
        device=device, 
        mask=mask, 
        backward=True
    )
    features['ftle_backward'] = ftle_bwd.cpu().numpy()

    # Free GPU memory if we used it
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return features
