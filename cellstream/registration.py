"""
cellstream.registration

Image registration utilities for aligning time-resolved stacks.
"""

import torch
import torch.nn.functional as F
from .utils import downsample

def register_and_transform_image_timeseries(
    img, 
    reg_channel=0, 
    downsample_factor=0.25,
    return_tmats=False,
    padding_mode='reflection',
    registration_mode='RIGID_BODY',
    reference_mode='previous'
):
    """
    Performs image registration on a multi-channel time-series stack.
    """
    try:
        from pystackreg import StackReg
    except ImportError:
        raise ImportError("pystackreg is required for registration.")

    N, C, H, W = img.shape
    
    # Extract channel for registration and downsample
    for_registration = img[:, reg_channel, :, :]
    for_registration_ds = downsample(for_registration, downsample_factor).cpu().numpy()
    
    # Use StackReg to generate transformation matrices on downsampled image
    sr_mode = getattr(StackReg, registration_mode)
    sr = StackReg(sr_mode)
    tmats = sr.register_stack(for_registration_ds, reference=reference_mode)
    
    # Rescale transformation matrix translations
    upsample_by = 1.0 / downsample_factor
    tmats[:, 0, 2] *= upsample_by  # X-translation
    tmats[:, 1, 2] *= upsample_by  # Y-translation
    
    # Adjust tmats to work on torch affine grid
    tmats_tensor = torch.as_tensor(tmats, dtype=torch.float32, device=img.device)
    theta = torch.zeros((N, 2, 3), dtype=torch.float32, device=img.device)
    
    A = tmats_tensor[:, 0, 0]
    B = tmats_tensor[:, 0, 1]
    C = tmats_tensor[:, 0, 2]
    D = tmats_tensor[:, 1, 0]
    E = tmats_tensor[:, 1, 1]
    G = tmats_tensor[:, 1, 2]
    
    h_ratio = (H - 1) / (W - 1)
    w_ratio = (W - 1) / (H - 1)
    
    theta[:, 0, 0] = A
    theta[:, 0, 1] = B * h_ratio
    theta[:, 0, 2] = A + B * h_ratio - 1 + 2.0 * C / (W - 1)
    
    theta[:, 1, 0] = D * w_ratio
    theta[:, 1, 1] = E
    theta[:, 1, 2] = D * w_ratio + E - 1 + 2.0 * G / (H - 1)
    
    # Create affine grid and sample
    grid = F.affine_grid(theta, img.size(), align_corners=True)

    img_reg = F.grid_sample(img, grid, align_corners=True, padding_mode=padding_mode)
    
    if return_tmats:
        return img_reg, tmats
    return img_reg
