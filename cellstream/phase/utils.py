"""
cellstream.phase.utils

Phase analysis utilities: winding number (topological charge) computation
and unified phase-field feature generation (defects, velocity, FTLE).
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

def winding_number(phase_img, n=5, mode="replicate", row_blocks=1, device='cpu', disable_tqdm=False):
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
    for row_start in tqdm(range(0, H, row_blocks), desc="Winding number", leave=False, disable=disable_tqdm):
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


# All requestable phase features
PHASE_FEATURES = [
    'winding_number',
    'velocity',
    'speed',
    'wavenumber',
    'ftle_forward',
    'ftle_backward',
    'streamlines',
    'phase_streamlines',
]


def generate_phase_features(
    phase: torch.Tensor,
    mask: torch.Tensor = None,
    device: str = None,
    ftle_integration_time: int = 20,
    smooth_sigma: float = 1.0,
    defect_window_size: int = 5,
    phase_features_to_process: list = None,
    stream_particles: int = 20000,
    stream_decay: float = 0.85,
    stream_inject_rate: float = 0.05,
    use_mask: bool = True,
    **kwargs
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
        phase_features_to_process: List of feature keys to compute.  Valid keys
            are defined in ``PHASE_FEATURES``.  Defaults to
            ``['velocity', 'speed', 'ftle_forward', 'ftle_backward', 'winding_number']``.
        stream_particles: Number of tracer particles for streamline generation.
        stream_decay: Exponential decay factor for streamline tails.
        stream_inject_rate: Fraction of particles to inject per frame.
        use_mask: Whether to apply mask filtering to phase features. Default True.
        
    Returns:
        dict  — keys are the computed feature names, values are numpy arrays.
        Also contains ``'_attrs'`` with processing parameters for reproducibility.

        Output shapes (all float32 numpy):
            - 'winding_number': (T, Y, X)
            - 'velocity':       (T, 2, Y, X)
            - 'speed':          (T, Y, X)
            - 'wavenumber':     (T, Y, X)
            - 'ftle_forward':   (T, Y, X)
            - 'ftle_backward':  (T, Y, X)
            - 'streamlines':    (T, Y, X)
            - 'phase_streamlines': (T, 3, Y, X)
    """
    from .analytic import phase_velocity, compute_ftle, generate_streamlines, generate_phase_colored_streamlines

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if phase_features_to_process is None:
        phase_features_to_process = ['velocity', 'speed', 'ftle_forward', 'ftle_backward', 'winding_number']

    if not use_mask:
        mask = None

    phase = phase.to(device)
    if mask is not None:
        mask = mask.to(device)
        
    rich_progress = kwargs.get('rich_progress', None)
    rich_sub_task = kwargs.get('rich_sub_task', None)
    rich_cell_name = kwargs.get('rich_cell_name', 'cell')
    disable_tqdm = rich_progress is not None
    
    own_task = False
    task = None
    if rich_progress:
        if rich_sub_task is not None:
            task = rich_sub_task
        else:
            task = rich_progress.add_task(f"[cyan]Processing Phase for {rich_cell_name}...", total=None)
            own_task = True
        
    features = {}
    
    # 1. Defects (Winding Number Field)
    if 'winding_number' in phase_features_to_process:
        if rich_progress and task:
            rich_progress.update(task, description=f"[cyan]Computing Winding Numbers for {rich_cell_name}...")
        wn = winding_number(phase, n=defect_window_size, device=device, disable_tqdm=disable_tqdm)
        features['winding_number'] = wn.cpu().numpy()
    
    # Needs velocity?
    needs_vel = any(k in phase_features_to_process for k in [
        'velocity', 'speed', 'wavenumber', 'ftle_forward', 'ftle_backward', 
        'streamlines', 'phase_streamlines'
    ])
    
    if needs_vel:
        if rich_progress and task:
            rich_progress.update(task, description=f"[cyan]Computing Velocity for {rich_cell_name}...")
        v, speed, wavenumber = phase_velocity(phase, smooth_sigma=smooth_sigma, device=device, disable_tqdm=disable_tqdm)
            
        if 'velocity' in phase_features_to_process:
            features['velocity'] = v.cpu().numpy()
        if 'speed' in phase_features_to_process:
            features['speed'] = speed.cpu().numpy()
        if 'wavenumber' in phase_features_to_process:
            features['wavenumber'] = wavenumber.cpu().numpy()
            
        if 'ftle_forward' in phase_features_to_process:
            if rich_progress and task:
                rich_progress.update(task, description=f"[cyan]Computing Forward FTLE for {rich_cell_name}...")
            ftle_fwd = compute_ftle(v, integration_time=ftle_integration_time, device=device, mask=mask, backward=False, progress_bar=False if disable_tqdm else None)
            features['ftle_forward'] = ftle_fwd.cpu().numpy()
            
        if 'ftle_backward' in phase_features_to_process:
            if rich_progress and task:
                rich_progress.update(task, description=f"[cyan]Computing Backward FTLE for {rich_cell_name}...")
            ftle_bwd = compute_ftle(v, integration_time=ftle_integration_time, device=device, mask=mask, backward=True, progress_bar=False if disable_tqdm else None)
            features['ftle_backward'] = ftle_bwd.cpu().numpy()
            
        if 'streamlines' in phase_features_to_process:
            if rich_progress and task:
                rich_progress.update(task, description=f"[cyan]Tracing Streamlines for {rich_cell_name}...")
            streams = generate_streamlines(
                v, num_particles=stream_particles, decay=stream_decay, 
                device=device, mask=mask, inject_rate=stream_inject_rate
            )
            features['streamlines'] = streams.cpu().numpy()
            
        if 'phase_streamlines' in phase_features_to_process:
            if rich_progress and task:
                rich_progress.update(task, description=f"[cyan]Tracing Phase Streamlines for {rich_cell_name}...")
            p_streams = generate_phase_colored_streamlines(
                v, phase, num_particles=stream_particles, decay=stream_decay, 
                device=device, mask=mask, inject_rate=stream_inject_rate
            )
            features['phase_streamlines'] = p_streams.cpu().numpy()

    if rich_progress and own_task and task:
        rich_progress.remove_task(task)

    # Store processing parameters for reproducibility (P3)
    features['_attrs'] = {
        'smooth_sigma': smooth_sigma,
        'ftle_integration_time': ftle_integration_time,
        'defect_window_size': defect_window_size,
        'phase_features_to_process': list(phase_features_to_process),
        'stream_particles': stream_particles,
        'stream_decay': stream_decay,
        'stream_inject_rate': stream_inject_rate,
        'device': device,
    }

    # Free GPU memory if we used it
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return features
