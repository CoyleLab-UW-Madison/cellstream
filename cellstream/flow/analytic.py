import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def phase_velocity(phase, epsilon=1e-4, smooth_sigma=None, device='cpu', row_blocks='auto'):
    """
    Compute instantaneous wave velocity field from a phase timeseries
    using the analytic level-set velocity formula on the complex signal.
    
    Args:
        phase: (T, X, Y) tensor of phase values in radians
        epsilon: regularization for |grad_phi|^2 denominator near defects
        smooth_sigma: optional Gaussian smoothing kernel size (float) for complex signal before gradient computation
        device: 'cpu' or 'cuda'
        row_blocks: memory blocking along spatial rows
        
    Returns:
        velocity: (T, 2, X, Y) tensor where [t, 0] = v_x, [t, 1] = v_y
        speed: (T, X, Y) tensor of |v| (magnitude)
    """
    if phase.ndim < 3:
        raise ValueError("Phase must be at least 3D (T, X, Y)")
        
    original_shape = phase.shape
    T, H, W = original_shape[-3:]
    
    if phase.ndim == 3:
        phase_4d = phase.unsqueeze(0)
    else:
        N = torch.tensor(original_shape[:-3]).prod().item()
        phase_4d = phase.reshape(N, T, H, W)
        
    phase_4d = phase_4d.to(device)
    N_batch, T, H, W = phase_4d.shape
    
    velocity = torch.empty((N_batch, T, 2, H, W), dtype=phase_4d.dtype, device=device)
    speed = torch.empty((N_batch, T, H, W), dtype=phase_4d.dtype, device=device)
    
    # Gaussian smoothing kernel setup
    kernel = None
    kernel_size = 3
    if smooth_sigma is not None and smooth_sigma > 0:
        kernel_size = int(6 * smooth_sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = torch.arange(kernel_size, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(x, x, indexing='ij')
        center = kernel_size // 2
        kernel = torch.exp(-((grid_x - center)**2 + (grid_y - center)**2) / (2 * smooth_sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(phase_4d.dtype)
        
    total_pad = (kernel_size // 2 + 1) if kernel is not None else 1

    if row_blocks == 'auto':
        from ..utils import get_auto_batch_size
        row_blocks = get_auto_batch_size(
            (N_batch * T * W,),
            dtype=phase_4d.dtype,
            device=device,
            bytes_per_element_multiplier=40
        )
        row_blocks = max(1, row_blocks)

    phase_t_pad = F.pad(phase_4d, (0, 0, 0, 0, 1, 1), mode='replicate')
    z_all = torch.exp(1j * phase_t_pad)

    for row_start in tqdm(range(0, H, row_blocks), desc="Analytic phase velocity", leave=False):
        row_end = min(row_start + row_blocks, H)
        
        y_start = max(0, row_start - total_pad)
        y_end = min(H, row_end + total_pad)
        y_pad_top = total_pad if row_start == 0 else (total_pad - (row_start - y_start))
        y_pad_bot = total_pad if row_end == H else (total_pad - (y_end - row_end))
        
        z_slice = z_all[:, :, y_start:y_end, :]
        z_slice = F.pad(z_slice, (total_pad, total_pad, y_pad_top, y_pad_bot), mode='replicate')
        
        if kernel is not None:
            z_slice_c = z_slice.reshape(N_batch * (T+2), 1, z_slice.shape[-2], z_slice.shape[-1])
            z_real = F.conv2d(z_slice_c.real, kernel, padding=0)
            z_imag = F.conv2d(z_slice_c.imag, kernel, padding=0)
            z_slice = torch.complex(z_real, z_imag).reshape(N_batch, T+2, z_real.shape[-2], z_real.shape[-1])
        
        # Central differences
        dz_dx = (z_slice[:, 1:-1, 1:-1, 2:] - z_slice[:, 1:-1, 1:-1, :-2]) / 2.0
        dz_dy = (z_slice[:, 1:-1, 2:, 1:-1] - z_slice[:, 1:-1, :-2, 1:-1]) / 2.0
        dz_dt = (z_slice[:, 2:, 1:-1, 1:-1] - z_slice[:, :-2, 1:-1, 1:-1]) / 2.0
        
        z_center = z_slice[:, 1:-1, 1:-1, 1:-1]
        z_conj = z_center.conj()
        
        dphi_dx = (z_conj * dz_dx).imag
        dphi_dy = (z_conj * dz_dy).imag
        dphi_dt = (z_conj * dz_dt).imag
        
        grad_sq = dphi_dx**2 + dphi_dy**2 + epsilon
        
        vx = -dphi_dt * dphi_dx / grad_sq
        vy = -dphi_dt * dphi_dy / grad_sq
        
        velocity[:, :, 0, row_start:row_end, :] = vx
        velocity[:, :, 1, row_start:row_end, :] = vy
        speed[:, :, row_start:row_end, :] = torch.sqrt(vx**2 + vy**2)
        
    velocity = velocity.reshape(original_shape[:-3] + (T, 2, H, W))
    speed = speed.reshape(original_shape[:-3] + (T, H, W))
    
    return velocity, speed
