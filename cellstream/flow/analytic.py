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
    wavenumber = torch.empty((N_batch, T, H, W), dtype=phase_4d.dtype, device=device)
    
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
        
        # Divide by R^2 to get the true phase gradient of the smoothed complex field
        # R^2 = |Z|^2 = Z * Z_conj
        R2 = (z_center * z_conj).real + 1e-8
        
        dphi_dx = (z_conj * dz_dx).imag / R2
        dphi_dy = (z_conj * dz_dy).imag / R2
        dphi_dt = (z_conj * dz_dt).imag / R2
        
        grad_sq = dphi_dx**2 + dphi_dy**2 + epsilon
        
        vx = -dphi_dt * dphi_dx / grad_sq
        vy = -dphi_dt * dphi_dy / grad_sq
        
        velocity[:, :, 0, row_start:row_end, :] = vx
        velocity[:, :, 1, row_start:row_end, :] = vy
        speed[:, :, row_start:row_end, :] = torch.sqrt(vx**2 + vy**2)
        wavenumber[:, :, row_start:row_end, :] = torch.sqrt(grad_sq)
        
    velocity = velocity.reshape(original_shape[:-3] + (T, 2, H, W))
    speed = speed.reshape(original_shape[:-3] + (T, H, W))
    wavenumber = wavenumber.reshape(original_shape[:-3] + (T, H, W))
    
    return velocity, speed, wavenumber

def generate_streamlines(
    velocity: torch.Tensor, 
    num_particles: int = 20000, 
    decay: float = 0.85, 
    velocity_multiplier: float = 1.0,
    device: str = 'cpu',
    mask: torch.Tensor = None
):
    """
    Generate a dense (T, Y, X) image stack of glowing particle streamlines
    traced through the velocity field using Euler integration.
    
    Args:
        velocity: (T, 2, Y, X) tensor of phase velocity [t, vx, vy]
        num_particles: Number of virtual tracer particles to simulate
        decay: Persistence of the comet tails [0.0 - 1.0]
        velocity_multiplier: Scale particle speeds
        device: 'cpu' or 'cuda'
        
    Returns:
        images: (T, Y, X) image tensor of glowing particle traces
    """
    velocity = velocity.to(device)
    T, _, H, W = velocity.shape
    
    if mask is not None:
        mask = mask.to(device)
        # Collapse time dimension if present to find all possible valid spatial locations
        spatial_mask = mask.max(dim=0)[0] if mask.ndim == 3 else mask
        valid_y, valid_x = torch.where(spatial_mask > 0)
        num_valid = len(valid_y)
        if num_valid == 0:
            mask = None
            
    def respawn_particles(n):
        if mask is not None:
            idx = torch.randint(0, num_valid, (n,), device=device)
            py = valid_y[idx].float()
            px = valid_x[idx].float()
            # Convert to [-1, 1] range
            px = px / (W - 1) * 2 - 1
            py = py / (H - 1) * 2 - 1
            # Add sub-pixel jitter
            px += (torch.rand(n, device=device) - 0.5) * 2 / W
            py += (torch.rand(n, device=device) - 0.5) * 2 / H
            return torch.stack([px, py], dim=1)
        else:
            return (torch.rand((n, 2), device=device, dtype=torch.float32) * 2 - 1)
            
    # Initialize random particle coordinates
    particles = respawn_particles(num_particles)
    
    images = torch.zeros((T, H, W), device=device, dtype=torch.float32)
    canvas = torch.zeros((H, W), device=device, dtype=torch.float32)
    
    # Pre-compute pixel-to-grid coordinate scaling
    vel_scale = torch.tensor([2.0/(W-1), 2.0/(H-1)], device=device).view(1, 2)
    vel_scale = vel_scale * velocity_multiplier
    
    for t in range(T):
        # Decay previous comet tails
        canvas = canvas * decay
        
        v_t = velocity[t:t+1] # (1, 2, H, W)
        
        # grid_sample expects (N, H_out, W_out, 2) where the last dim is (x, y)
        grid_coords = particles.view(1, 1, num_particles, 2)
        
        # Interpolate the velocity vector exactly at each particle's current floating-point position
        v_sampled = F.grid_sample(v_t, grid_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        v_sampled = v_sampled.view(2, num_particles).t() # (num_particles, 2)
        
        # Move particles (Euler step)
        particles = particles + v_sampled * vel_scale
        
        # Check out of bounds
        out_of_bounds = (particles[:, 0] < -1) | (particles[:, 0] > 1) | \
                        (particles[:, 1] < -1) | (particles[:, 1] > 1)
                        
        if mask is not None:
            mask_t = mask[t:t+1] if mask.ndim == 3 else mask.unsqueeze(0)
            mask_t = mask_t.unsqueeze(0).float()
            m_sampled = F.grid_sample(mask_t, grid_coords, mode='nearest', padding_mode='zeros', align_corners=True)
            out_of_bounds = out_of_bounds | (m_sampled.view(num_particles) < 0.5)
                        
        if out_of_bounds.any():
            n_out = out_of_bounds.sum().item()
            particles[out_of_bounds] = respawn_particles(n_out)
            
        # Rasterize particles onto the pixel grid
        px = ((particles[:, 0] + 1) / 2 * (W - 1)).round().long()
        py = ((particles[:, 1] + 1) / 2 * (H - 1)).round().long()
        
        px = torch.clamp(px, 0, W - 1)
        py = torch.clamp(py, 0, H - 1)
        
        # Draw onto canvas (accumulate brightness)
        canvas.index_put_((py, px), torch.tensor(1.0, device=device), accumulate=True)
        canvas = torch.clamp(canvas, 0, 1.0)
        
        images[t] = canvas
    return images

def generate_instantaneous_streamlines(
    velocity: torch.Tensor, 
    num_particles: int = 20000, 
    steps: int = 50, 
    velocity_multiplier: float = 1.0,
    device: str = 'cpu',
    mask: torch.Tensor = None
):
    """
    Generate a dense (T, Y, X) image stack of instantaneous streamlines.
    For each frame, time is frozen and particles are integrated along the static field.
    
    Args:
        velocity: (T, 2, Y, X) tensor
        num_particles: Number of tracer particles per frame
        steps: How many integration steps to trace the line
    """
    velocity = velocity.to(device)
    T, _, H, W = velocity.shape
    
    if mask is not None:
        mask = mask.to(device)
        spatial_mask = mask.max(dim=0)[0] if mask.ndim == 3 else mask
        valid_y, valid_x = torch.where(spatial_mask > 0)
        num_valid = len(valid_y)
        if num_valid == 0:
            mask = None
            
    def get_particles(n):
        if mask is not None:
            idx = torch.randint(0, num_valid, (n,), device=device)
            py = valid_y[idx].float()
            px = valid_x[idx].float()
            px = px / (W - 1) * 2 - 1
            py = py / (H - 1) * 2 - 1
            px += (torch.rand(n, device=device) - 0.5) * 2 / W
            py += (torch.rand(n, device=device) - 0.5) * 2 / H
            return torch.stack([px, py], dim=1)
        else:
            return (torch.rand((n, 2), device=device, dtype=torch.float32) * 2 - 1)
            
    images = torch.zeros((T, H, W), device=device, dtype=torch.float32)
    vel_scale = torch.tensor([2.0/(W-1), 2.0/(H-1)], device=device).view(1, 2) * velocity_multiplier
    
    for t in range(T):
        canvas = torch.zeros((H, W), device=device, dtype=torch.float32)
        particles = get_particles(num_particles)
        v_t = velocity[t:t+1] # Frozen time field (1, 2, H, W)
        
        mask_t = None
        if mask is not None:
            mask_t = mask[t:t+1] if mask.ndim == 3 else mask.unsqueeze(0)
            mask_t = mask_t.unsqueeze(0).float()
            
        for step in range(steps):
            grid_coords = particles.view(1, 1, num_particles, 2)
            v_sampled = F.grid_sample(v_t, grid_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
            v_sampled = v_sampled.view(2, num_particles).t()
            
            particles = particles + v_sampled * vel_scale
            
            out_of_bounds = (particles[:, 0] < -1) | (particles[:, 0] > 1) | \
                            (particles[:, 1] < -1) | (particles[:, 1] > 1)
                            
            if mask_t is not None:
                m_sampled = F.grid_sample(mask_t, grid_coords, mode='nearest', padding_mode='zeros', align_corners=True)
                out_of_bounds = out_of_bounds | (m_sampled.view(num_particles) < 0.5)
                
            # For static streamlines, we don't necessarily respawn, we just stop drawing them.
            # But to keep density even, we can respawn out-of-bounds particles!
            if out_of_bounds.any():
                n_out = out_of_bounds.sum().item()
                particles[out_of_bounds] = get_particles(n_out)
                
            px = ((particles[:, 0] + 1) / 2 * (W - 1)).round().long()
            py = ((particles[:, 1] + 1) / 2 * (H - 1)).round().long()
            px = torch.clamp(px, 0, W - 1)
            py = torch.clamp(py, 0, H - 1)
            
            # Draw onto canvas
            # Using accumulate=True avoids undefined behavior on GPU with duplicate coordinates
            canvas.index_put_((py, px), torch.tensor(1.0, device=device), accumulate=True)
            
        images[t] = torch.clamp(canvas, 0, 1.0)
        
    return images
