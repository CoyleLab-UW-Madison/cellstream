import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def phase_velocity(phase, epsilon=1e-4, smooth_sigma=None, device='cpu', row_blocks='auto', disable_tqdm=False):
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

    for row_start in tqdm(range(0, H, row_blocks), desc="Analytic phase velocity", leave=False, disable=disable_tqdm):
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
        wavenumber[:, :, row_start:row_end, :] = torch.sqrt(dphi_dx**2 + dphi_dy**2)
        
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
    mask: torch.Tensor = None,
    inject_rate: float = 0.05
):
    """
    Generate a dense (T, Y, X) image stack of glowing particle streamlines
    traced through the velocity field using Euler integration with continuous
    particle injection.
    
    Args:
        velocity: (T, 2, Y, X) tensor of phase velocity
        num_particles: Target number of tracer particles
        decay: Persistence of comet tails [0.0 - 1.0]
        velocity_multiplier: Scale particle speeds
        device: 'cpu' or 'cuda'
        mask: Optional (T, Y, X) or (Y, X) mask tensor
        inject_rate: Fraction of num_particles to inject per frame
        
    Returns:
        images: (T, Y, X) image tensor of glowing particle traces
    """
    velocity = velocity.to(device)
    T, _, H, W = velocity.shape
    max_particles = num_particles * 2
    inject_per_frame = max(1, int(num_particles * inject_rate))
    
    if mask is not None:
        mask = mask.to(device)
        spatial_mask = mask.max(dim=0)[0] if mask.ndim == 3 else mask
        valid_y, valid_x = torch.where(spatial_mask > 0)
        num_valid = len(valid_y)
        if num_valid == 0:
            mask = None
            
    def spawn_particles(n):
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
            
    # Seed initial population
    particles = spawn_particles(num_particles)
    
    images = torch.zeros((T, H, W), device=device, dtype=torch.float32)
    canvas = torch.zeros((H, W), device=device, dtype=torch.float32)
    
    vel_scale = torch.tensor([2.0/(W-1), 2.0/(H-1)], device=device).view(1, 2)
    vel_scale = vel_scale * velocity_multiplier
    
    for t in range(T):
        canvas = canvas * decay
        
        n = particles.shape[0]
        if n == 0:
            # All particles died, re-seed
            particles = spawn_particles(num_particles)
            n = num_particles
        
        v_t = velocity[t:t+1]
        grid_coords = particles.view(1, 1, n, 2)
        
        v_sampled = F.grid_sample(v_t, grid_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        v_sampled = v_sampled.view(2, n).t()
        
        # Euler step
        particles = particles + v_sampled * vel_scale
        
        # Check bounds at updated position
        alive = (particles[:, 0] >= -1) & (particles[:, 0] <= 1) & \
                (particles[:, 1] >= -1) & (particles[:, 1] <= 1)
                
        if mask is not None:
            grid_coords_updated = particles.view(1, 1, n, 2)
            mask_t = mask[t:t+1] if mask.ndim == 3 else mask.unsqueeze(0)
            mask_t = mask_t.unsqueeze(0).float()
            m_sampled = F.grid_sample(mask_t, grid_coords_updated, mode='nearest', padding_mode='zeros', align_corners=True)
            alive = alive & (m_sampled.view(n) >= 0.5)
        
        # Kill dead particles (don't respawn)
        particles = particles[alive]
        
        # Rasterize surviving particles
        n_alive = particles.shape[0]
        if n_alive > 0:
            px = ((particles[:, 0] + 1) / 2 * (W - 1)).round().long()
            py = ((particles[:, 1] + 1) / 2 * (H - 1)).round().long()
            px = torch.clamp(px, 0, W - 1)
            py = torch.clamp(py, 0, H - 1)
            canvas.index_put_((py, px), torch.tensor(1.0, device=device), accumulate=True)
            canvas = torch.clamp(canvas, 0, 1.0)
        
        # Inject fresh particles
        new_particles = spawn_particles(inject_per_frame)
        particles = torch.cat([particles, new_particles], dim=0)
        
        # Cap total to prevent unbounded growth
        if particles.shape[0] > max_particles:
            particles = particles[:max_particles]
        
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
                grid_coords = particles.view(1, 1, num_particles, 2)
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

def compute_ftle(
    velocity: torch.Tensor,
    integration_time: int = 20,
    delta: float = 1.0,
    device: str = 'cpu',
    mask: torch.Tensor = None,
    backward: bool = False
):
    """
    Compute the Finite-Time Lyapunov Exponent field from a velocity tensor.
    
    Args:
        velocity: (T, 2, Y, X) tensor of velocity
        integration_time: Number of frames to integrate forward (or backward)
        delta: Perturbation size in pixels for gradient computation
        device: 'cpu' or 'cuda'
        mask: Optional (Y, X) or (T, Y, X) mask
        backward: If True, computes backward FTLE (attracting structures). If False, forward FTLE (repelling).
        
    Returns:
        ftle: (T, Y, X) tensor of FTLE values, zero-padded for frames where integration isn't possible
    """
    velocity = velocity.to(device)
    T, _, H, W = velocity.shape
    
    if integration_time >= T:
        raise ValueError(f"integration_time ({integration_time}) must be less than T ({T})")
    
    ftle = torch.zeros((T, H, W), device=device, dtype=torch.float32)
    
    # Create base grid of all pixel coordinates in [-1, 1] range
    gy = torch.linspace(-1, 1, H, device=device)
    gx = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # (H, W)
    
    # Perturbation in normalized coordinates  
    dx_norm = delta * 2.0 / (W - 1)
    dy_norm = delta * 2.0 / (H - 1)
    
    vel_scale = torch.tensor([2.0/(W-1), 2.0/(H-1)], device=device)
    if backward:
        vel_scale = -vel_scale
    
    valid_frames = range(integration_time, T) if backward else range(T - integration_time)
    
    for t0 in valid_frames:
        # 4 perturbed grids: +x, -x, +y, -y
        # Each is (H, W, 2) with last dim = (x, y) for grid_sample
        px_pos = torch.stack([grid_x + dx_norm, grid_y], dim=-1)  # +x
        px_neg = torch.stack([grid_x - dx_norm, grid_y], dim=-1)  # -x
        py_pos = torch.stack([grid_x, grid_y + dy_norm], dim=-1)  # +y
        py_neg = torch.stack([grid_x, grid_y - dy_norm], dim=-1)  # -y
        
        # Advect all 4 grids forward for integration_time steps
        grids = torch.stack([px_pos, px_neg, py_pos, py_neg], dim=0)  # (4, H, W, 2)
        
        for dt in range(integration_time):
            t = t0 - dt if backward else t0 + dt
            v_t = velocity[t:t+1]  # (1, 2, H, W)
            
            # Sample velocity at all 4 grid positions
            # grid_sample expects (N, H, W, 2)
            v_sampled = F.grid_sample(v_t.expand(4, -1, -1, -1), grids, 
                                       mode='bilinear', padding_mode='border', align_corners=True)
            # v_sampled: (4, 2, H, W)
            
            # Update positions: grids[..., 0] += vx * scale_x, grids[..., 1] += vy * scale_y
            grids[..., 0] = grids[..., 0] + v_sampled[:, 0] * vel_scale[0]
            grids[..., 1] = grids[..., 1] + v_sampled[:, 1] * vel_scale[1]
            
            # Clamp to valid range
            grids = torch.clamp(grids, -1, 1)
        
        # Compute deformation gradient from final positions
        # dx/dx0 = (x_plus - x_minus) / (2 * delta)
        # But we need to convert back from normalized to pixel coords
        # F_xx = d(final_x)/d(initial_x)
        F_xx = (grids[0, :, :, 0] - grids[1, :, :, 0]) / (2 * dx_norm)
        F_xy = (grids[2, :, :, 0] - grids[3, :, :, 0]) / (2 * dy_norm)
        F_yx = (grids[0, :, :, 1] - grids[1, :, :, 1]) / (2 * dx_norm)
        F_yy = (grids[2, :, :, 1] - grids[3, :, :, 1]) / (2 * dy_norm)
        
        # Cauchy-Green tensor C = F^T F
        C_xx = F_xx**2 + F_yx**2
        C_xy = F_xx * F_xy + F_yx * F_yy
        C_yy = F_xy**2 + F_yy**2
        
        # Max eigenvalue of 2x2 symmetric matrix:
        # lambda_max = (trace + sqrt(trace^2 - 4*det)) / 2
        trace = C_xx + C_yy
        det = C_xx * C_yy - C_xy**2
        discriminant = torch.clamp(trace**2 - 4 * det, min=0)
        lambda_max = (trace + torch.sqrt(discriminant)) / 2
        lambda_max = torch.clamp(lambda_max, min=1.0)  # eigenvalue >= 1 (no contraction below identity)
        
        # FTLE = (1/T) * ln(sqrt(lambda_max)) = (1/(2T)) * ln(lambda_max)
        ftle[t0] = torch.log(lambda_max) / (2.0 * integration_time)
    
    # Apply mask if provided
    if mask is not None:
        mask = mask.to(device)
        if mask.ndim == 3:
            mask_ftle = mask
        else:
            mask_ftle = mask.unsqueeze(0).expand(T, -1, -1)
        ftle = ftle * (mask_ftle > 0).float()
    
    return ftle

def generate_phase_colored_streamlines(
    velocity: torch.Tensor,
    phase: torch.Tensor,
    num_particles: int = 20000,
    decay: float = 0.85,
    velocity_multiplier: float = 1.0,
    device: str = 'cpu',
    mask: torch.Tensor = None,
    inject_rate: float = 0.05
):
    """
    Generate an RGB (T, 3, Y, X) image stack of particle streamlines where
    each particle is colored by the local phase value it is riding on.
    
    Args:
        velocity: (T, 2, Y, X) tensor of phase velocity
        phase: (T, Y, X) tensor of phase values in radians [-pi, pi]
        num_particles: Target number of tracer particles
        decay: Persistence of comet tails [0.0 - 1.0]
        velocity_multiplier: Scale particle speeds  
        device: 'cpu' or 'cuda'
        mask: Optional mask tensor
        inject_rate: Fraction of num_particles to inject per frame
        
    Returns:
        images: (T, 3, Y, X) float32 tensor, RGB channels in [0, 1]
    """
    velocity = velocity.to(device)
    phase = phase.to(device)
    T, _, H, W = velocity.shape
    max_particles = num_particles * 2
    inject_per_frame = max(1, int(num_particles * inject_rate))
    
    if mask is not None:
        mask = mask.to(device)
        spatial_mask = mask.max(dim=0)[0] if mask.ndim == 3 else mask
        valid_y, valid_x = torch.where(spatial_mask > 0)
        num_valid = len(valid_y)
        if num_valid == 0:
            mask = None
            
    def spawn_particles(n):
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
    
    def hsv_to_rgb_fast(h, s, v):
        """Vectorized HSV to RGB. h in [0,1], s in [0,1], v in [0,1]. Returns (3,) per element."""
        h6 = h * 6.0
        i = h6.long() % 6
        f = h6 - h6.floor()
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v)))))
        g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p)))))
        b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))
        
        return r, g, b
    
    particles = spawn_particles(num_particles)
    
    images = torch.zeros((T, 3, H, W), device=device, dtype=torch.float32)
    canvas_r = torch.zeros((H, W), device=device, dtype=torch.float32)
    canvas_g = torch.zeros((H, W), device=device, dtype=torch.float32)
    canvas_b = torch.zeros((H, W), device=device, dtype=torch.float32)
    
    vel_scale = torch.tensor([2.0/(W-1), 2.0/(H-1)], device=device).view(1, 2)
    vel_scale = vel_scale * velocity_multiplier
    
    for t in range(T):
        canvas_r = canvas_r * decay
        canvas_g = canvas_g * decay
        canvas_b = canvas_b * decay
        
        n = particles.shape[0]
        if n == 0:
            particles = spawn_particles(num_particles)
            n = num_particles
        
        v_t = velocity[t:t+1]
        grid_coords = particles.view(1, 1, n, 2)
        
        v_sampled = F.grid_sample(v_t, grid_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        v_sampled = v_sampled.view(2, n).t()
        
        particles = particles + v_sampled * vel_scale
        
        alive = (particles[:, 0] >= -1) & (particles[:, 0] <= 1) & \
                (particles[:, 1] >= -1) & (particles[:, 1] <= 1)
                
        if mask is not None:
            grid_coords_updated = particles.view(1, 1, n, 2)
            mask_t = mask[t:t+1] if mask.ndim == 3 else mask.unsqueeze(0)
            mask_t = mask_t.unsqueeze(0).float()
            m_sampled = F.grid_sample(mask_t, grid_coords_updated, mode='nearest', padding_mode='zeros', align_corners=True)
            alive = alive & (m_sampled.view(n) >= 0.5)
        
        particles = particles[alive]
        
        n_alive = particles.shape[0]
        if n_alive > 0:
            # Sample the phase at each particle's position
            phase_t = phase[t:t+1].unsqueeze(0)  # (1, 1, H, W)
            gc = particles.view(1, 1, n_alive, 2)
            phase_sampled = F.grid_sample(phase_t, gc, mode='bilinear', padding_mode='border', align_corners=True)
            phase_vals = phase_sampled.view(n_alive)
            
            # Map phase [-pi, pi] to hue [0, 1]
            hue = (phase_vals + torch.pi) / (2 * torch.pi)
            hue = torch.clamp(hue, 0, 1)
            
            sat = torch.ones_like(hue)
            val = torch.ones_like(hue)
            r, g, b = hsv_to_rgb_fast(hue, sat, val)
            
            px = ((particles[:, 0] + 1) / 2 * (W - 1)).round().long()
            py = ((particles[:, 1] + 1) / 2 * (H - 1)).round().long()
            px = torch.clamp(px, 0, W - 1)
            py = torch.clamp(py, 0, H - 1)
            
            # Paint each channel with the particle's color
            canvas_r.index_put_((py, px), r, accumulate=True)
            canvas_g.index_put_((py, px), g, accumulate=True)
            canvas_b.index_put_((py, px), b, accumulate=True)
        
        new_particles = spawn_particles(inject_per_frame)
        particles = torch.cat([particles, new_particles], dim=0)
        
        if particles.shape[0] > max_particles:
            particles = particles[:max_particles]
        
        images[t, 0] = torch.clamp(canvas_r, 0, 1)
        images[t, 1] = torch.clamp(canvas_g, 0, 1)
        images[t, 2] = torch.clamp(canvas_b, 0, 1)
    
    return images
