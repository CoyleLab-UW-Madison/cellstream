import torch
import torch.nn.functional as F
import numpy as np

def binned_piv_velocity(
    phase, 
    num_bins=8, 
    window_size=16, 
    overlap=8, 
    morph_kernel=3, 
    device='cpu', 
    upsample=False
):
    """
    Compute wave velocity field by binning phase into iso-contour masks,
    cleaning with morphological ops, and running FFT-based cross-correlation
    (PIV) between consecutive frames.
    
    Args:
        phase: (T, X, Y) tensor of phase values in radians
        num_bins: number of phase bins across [-pi, pi]
        window_size: PIV interrogation window size in pixels
        overlap: overlap between adjacent windows
        morph_kernel: erosion/dilation kernel size for cleaning binary masks
        device: 'cpu' or 'cuda'
        upsample: If True, uses bilinear interpolation to upsample the velocity field back to (X, Y)
        
    Returns:
        if upsample=False:
            velocity: (T-1, 2, grid_y, grid_x) tensor of displacement vectors
            grid_coords: tuple of (y_centers, x_centers) arrays for the PIV grid
        if upsample=True:
            velocity: (T-1, 2, X, Y) tensor
    """
    phase = phase.to(device)
    
    if phase.ndim < 3:
        raise ValueError("Phase must be at least 3D (T, X, Y)")
        
    original_shape = phase.shape
    T, H, W = original_shape[-3:]
    
    if phase.ndim == 3:
        phase_4d = phase.unsqueeze(0)
    else:
        N = torch.tensor(original_shape[:-3]).prod().item()
        phase_4d = phase.reshape(N, T, H, W)
        
    N_batch = phase_4d.shape[0]
    
    # 1. Bin the phase field
    bin_edges = torch.linspace(-np.pi, np.pi, num_bins + 1, device=device)
    bin_idx = torch.bucketize(phase_4d, bin_edges) - 1
    bin_idx = torch.clamp(bin_idx, 0, num_bins - 1)
    
    masks = torch.zeros((num_bins, N_batch, T, H, W), dtype=torch.float32, device=device)
    for b in range(num_bins):
        masks[b] = (bin_idx == b).float()
        
    # 2. Morphological cleanup (open: erode then dilate)
    if morph_kernel > 1:
        pad = morph_kernel // 2
        masks = masks.view(num_bins * N_batch * T, 1, H, W)
        # Erode (min pool) = 1 - max_pool(1 - x)
        eroded = 1.0 - F.max_pool2d(1.0 - masks, kernel_size=morph_kernel, stride=1, padding=pad)
        # Dilate (max pool)
        masks = F.max_pool2d(eroded, kernel_size=morph_kernel, stride=1, padding=pad)
        masks = masks.view(num_bins, N_batch, T, H, W)
        
    # 3. FFT-based PIV
    step = window_size - overlap
    y_starts = torch.arange(0, H - window_size + 1, step)
    x_starts = torch.arange(0, W - window_size + 1, step)
    grid_y = len(y_starts)
    grid_x = len(x_starts)
    
    # Extract windows: (num_bins, N, T, grid_y, grid_x, window_size, window_size)
    windows = torch.empty((num_bins, N_batch, T, grid_y, grid_x, window_size, window_size), dtype=torch.float32, device=device)
    for i, ys in enumerate(y_starts):
        for j, xs in enumerate(x_starts):
            windows[:, :, :, i, j, :, :] = masks[:, :, :, ys:ys+window_size, xs:xs+window_size]
            
    # Reshape for batched FFT: (num_bins * N * (T-1) * grid_y * grid_x, window_size, window_size)
    W_t = windows[:, :, :-1].reshape(-1, window_size, window_size)
    W_t_plus_1 = windows[:, :, 1:].reshape(-1, window_size, window_size)
    
    # FFT cross-correlation
    fft_t = torch.fft.fft2(W_t)
    fft_t1 = torch.fft.fft2(W_t_plus_1)
    cross_corr_fft = fft_t * fft_t1.conj()
    cross_corr = torch.fft.ifft2(cross_corr_fft).real
    
    # FFT shift to put zero displacement at the center
    cross_corr = torch.fft.fftshift(cross_corr, dim=(-2, -1))
    
    # Find peaks
    batch_size = cross_corr.shape[0]
    cross_corr_flat = cross_corr.view(batch_size, -1)
    max_idx = torch.argmax(cross_corr_flat, dim=1)
    
    peak_y = max_idx // window_size
    peak_x = max_idx % window_size
    
    # Calculate displacements relative to center
    center = window_size // 2
    dy = peak_y - center
    dx = peak_x - center
    
    # Reshape back
    dy = dy.view(num_bins, N_batch, T-1, grid_y, grid_x).float()
    dx = dx.view(num_bins, N_batch, T-1, grid_y, grid_x).float()
    
    # 4. Combine bin-wise velocity fields (weighted by foreground pixels)
    weights = W_t.sum(dim=(-2, -1)).view(num_bins, N_batch, T-1, grid_y, grid_x)
    weights_sum = weights.sum(dim=0) + 1e-6
    
    vx_combined = (dx * weights).sum(dim=0) / weights_sum
    vy_combined = (dy * weights).sum(dim=0) / weights_sum
    
    velocity = torch.stack([vx_combined, vy_combined], dim=2) # (N, T-1, 2, grid_y, grid_x)
    
    if upsample:
        # Interpolate back to original size
        # F.interpolate needs (N, C, H, W) where C=2 in our case
        velocity_flat = velocity.view(N_batch * (T-1), 2, grid_y, grid_x)
        velocity_up = F.interpolate(velocity_flat, size=(H, W), mode='bilinear', align_corners=False)
        velocity = velocity_up.view(original_shape[:-3] + (T-1, 2, H, W))
        return velocity
    else:
        velocity = velocity.view(original_shape[:-3] + (T-1, 2, grid_y, grid_x))
        y_centers = (y_starts + center).numpy()
        x_centers = (x_starts + center).numpy()
        return velocity, (y_centers, x_centers)
