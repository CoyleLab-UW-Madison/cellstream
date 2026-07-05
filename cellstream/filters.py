"""
cellstream.filters

Time-domain filtering utilities for image timeseries.
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from .utils import normalize_dims

def create_ir_filter(cutoff_freq, high_pass=False, window_size=101):
    """
    Create a sinc impulse response filter natively in PyTorch.
    """
    if not isinstance(cutoff_freq, torch.Tensor):
        cutoff_freq = torch.as_tensor(cutoff_freq, dtype=torch.float32)
    elif not cutoff_freq.is_floating_point():
        cutoff_freq = cutoff_freq.float()
        
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd. Given: {window_size}")

    half = window_size // 2
    device, dtype = cutoff_freq.device, cutoff_freq.dtype
    idx = torch.linspace(-half, half, window_size, device=device, dtype=dtype)

    impulse_response = torch.special.sinc(cutoff_freq.unsqueeze(-1) * idx.unsqueeze(0))
    impulse_response = impulse_response * torch.hamming_window(window_size, device=device, dtype=dtype, periodic=False).unsqueeze(0)
    impulse_response = impulse_response / impulse_response.sum(dim=-1, keepdim=True).abs()
    
    if high_pass:
        delta = torch.zeros_like(impulse_response)
        delta[..., window_size // 2] = 1
        impulse_response = delta - impulse_response
    
    return impulse_response

def create_bandpass_filter(low_cutoff, high_cutoff, window_size=101):
    """
    Create a bandpass impulse response filter natively in PyTorch
    by subtracting two normalized low-pass sinc filters.
    """
    if not isinstance(low_cutoff, torch.Tensor):
        low_cutoff = torch.as_tensor(low_cutoff, dtype=torch.float32)
    elif not low_cutoff.is_floating_point():
        low_cutoff = low_cutoff.float()
        
    if not isinstance(high_cutoff, torch.Tensor):
        high_cutoff = torch.as_tensor(high_cutoff, dtype=torch.float32)
    elif not high_cutoff.is_floating_point():
        high_cutoff = high_cutoff.float()

    if torch.any(low_cutoff >= high_cutoff):
        raise ValueError("low_cutoff must be strictly less than high_cutoff.")

    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd. Given: {window_size}")

    device, dtype = low_cutoff.device, low_cutoff.dtype
    high_cutoff = high_cutoff.to(device=device, dtype=dtype)
    
    half = window_size // 2
    idx = torch.linspace(-half, half, window_size, device=device, dtype=dtype)
    
    # 1. Generate the two underlying low-pass sinc responses
    lp_low = torch.special.sinc(low_cutoff.unsqueeze(-1) * idx.unsqueeze(0))
    lp_high = torch.special.sinc(high_cutoff.unsqueeze(-1) * idx.unsqueeze(0))
    
    # 2. Apply the window function to both to reduce spectral leakage
    window = torch.hamming_window(window_size, device=device, dtype=dtype, periodic=False).unsqueeze(0)
    lp_low = lp_low * window
    lp_high = lp_high * window
    
    # 3. Normalize each low-pass component individually so their DC gains equal 1
    lp_low = lp_low / lp_low.sum(dim=-1, keepdim=True).abs()
    lp_high = lp_high / lp_high.sum(dim=-1, keepdim=True).abs()
    
    # 4. Subtract the lower low-pass from the higher low-pass to get the bandpass
    bandpass_ir = lp_high - lp_low
    
    return bandpass_ir

def apply_fir_filter(image, impulse_response, batch_size='auto'):
    """
    Apply a FIR filter along the time axis of an image tensor (T, C, X, Y).
    """
    from .utils import get_auto_batch_size
    image = normalize_dims(image, 1)
    T, C, X, Y = image.shape
    N = C * X * Y
    
    if batch_size == 'auto':
        batch_size = get_auto_batch_size(
            (T,), 
            dtype=image.dtype, 
            device=image.device,
            bytes_per_element_multiplier=4
        )

    # Reshape to (N, 1, T) for Conv1d
    input_reshaped = image.permute(1, 2, 3, 0).reshape(N, 1, T)

    # Ensure impulse response is (1, 1, K)
    kernel = torch.as_tensor(impulse_response, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = kernel.view(1, 1, -1)
    elif kernel.ndim == 2:
        kernel = kernel.view(1, *kernel.shape)

    kernel_size = kernel.shape[-1]
    padding = (kernel_size - 1) // 2

    if batch_size is None:
        output = F.conv1d(input_reshaped, kernel, padding=padding)
    else:
        output_chunks = []
        for batch in tqdm(torch.split(input_reshaped, batch_size, dim=0)):
            output_chunks.append(F.conv1d(batch, kernel, padding=padding))
        output = torch.cat(output_chunks, dim=0)

    # Reshape back to (T, C, X, Y)
    return output.reshape(C, X, Y, T).permute(3, 0, 1, 2)
