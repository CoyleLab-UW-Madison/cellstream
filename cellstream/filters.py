"""
cellstream.filters

Time-domain filtering utilities for image timeseries.
"""

import torch
import torch.nn.functional as F
import progressbar
from .utils import normalize_dims

def create_ir_filter(cutoff_freq, high_pass=False, window_size=101):
    """
    Create a sinc impulse response filter.
    Note: Requires torchaudio for sinc_impulse_response.
    """
    try:
        import torchaudio.prototype.functional as F_proto
    except ImportError:
        raise ImportError("torchaudio is required for create_ir_filter.")

    impulse_response = F_proto.sinc_impulse_response(cutoff_freq, window_size=window_size)
    
    if high_pass:
        delta = torch.zeros_like(impulse_response)
        delta[..., impulse_response.size(-1) // 2] = 1
        impulse_response = delta - impulse_response
    
    return impulse_response

def apply_fir_filter(image, impulse_response, batch_size=None):
    """
    Apply a FIR filter along the time axis of an image tensor (T, C, X, Y).
    """
    image = normalize_dims(image, 1)
    T, C, X, Y = image.shape
    N = C * X * Y

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
        for batch in progressbar.progressbar(torch.split(input_reshaped, batch_size, dim=0)):
            output_chunks.append(F.conv1d(batch, kernel, padding=padding))
        output = torch.cat(output_chunks, dim=0)

    # Reshape back to (T, C, X, Y)
    return output.reshape(C, X, Y, T).permute(3, 0, 1, 2)
