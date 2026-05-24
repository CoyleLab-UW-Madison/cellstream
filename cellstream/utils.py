"""
cellstream.utils

General utility functions for image processing and tensor manipulation.
"""

import torch
import torch.nn.functional as F
import progressbar

def normalize_dims(image, channel_dim=1):
    """
    Ensure the image tensor is 4D (T, C, X, Y).
    If 3D, adds a channel dimension at channel_dim.
    """
    if image.dim() == 4:
        return image
    elif image.dim() == 3:
        print("Single-channel image detected; adding channel dimension...")
        return image.unsqueeze(channel_dim)
    else:
        raise ValueError(
            f"Expected image with 3 or 4 dimensions, got {image.dim()}."
        )

def downsample(tensor, scale, is_mask=False):
    """
    Spatially downsample an image or mask tensor.
    """
    original_dim = tensor.dim()
    dtype = tensor.dtype

    # Standardize shape to (B, C, H, W)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    _, _, H, W = tensor.shape
    new_H = max(1, int(round(H * scale)))
    new_W = max(1, int(round(W * scale)))
    target_size = (new_H, new_W)

    if is_mask:
        tensor = tensor.float()
        out = F.interpolate(tensor, size=target_size, mode="nearest")
        out = out.to(dtype)
    else:
        out = F.adaptive_avg_pool2d(tensor, output_size=target_size)

    # Restore original shape
    if original_dim == 2:
        return out.squeeze(0).squeeze(0)
    elif original_dim == 3:
        return out.squeeze(0)
    return out

def normalize_histogram(image):
    """
    Normalize image intensity per-frame to zero mean and unit variance.
    """
    image = normalize_dims(image, 1)
    T, C, X, Y = image.shape
    image_flat = image.reshape(T, C, X * Y)
    means = image_flat.mean(axis=2, keepdim=True)
    stds = image_flat.std(axis=2, keepdim=True)
    normed = (image_flat - means) / (stds + 1e-8)
    return normed.reshape(T, C, X, Y)

def convolve_along_timeseries(video_tensor, kernel_weights, batch_size=512):
    """
    Apply a 1D convolution along the time axis (dimension 0).
    """
    T, C, H, W = video_tensor.shape
    # Reshape to (N, 1, T) for Conv1d
    input_reshaped = video_tensor.permute(1, 2, 3, 0).reshape(-1, 1, T)

    kernel_size = len(kernel_weights)
    padding = (kernel_size - 1) // 2

    conv = torch.nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
    conv.weight.data = torch.as_tensor(kernel_weights, dtype=torch.float32).view(1, 1, -1)
    conv.weight.requires_grad_(False)

    output_chunks = []
    for batch in progressbar.progressbar(torch.split(input_reshaped, batch_size, dim=0)):
        with torch.no_grad():
            output_chunk = conv(batch)
        output_chunks.append(output_chunk)

    output = torch.cat(output_chunks, dim=0)
    # Reshape back to (T, C, H, W)
    return output.reshape(C, H, W, T).permute(3, 0, 1, 2)

def corr_along_axis(series_a, series_b, window=10, step=1, normalize_histogram=True):
    """
    Compute sliding window Pearson correlation between two image timeseries.
    Assumes (T, 1, H, W) shape.
    """
    if normalize_histogram:
        series_a = normalize_histogram(series_a)
        series_b = normalize_histogram(series_b)
    
    # Unfold into windows
    a_unfold = series_a.unfold(0, window, step) # (num_windows, 1, H, W, window_size)
    b_unfold = series_b.unfold(0, window, step)

    # Center within windows
    a_centered = a_unfold - a_unfold.mean(dim=-1, keepdim=True)
    b_centered = b_unfold - b_unfold.mean(dim=-1, keepdim=True)
    
    # Pearson numerator
    pearson_num = (a_centered * b_centered).sum(dim=-1)
    
    # Pearson denominator
    a_var = torch.sum(a_centered**2, dim=-1)
    b_var = torch.sum(b_centered**2, dim=-1)
    pearson_denom = torch.sqrt(a_var * b_var)
    
    corr = pearson_num / (pearson_denom + 1e-8)
    return corr

def hann_image_series(img,norm_histogram=False):
    """
    Apply hanning window to image timeseries.
    Assumes (T,C,H,W) tensor input.
    """
    
    #for convenience
    if norm_histogram==True:
        img=normalize_histogram(img)
        
    T=img.shape[0]
    hann=torch.hann_window(T)
    hann_series=hann.view(T, 1, 1, 1)
    hann_img=img*hann_series
    
    return hann_img
