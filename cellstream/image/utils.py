"""
cellstream.image.utils

Low-level utilities for image and mask processing.
@authors: smcoyle,cxwong

Functions:
- downsample(tensor, scale, is_mask=False): Resize an image or mask tensor by a scale factor.
    Uses average pooling for image data and nearest neighbor for masks to preserve label integrity.

- normalize_histogram(image): Normalize a 5D image tensor (C, T, H, W) per-frame to zero mean and unit variance.
    Flattens each spatial frame (across C, T) and normalizes individually.

- convolve_along_timeseries(video_tensor, kernel_weights, batch_size=512): 
    Convolve a 1D kernel along the time axis of a video tensor (C,T,H,W) using grouped Conv1d.
    Efficient batched implementation that supports GPU and large inputs.

- color_by_axis(img, cmap='turbo', proj='max'):
    Color-codes a (T,C, H, W) timeseries stack along the time axis using a specified colormap.
    Supports max or sum projection for visualizing time-resolved activity in false color.
"""


import torch
import progressbar
import matplotlib.pyplot as plt

def downsample(
        tensor,
        scale, 
        is_mask=False
    ):
    
    
    original_dim = tensor.dim()
    dtype = tensor.dtype

    # --- Standardize shape to (B, C, H, W) ---
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    _, _, H, W = tensor.shape

    # --- Determine target size from scale ---
    if isinstance(scale, float):
        new_H = max(1, int(round(H * scale)))
        new_W = max(1, int(round(W * scale)))
    else:
        new_H, new_W = scale

    target_size = (new_H, new_W)

    # --- Downsample using appropriate method ---
    if is_mask:
        tensor = tensor.float()
        out = torch.nn.functional.interpolate(tensor, size=target_size, mode='nearest')
        out = out.to(dtype)
    else:
        out = torch.nn.functional.adaptive_avg_pool2d(tensor, output_size=target_size)

    # --- Restore original shape conventions ---
    if original_dim == 2:
        return out.squeeze(0).squeeze(0)
    elif original_dim == 3:
        return out.squeeze(0)
    else:
        return out

def normalize_histogram(image):
    
    #correct for intensity changes over time
    T,C,X,Y=image.shape
    image=image.reshape(T,C,X*Y)
    image=((image-image.mean(axis=2,keepdim=True))/(image.std(axis=2,keepdim=True))).reshape(T,C,X,Y)
    return image

def convolve_along_timeseries(video_tensor, kernel_weights, batch_size=512):
    T, C, H, W = video_tensor.shape
    input_reshaped = video_tensor.permute(1, 2, 3, 0).reshape(-1, 1, T)
    

    my_kernel_size = len(kernel_weights)
    my_padding = (my_kernel_size - 1) // 2

    # Define the conv layer on CPU
    conv = torch.nn.Conv1d(1, 1, kernel_size=my_kernel_size, padding=my_padding, bias=False)
    conv.weight.data = torch.tensor([[kernel_weights]], dtype=torch.float32)
    conv.weight.requires_grad_(False)

    # Batch processing
    output_chunks = []
    for batch in progressbar.progressbar(torch.split(input_reshaped, batch_size, dim=0)):
        with torch.no_grad():
            output_chunk = conv(batch)
        output_chunks.append(output_chunk)

    output = torch.cat(output_chunks, dim=0)
    return output.reshape(C, H, W, T).permute(3, 0, 1, 2)

def color_by_axis(img: torch.Tensor, cmap='turbo', proj='max', minmax_norm=True):
    """
    Apply a colormap along time (T) for each channel in (T, C, X, Y),
    returning (C, X, Y, 3) RGB images.

    Parameters:
        img: (T, C, X, Y) tensor
        cmap: matplotlib colormap name
        proj: 'max' or 'sum'
        minmax_norm: True -> performs minmax normalization before coloring

    Returns:
        (C, X, Y, 3) tensor of RGB images
    """
    T, C, X, Y = img.shape
    
    if minmax_norm==True:
        img_min=torch.min(img)
        img_max=torch.max(img)
        img=(img-img_min)/(img_max-img_min)

    # (T, 3) colormap, normalized to [0,1]
    colors = torch.tensor(plt.get_cmap(cmap).resampled(T)(range(T)), dtype=img.dtype)[:, :3]  # (T, 3)

    # (T, 1, 1, 3) for broadcasting
    colors = colors[:, None, None, :]

    # Allocate output (C, X, Y, 3)
    out = torch.zeros((C, X, Y, 3), dtype=img.dtype)

    for c in range(C):
        # (T, X, Y)
        channel_img = img[:, c, :, :]

        # (T, X, Y, 3)
        color_stack = colors * channel_img[:, :, :, None]

        if proj == 'max':
            proj_rgb = color_stack.max(dim=0).values  # (X, Y, 3)
        elif proj == 'sum':
            proj_rgb = color_stack.sum(dim=0)  # (X, Y, 3)
        else:
            raise ValueError("proj must be 'max' or 'sum'")

        out[c] = proj_rgb

    return out  # shape: (C, X, Y, 3)

# def color_by_fft_features(img,cmap='turbo',proj='max'):
#     slices,x,y=img.shape
#     cc=plt.colormaps.get_cmap(cmap).resampled(slices)(range(slices))
#     CC=torch.broadcast_to(cc,(x,y,slices,4)).swapaxes(0,2)[:,:,:,0:3].T
#     outstack=((CC*(img.T).swapaxes(0,1)).T)
#     if proj=='max':
#         out=(outstack.max(axis=0)).swapaxes(0,1)
#     elif proj=='sum':
#         out=(outstack.sum(axis=0)).swapaxes(0,1)
#     return out
