"""
cellstream.image.utils

Low-level utilities for image and mask processing.
@authors: coylelab

Functions:
- downsample(tensor, scale, is_mask=False):
    Resize an image or mask tensor by a scale factor.
    Uses average pooling for image data and nearest neighbor for masks to preserve label integrity.

- normalize_histogram(image):
    Normalize a 4D image tensor (C, T, H, W) per-frame to zero mean and unit variance.
    Flattens each spatial frame (across C, T) and normalizes individually.

- convolve_along_timeseries(video_tensor, kernel_weights, batch_size=512):
    Convolve a 1D kernel along the time axis of a video tensor (C,T,H,W) using grouped Conv1d.
    Efficient batched implementation that supports GPU and large inputs.

- color_by_axis(img, cmap='turbo', proj='max'):
    Color-codes a (T,C, H, W) timeseries stack along the time axis using a specified colormap.
    Supports max or sum projection for visualizing time-resolved activity in false color.
"""

import functools

import matplotlib.pyplot as plt
import progressbar
import torch


def downsample(tensor, scale, is_mask=False):
    original_dim = tensor.dim()
    dtype = tensor.dtype

    # --- Standardize shape to (B, C, H, W) ---
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    _, _, H, W = tensor.shape

    # --- Determine target size from scale ---

    new_H = max(1, int(round(H * scale)))
    new_W = max(1, int(round(W * scale)))

    target_size = (new_H, new_W)

    # --- Downsample using appropriate method ---
    if is_mask:
        tensor = tensor.float()
        out = torch.nn.functional.interpolate(tensor, size=target_size, mode="nearest")
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
    # correct for intensity changes over time
    image = normalize_dims(image, 1)
    T, C, X, Y = image.shape
    image = image.reshape(T, C, X * Y)
    image = (
        (image - image.mean(axis=2, keepdim=True)) / (image.std(axis=2, keepdim=True))
    ).reshape(T, C, X, Y)
    return image


def normalize_dims(image, channel_dim):
    """
    Normalize the number of dimensions for input images to 4, and return the image.
    If no channel dimension is detected, one is added using unsqueeze().
    Similiar to cellstream.image.loaders.load_image().
    Parameters:
        image: image tensor with 3 dimensions (single channel) or 4 dimensions (multi-channel).
        channel_dim: axis to unsqueeze if single-channel image is detected.
    Returns:
        image: 4D tensor of images with C channels. Dimensions are
                determined by input image and channel_dim parameter.
    """
    if len(image.shape) == 4:
        return image
    elif len(image.shape) == 3:
        print("Single-channel image detected; adding channel dimension...")
        image = image.unsqueeze(channel_dim)
        return image
    else:
        raise ValueError(
            f"Expected an image with dimension 3 or 4. Got an image with dimension {len(image.shape)}"
        )


def convolve_along_timeseries(video_tensor, kernel_weights, batch_size=512):
    T, C, H, W = video_tensor.shape
    input_reshaped = video_tensor.permute(1, 2, 3, 0).reshape(-1, 1, T)

    my_kernel_size = len(kernel_weights)
    my_padding = (my_kernel_size - 1) // 2

    # Define the conv layer on CPU
    conv = torch.nn.Conv1d(
        1, 1, kernel_size=my_kernel_size, padding=my_padding, bias=False
    )
    conv.weight.data = torch.tensor([[kernel_weights]], dtype=torch.float32)
    conv.weight.requires_grad_(False)

    # Batch processing
    output_chunks = []
    for batch in progressbar.progressbar(
        torch.split(input_reshaped, batch_size, dim=0)
    ):
        with torch.no_grad():
            output_chunk = conv(batch)
        output_chunks.append(output_chunk)

    output = torch.cat(output_chunks, dim=0)
    return output.reshape(C, H, W, T).permute(3, 0, 1, 2)


def color_by_axis(img: torch.Tensor, cmap="turbo", proj="max", minmax_norm=True):
    """
    Apply a colormap along 0 axis (typically frequencey or scale bins) (F, C, X, Y),
    returning (C, X, Y, 3) RGB images.

    5D images are accomodated by conversion to 4D and back.

    Parameters:
        img: (T, C, X, Y) tensor
        cmap: matplotlib colormap name
        proj: 'max' or 'sum'
        minmax_norm: True -> performs minmax normalization before coloring

    Returns:
        (C, X, Y, 3) tensor of RGB images
    """

    img_ndim = img.dim()
    five_d = False
    if img_ndim == 4:
        print("4D")
        F, C, X, Y = img.shape
    elif img_ndim == 5:  # coming from generate_cwt_features
        five_d = True
        C, T, F, X, Y = img.shape
        img = img.permute(2, 0, 1, 3, 4)  # F,T,C,X,Y
        img = img.reshape(F, T * C, X, Y)  # 4D w/ T*C bins
        C = T * C

    if minmax_norm:
        img_min = torch.min(img)
        img_max = torch.max(img)
        img = (img - img_min) / (img_max - img_min)

    # (T, 3) colormap, normalized to [0,1]
    colors = torch.tensor(plt.get_cmap(cmap).resampled(F)(range(F)), dtype=img.dtype)[
        :, :3
    ]  # (F, 3)

    # (T, 1, 1, 3) for broadcasting
    colors = colors[:, None, None, :]

    # Allocate output (C, X, Y, 3)
    out = torch.zeros((C, X, Y, 3), dtype=img.dtype)

    for c in range(C):
        # (T, X, Y)
        channel_img = img[:, c, :, :]

        # (T, X, Y, 3)
        color_stack = colors * channel_img[:, :, :, None]

        if proj == "max":
            proj_rgb = color_stack.max(dim=0).values  # (X, Y, 3)
        elif proj == "sum":
            proj_rgb = color_stack.sum(dim=0)  # (X, Y, 3)
        else:
            raise ValueError("proj must be 'max' or 'sum'")

        out[c] = proj_rgb

    if five_d:
        out = torch.stack(out.split(T), dim=1)

    return out


# patch napari to accept tensors for adding to viewer
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


# for Viewer.add_* methods
def _wrap_add_method(method):
    @functools.wraps(method)
    def wrapper(self, data, *args, **kwargs):
        return method(self, _to_numpy(data), *args, **kwargs)

    return wrapper


# for top-level napari.view_* helpers
def _wrap_add_func(func):
    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        return func(_to_numpy(data), *args, **kwargs)

    return wrapper


def patch_napari_for_torch():
    # patch Viewer methods
    import napari

    napari.Viewer.add_image = _wrap_add_method(napari.Viewer.add_image)
    napari.Viewer.add_labels = _wrap_add_method(napari.Viewer.add_labels)
    napari.Viewer.add_points = _wrap_add_method(napari.Viewer.add_points)
    napari.Viewer.add_shapes = _wrap_add_method(napari.Viewer.add_shapes)

    # patch top-level helpers
    napari.view_image = _wrap_add_func(napari.view_image)
    napari.view_labels = _wrap_add_func(napari.view_labels)
    napari.view_points = _wrap_add_func(napari.view_points)
    napari.view_shapes = _wrap_add_func(napari.view_shapes)
