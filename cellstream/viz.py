"""
cellstream.viz

Visualization utilities for images and masks.
"""

import matplotlib.pyplot as plt
import torch
import functools

def color_by_axis(img, cmap="turbo", proj="max", minmax_norm=True):
    """
    Apply a colormap along the 0-th axis (e.g., frequency or scale bins).
    Returns (C, X, Y, 3) RGB images.
    """
    img_ndim = img.dim()
    five_d = False
    if img_ndim == 5:
        five_d = True
        C, T, F, X, Y = img.shape
        img = img.permute(2, 0, 1, 3, 4).reshape(F, T * C, X, Y)
        C = T * C
    elif img_ndim == 4:
        F, C, X, Y = img.shape
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {img_ndim}D")

    if minmax_norm:
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min + 1e-8)

    colors = torch.tensor(plt.get_cmap(cmap).resampled(F)(range(F)), dtype=img.dtype)[:, :3]
    colors = colors[:, None, None, :] # (F, 1, 1, 3)

    out = torch.zeros((C, X, Y, 3), dtype=img.dtype)

    for c in range(C):
        channel_img = img[:, c, :, :] # (F, X, Y)
        color_stack = colors * channel_img[:, :, :, None] # (F, X, Y, 3)

        if proj == "max":
            proj_rgb = color_stack.max(dim=0).values
        elif proj == "sum":
            proj_rgb = color_stack.sum(dim=0)
        else:
            raise ValueError("proj must be 'max' or 'sum'")
        
        out[c] = proj_rgb

    if five_d:
        out = torch.stack(out.split(T), dim=1)

    return out

def patch_napari_for_torch():
    """Patch Napari to automatically convert Torch tensors to numpy arrays."""
    try:
        import napari
    except ImportError:
        print("Napari not found. Skipping patch.")
        return

    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _wrap(method):
        @functools.wraps(method)
        def wrapper(self, data, *args, **kwargs):
            return method(self, _to_numpy(data), *args, **kwargs)
        return wrapper

    def _wrap_func(func):
        @functools.wraps(func)
        def wrapper(data, *args, **kwargs):
            return func(_to_numpy(data), *args, **kwargs)
        return wrapper

    napari.Viewer.add_image = _wrap(napari.Viewer.add_image)
    napari.Viewer.add_labels = _wrap(napari.Viewer.add_labels)
    napari.Viewer.add_points = _wrap(napari.Viewer.add_points)
    napari.Viewer.add_shapes = _wrap(napari.Viewer.add_shapes)

    napari.view_image = _wrap_func(napari.view_image)
    napari.view_labels = _wrap_func(napari.view_labels)
    napari.view_points = _wrap_func(napari.view_points)
    napari.view_shapes = _wrap_func(napari.view_shapes)
    
    print("Napari patched for Torch.")
