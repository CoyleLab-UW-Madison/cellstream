"""
cellstream.viz

Visualization utilities for images and masks.
"""

import matplotlib.pyplot as plt
import torch
import functools

def color_by_axis(img, axis=0, cmap="turbo", proj="max", minmax_norm=True):
    """
    Apply a colormap along the specified axis (e.g., frequency or filter banks).
    Returns an RGB image tensor where the chosen axis is reduced and a trailing '3' dimension is added.
    """
    if axis < 0:
        axis = img.ndim + axis

    if axis >= img.ndim - 2:
        raise ValueError("Cannot color along the spatial dimensions (last two dimensions).")

    # Normalize
    if minmax_norm:
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min + 1e-8)

    # Move the chosen axis to the front
    img_moved = img.movedim(axis, 0)
    F = img_moved.shape[0]
    X, Y = img_moved.shape[-2], img_moved.shape[-1]
    
    # Identify the remaining dimensions before X and Y
    other_dims = list(img_moved.shape[1:-2])
    
    # Flatten them into a single dimension C to process sequentially (prevents OOM on large stacks)
    import numpy as np
    C = int(np.prod(other_dims)) if len(other_dims) > 0 else 1
    
    img_reshaped = img_moved.reshape(F, C, X, Y)

    # Generate colors
    colors = torch.tensor(plt.get_cmap(cmap).resampled(F)(range(F)), dtype=img.dtype, device=img.device)[:, :3]
    colors = colors[:, None, None, :] # (F, 1, 1, 3)

    out = torch.zeros((C, X, Y, 3), dtype=img.dtype, device=img.device)

    from tqdm.auto import tqdm
    for c in tqdm(range(C), desc="Coloring frames", leave=False):
        channel_img = img_reshaped[:, c, :, :] # (F, X, Y)
        color_stack = colors * channel_img[:, :, :, None] # (F, X, Y, 3)

        if proj == "max":
            out[c] = color_stack.max(dim=0).values
        elif proj == "sum":
            out[c] = color_stack.sum(dim=0)
        else:
            raise ValueError("proj must be 'max' or 'sum'")

    # Reshape the output back to the original layout (excluding the colored axis, plus RGB)
    out = out.reshape(*other_dims, X, Y, 3)

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

def map_data_onto_mask(mask, df, column):
    """
    Map values from a DataFrame column back onto a label mask.
    """
    if "cell_id" in df.columns:
        df = df.set_index("cell_id")

    cell_ids = df.index.to_numpy()
    values = df[column].to_numpy()

    lut = torch.zeros(int(mask.max().item()) + 1, dtype=torch.as_tensor(values).dtype)
    lut[cell_ids] = torch.as_tensor(values)

    return lut[mask]
