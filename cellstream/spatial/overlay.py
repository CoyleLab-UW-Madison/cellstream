"""
cellstream.spatial.overlay

Utilities for projecting single-cell properties back onto full-field spatial masks.
"""

import numpy as np

def paint_masks_with_property(label_image, property_dict, background=np.nan, dtype=np.float32):
    """
    Fill each labeled region in `label_image` with a scalar property value.

    Parameters
    ----------
    label_image : array-like of shape (H, W) or (1, H, W)
        Segmentation label image where pixels > 0 represent cell IDs.
    property_dict : dict
        Mapping of label_id (int) or cell_key (str, e.g. 'cell_1') to scalar value.
    background : float, optional
        Value assigned to non-cell background pixels. Default is np.nan (transparent in Napari).
    dtype : data-type, optional
        Desired data-type for the output array. Default is np.float32.

    Returns
    -------
    painted : ndarray of shape matching label_image
        2D spatial map with cell masks filled by their property value.
    """
    if hasattr(label_image, "cpu"):
        label_image = label_image.cpu().numpy()
    label_image = np.asarray(label_image)

    # Normalize property_dict keys to int label IDs
    clean_prop_dict = {}
    for k, v in property_dict.items():
        if isinstance(k, str):
            if k.startswith("cell_"):
                try:
                    k_id = int(k.replace("cell_", ""))
                except ValueError:
                    continue
            else:
                try:
                    k_id = int(k)
                except ValueError:
                    continue
        else:
            k_id = int(k)
            
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            try:
                clean_prop_dict[k_id] = float(v)
            except (ValueError, TypeError):
                continue

    painted = np.full(label_image.shape, background, dtype=dtype)
    
    unique_labels = np.unique(label_image)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        lbl_int = int(lbl)
        if lbl_int in clean_prop_dict:
            painted[label_image == lbl] = clean_prop_dict[lbl_int]

    return painted
