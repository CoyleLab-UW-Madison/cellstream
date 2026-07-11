"""
cellstream.spatial.crop

Crop feature arrays into per-cell zarr stores using a 2D label mask.
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
import zarr
from tqdm.auto import tqdm

from ..io import TorchZarrStore, _sanitize_metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    """Convert a torch.Tensor or array-like to a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _find_first_spatial_array(d, spatial_shape):
    """Walk dict recursively to find the first array matching spatial_shape."""
    for k, v in d.items():
        if k == "_attrs":
            continue
        if isinstance(v, dict):
            res = _find_first_spatial_array(v, spatial_shape)
            if res is not None:
                return res
        else:
            arr = _to_numpy(v)
            if arr.ndim >= 2 and arr.shape[-2:] == spatial_shape:
                return arr
    return None


def _load_features_from_source(source):
    """Recursively load all arrays from a *TorchZarrStore* or *zarr.Group*
    into a plain dict of numpy arrays, plus extract the ``_attrs`` dict.

    Parameters
    ----------
    source : TorchZarrStore | zarr.Group
        The source to load from.

    Returns
    -------
    data : dict
        Nested dict whose leaves are numpy arrays.
    attrs : dict
        The ``_attrs`` metadata dict (empty dict if none present).
    """
    data = {}
    attrs = {}

    for key in source.keys():
        if key == "_attrs":
            attrs = dict(source[key]) if isinstance(source[key], dict) else dict(source[key])
            continue

        value = source[key]

        # Sub-group / nested TorchZarrStore
        if isinstance(value, (TorchZarrStore, dict)):
            sub_data, sub_attrs = _load_features_from_source(value)
            if sub_attrs:
                sub_data["_attrs"] = sub_attrs
            data[key] = sub_data
        elif hasattr(value, "keys"):
            # zarr sub-group (has .keys() but isn't a plain dict)
            sub_data, sub_attrs = _load_features_from_source(value)
            if sub_attrs:
                sub_data["_attrs"] = sub_attrs
            data[key] = sub_data
        else:
            # Leaf array — convert to numpy
            data[key] = _to_numpy(value)

    return data, attrs


def _compute_padded_bbox(mask_2d, padding_fraction, min_padding_px, image_shape):
    """Compute padded bounding box for a single binary mask.

    Parameters
    ----------
    mask_2d : np.ndarray
        2-D boolean mask for one cell.
    padding_fraction : float
        Fraction of the bounding-box size to add as padding on each side.
    min_padding_px : int
        Minimum number of padding pixels on each side.
    image_shape : tuple[int, int]
        Full (Y, X) shape of the label image, used for clamping.

    Returns
    -------
    y_min_p, y_max_p, x_min_p, x_max_p : int
        Padded bounding-box coordinates (y_max_p and x_max_p are exclusive).
    y_min, y_max, x_min, x_max : int
        Original (unpadded) bounding-box coordinates.
    """
    ys, xs = np.where(mask_2d)
    y_min, y_max = int(ys.min()), int(ys.max()) + 1  # exclusive end
    x_min, x_max = int(xs.min()), int(xs.max()) + 1

    height = y_max - y_min
    width = x_max - x_min

    pad_y = max(min_padding_px, int(np.ceil(height * padding_fraction)))
    pad_x = max(min_padding_px, int(np.ceil(width * padding_fraction)))

    y_min_p = max(0, y_min - pad_y)
    y_max_p = min(image_shape[0], y_max + pad_y)
    x_min_p = max(0, x_min - pad_x)
    x_max_p = min(image_shape[1], x_max + pad_x)

    return y_min_p, y_max_p, x_min_p, x_max_p, y_min, y_max, x_min, x_max


def _crop_and_write_recursive(source_dict, zarr_group, y_slice, x_slice,
                               spatial_shape, compressor):
    """Recursively walk *source_dict* and write cropped arrays into *zarr_group*.

    Arrays whose last two dimensions match *spatial_shape* are spatially
    cropped; all other arrays are copied verbatim.

    Parameters
    ----------
    source_dict : dict
        Nested dict of numpy arrays (leaves) and sub-dicts (branches).
    zarr_group : zarr.Group
        Target zarr group to write into.
    y_slice : slice
        Row slice for the crop window.
    x_slice : slice
        Column slice for the crop window.
    spatial_shape : tuple[int, int]
        Expected (Y, X) shape of the full-field arrays.
    compressor : object
        Compressor to use for zarr arrays.
    """
    # Write _attrs if present
    if "_attrs" in source_dict and isinstance(source_dict["_attrs"], dict):
        for meta_k, meta_v in source_dict["_attrs"].items():
            try:
                zarr_group.attrs[str(meta_k)] = _sanitize_metadata(meta_v)
            except Exception as e:
                logger.warning(f"Could not save attribute {meta_k} to Zarr: {e}")

    for key, value in source_dict.items():
        if key == "_attrs":
            continue

        str_key = str(key)

        # Sub-dict → create sub-group and recurse
        if isinstance(value, dict):
            sub_group = zarr_group.create_group(str_key)
            _crop_and_write_recursive(value, sub_group, y_slice, x_slice,
                                      spatial_shape, compressor)
            continue

        # Leaf: ensure numpy
        arr = _to_numpy(value)

        # Decide whether to crop: ndim >= 2 and last two dims match spatial
        if arr.ndim >= 2 and arr.shape[-2:] == spatial_shape:
            arr = arr[..., y_slice, x_slice]

        # Write to zarr (v2/v3 compatible)
        if hasattr(zarr_group, "create_array"):
            out = zarr_group.create_array(
                name=str_key, shape=arr.shape, dtype=arr.dtype,
                chunks=True, compressor=compressor, overwrite=True,
            )
        else:
            out = zarr_group.create_dataset(
                name=str_key, shape=arr.shape, dtype=arr.dtype,
                chunks=True, compressor=compressor, overwrite=True,
            )
        out[:] = arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crop_zarr_from_masks(
    features,
    label_image,
    output_path,
    padding_fraction=0.1,
    min_padding_px=2,
    background_label=0,
    compressor="default",
    show_progress=True,
):
    """Crop per-cell feature arrays from a features dict using a 2-D label mask.

    Each unique label in *label_image* (excluding *background_label*) produces
    a sub-group in the output zarr store containing:

    * ``mask`` – the cropped binary mask for that cell
    * all feature arrays cropped to the same padded bounding box
    * per-cell metadata (centroid, area, bounding-box coordinates, …)

    Parameters
    ----------
    features : dict | str | TorchZarrStore
        Feature data to crop.  Accepted forms:

        * **dict** – a plain Python dict whose leaves are ``torch.Tensor`` or
          ``numpy.ndarray`` (as produced by ``fft_features``, ``cwt_features``,
          etc.).  May be flat or nested.
        * **str** – path to a zarr store on disk (opened read-only).
        * **TorchZarrStore** – a cellstream zarr wrapper.

    label_image : numpy.ndarray | torch.Tensor
        2-D integer array where each pixel value is a cell label.
        ``background_label`` pixels are ignored.

    output_path : str
        Path where the output zarr store will be written.

    padding_fraction : float, optional
        Fraction of each bounding-box dimension to add as padding on each
        side (default ``0.1``).

    min_padding_px : int, optional
        Minimum number of padding pixels on each side (default ``2``).

    background_label : int, optional
        Label value that represents background (default ``0``).

    compressor : str or object, optional
        Compressor for zarr arrays.  ``"default"`` uses Zstandard level 5
        (via ``zarr.Blosc`` on v2 or a config dict on v3).

    Returns
    -------
    zarr.Group
        The root group of the newly-created zarr store.
    """
    # ------------------------------------------------------------------
    # 1. Normalize inputs
    # ------------------------------------------------------------------
    if isinstance(label_image, torch.Tensor):
        label_image = label_image.detach().cpu().numpy()
    label_image = np.asarray(label_image)
    if label_image.ndim != 2:
        raise ValueError(
            f"label_image must be 2-D, got shape {label_image.shape}"
        )

    # Load features into a plain dict of numpy arrays
    source_attrs: dict = {}

    if isinstance(features, str):
        logger.info("Opening zarr store at %s", features)
        store = TorchZarrStore(features)
        features_dict, source_attrs = _load_features_from_source(store)
    elif isinstance(features, TorchZarrStore):
        features_dict, source_attrs = _load_features_from_source(features)
    elif isinstance(features, dict):
        # Work on a shallow reference; tensors are converted lazily later
        features_dict = features
        source_attrs = features.get("_attrs", {})
    else:
        raise TypeError(
            f"Unsupported features type: {type(features)}. "
            "Expected dict, str (zarr path), or TorchZarrStore."
        )

    # ------------------------------------------------------------------
    # 2. Setup compressor (zarr v2/v3 pattern from io.py)
    # ------------------------------------------------------------------
    is_v3 = zarr.__version__.startswith("3.")
    if compressor == "default":
        if is_v3:
            compressor = {"name": "zstd", "level": 5}
        else:
            compressor = zarr.Blosc(
                cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE
            )

    # ------------------------------------------------------------------
    # 3. Extract label info
    # ------------------------------------------------------------------
    spatial_shape = tuple(label_image.shape)  # (Y, X)
    unique_labels = np.unique(label_image)
    labels = unique_labels[unique_labels != background_label]
    logger.info(
        "Found %d unique cell labels in label image of shape %s",
        len(labels), spatial_shape,
    )

    # ------------------------------------------------------------------
    # 4. Create output zarr group
    # ------------------------------------------------------------------
    root = zarr.open_group(str(output_path), mode="w")

    # ------------------------------------------------------------------
    # 5. Write root attrs
    # ------------------------------------------------------------------
    root_attrs = {
        "source_attrs": source_attrs,
        "num_cells": len(labels),
        "padding_fraction": padding_fraction,
        "min_padding_px": min_padding_px,
        "label_image_shape": list(label_image.shape),
        "background_label": background_label,
        "created_by": "cellstream.spatial.crop_zarr_from_masks",
    }
    for k, v in _sanitize_metadata(root_attrs).items():
        try:
            root.attrs[str(k)] = v
        except Exception as e:
            logger.warning(f"Could not save root attribute {k}: {e}")

    # ------------------------------------------------------------------
    # 6. Per-cell loop: compute bbox, crop, write
    # ------------------------------------------------------------------
    loop_iterable = tqdm(labels, desc="Cropping cells") if show_progress else labels
    for label_id in loop_iterable:
        # 6a. Binary mask for this cell
        cell_mask = (label_image == label_id)

        # 6b. Padded bounding box
        (y_min_p, y_max_p, x_min_p, x_max_p,
         y_min, y_max, x_min, x_max) = _compute_padded_bbox(
            cell_mask, padding_fraction, min_padding_px, spatial_shape,
        )

        # 6c. Crop binary mask to bbox region
        cropped_mask = cell_mask[y_min_p:y_max_p, x_min_p:x_max_p].astype(
            np.uint8
        )

        # 6d. Centroid and area
        ys, xs = np.where(cell_mask)
        cy = float(ys.mean())
        cx = float(xs.mean())
        area = int(cell_mask.sum())

        # 6e. Create cell group
        cell_group = root.create_group(f"cell_{int(label_id)}")

        # 6f. Per-cell attrs
        cell_attrs = {
            "label_id": int(label_id),
            "bbox_original": [int(y_min), int(y_max), int(x_min), int(x_max)],
            "bbox_padded": [int(y_min_p), int(y_max_p), int(x_min_p), int(x_max_p)],
            "centroid_yx": [float(cy), float(cx)],
            "area_pixels": int(area),
            "crop_shape": [int(y_max_p - y_min_p), int(x_max_p - x_min_p)],
            "padding_fraction": padding_fraction,
            "min_padding_px": min_padding_px,
        }
        for k, v in _sanitize_metadata(cell_attrs).items():
            try:
                cell_group.attrs[str(k)] = v
            except Exception as e:
                logger.warning(
                    f"Could not save cell {label_id} attribute {k}: {e}"
                )

        # 6g. Write cropped mask
        if hasattr(cell_group, "create_array"):
            mask_arr = cell_group.create_array(
                name="mask", shape=cropped_mask.shape, dtype=cropped_mask.dtype,
                chunks=True, compressor=compressor, overwrite=True,
            )
        else:
            mask_arr = cell_group.create_dataset(
                name="mask", shape=cropped_mask.shape, dtype=cropped_mask.dtype,
                chunks=True, compressor=compressor, overwrite=True,
            )
        mask_arr[:] = cropped_mask

        # 6g_2. Generate and write thumbnail (mean projection of timeseries or fallback spatial array)
        thumbnail = None
        y_slice = slice(y_min_p, y_max_p)
        x_slice = slice(x_min_p, x_max_p)
        
        if "timeseries" in features_dict:
            ts = _to_numpy(features_dict["timeseries"])
            if ts.ndim >= 2 and ts.shape[-2:] == spatial_shape:
                cropped_ts = ts[..., y_slice, x_slice]
                # Compute mean over time dimension (axis 0)
                thumbnail = cropped_ts.mean(axis=0)
        else:
            # Fallback to the first spatial array we find in the dict
            fallback_arr = _find_first_spatial_array(features_dict, spatial_shape)
            if fallback_arr is not None:
                cropped_arr = fallback_arr[..., y_slice, x_slice]
                non_spatial_axes = tuple(range(cropped_arr.ndim - 2))
                thumbnail = cropped_arr.mean(axis=non_spatial_axes) if non_spatial_axes else cropped_arr

        if thumbnail is not None:
            if hasattr(cell_group, "create_array"):
                thumb_arr = cell_group.create_array(
                    name="thumbnail", shape=thumbnail.shape, dtype=thumbnail.dtype,
                    chunks=True, compressor=compressor, overwrite=True,
                )
            else:
                thumb_arr = cell_group.create_dataset(
                    name="thumbnail", shape=thumbnail.shape, dtype=thumbnail.dtype,
                    chunks=True, compressor=compressor, overwrite=True,
                )
            thumb_arr[:] = thumbnail

        # 6h. Crop and write all feature arrays
        y_slice = slice(y_min_p, y_max_p)
        x_slice = slice(x_min_p, x_max_p)
        _crop_and_write_recursive(
            features_dict, cell_group, y_slice, x_slice,
            spatial_shape, compressor,
        )

        logger.debug(
            "cell_%d: centroid=(%.1f, %.1f), area=%d, crop=%s",
            label_id, cy, cx, area,
            [y_max_p - y_min_p, x_max_p - x_min_p],
        )

    logger.info(
        "Wrote %d cell crops to %s", len(labels), output_path,
    )

    # ------------------------------------------------------------------
    # 7. Return the root zarr group
    # ------------------------------------------------------------------
    return root
