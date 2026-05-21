"""
cellstream.image

Legacy compatibility module for image loading and processing.
New code should use cellstream.io, cellstream.utils, and cellstream.viz.
"""

from .io import load_image, load_masks, load_zarr, write_to_zarr
from .utils import normalize_dims, downsample, normalize_histogram, convolve_along_timeseries
from .viz import color_by_axis, patch_napari_for_torch

__all__ = [
    "load_image",
    "load_masks",
    "load_zarr",
    "write_to_zarr",
    "normalize_dims",
    "downsample",
    "normalize_histogram",
    "convolve_along_timeseries",
    "color_by_axis",
    "patch_napari_for_torch",
]
