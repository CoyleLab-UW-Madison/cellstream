"""
cellstream.image.utils

Legacy compatibility module for image utilities, redirecting to cellstream.utils and cellstream.viz.
"""

from ..utils import downsample, normalize_histogram, normalize_dims, convolve_along_timeseries
from ..viz import color_by_axis, patch_napari_for_torch

__all__ = [
    "downsample",
    "normalize_histogram",
    "normalize_dims",
    "convolve_along_timeseries",
    "color_by_axis",
    "patch_napari_for_torch",
]
