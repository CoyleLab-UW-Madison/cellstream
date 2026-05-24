"""
cellstream

Microscopy and single-cell signal processing analysis tools.
"""

__version__ = "0.1.0"

# Core foundation modules
from . import io
from . import utils
from . import analysis
from . import viz
from . import image

# Specialized transform/processing modules
from . import fft
from . import cwt
from . import hilbert
from . import filters
from . import registration
from . import phase

# Convenience re-exports for common functions
from .io import load_image, load_masks, load_zarr, write_to_zarr
from .utils import normalize_dims, downsample, normalize_histogram, convolve_along_timeseries, corr_along_axis, hann_image_series
from .analysis import extract_single_cell_data, create_dataframe, reshape_to_longform
from .viz import color_by_axis, patch_napari_for_torch, map_data_onto_mask
from .phase import winding_number

__all__ = [
    "io",
    "utils",
    "analysis",
    "viz",
    "image",
    "fft",
    "cwt",
    "hilbert",
    "filters",
    "registration",
    "phase",
    "load_image",
    "load_masks",
    "load_zarr",
    "write_to_zarr",
    "normalize_dims",
    "downsample",
    "normalize_histogram",
    "convolve_along_timeseries",
    "corr_along_axis",
    "hann_image_series",
    "extract_single_cell_data",
    "create_dataframe",
    "reshape_to_longform",
    "map_data_onto_mask",
    "color_by_axis",
    "patch_napari_for_torch",
    "winding_number",
]
