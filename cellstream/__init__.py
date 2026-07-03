"""
cellstream

Microscopy and single-cell signal processing analysis tools.
"""

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("cellstream")
except PackageNotFoundError:
    __version__ = "0.1.1"  # fallback for dev/editable installs

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
from . import experimental

# Convenience re-exports for common functions
from .io import load_image, load_masks, load_zarr, write_to_zarr
from .large_data import convert_to_t_chunked_zarr, process_t_chunked_zarr
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
    "experimental",
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
