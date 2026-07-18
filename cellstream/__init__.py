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
from . import stft
from . import hilbert
from . import filters
from . import flow
from .flow import phase_velocity, binned_piv_velocity
from . import registration
from . import phase
from . import pixels
from . import experimental
from . import spatial

# Convenience re-exports for common functions
from .io import load_image, load_masks, load_zarr, write_to_zarr
from .large_data import convert_to_t_chunked_zarr, process_t_chunked_zarr
from .utils import normalize_dims, downsample, normalize_histogram, convolve_along_timeseries, corr_along_axis, hann_image_series
from .analysis import extract_single_cell_data, create_dataframe, reshape_to_longform
from .viz import color_by_axis, patch_napari_for_torch, map_data_onto_mask
from .phase import winding_number
from .spatial import crop_zarr_from_masks

__all__ = [
    "io",
    "utils",
    "analysis",
    "viz",
    "image",
    "fft",
    "cwt",
    "stft",
    "hilbert",
    "filters",
    "registration",
    "phase",
    "pixels",
    "experimental",
    "spatial",
    "features",
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
    "crop_zarr_from_masks",
]
