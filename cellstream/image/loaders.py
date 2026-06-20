"""
cellstream.image.loaders

Legacy compatibility module for image loading, redirecting to cellstream.io.
"""

from ..io import TorchZarrStore, load_zarr, load_image, load_masks, write_to_zarr

__all__ = [
    "TorchZarrStore",
    "load_zarr",
    "load_image",
    "load_masks",
    "write_to_zarr"
]
