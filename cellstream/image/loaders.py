"""
cellstream.image.loaders

Handles loading of images into tensors.

@author: coylelab
"""

from pathlib import Path

import nd2
import numpy as np
import tifffile
import torch
import zarr


class TorchZarrStore:

    def __init__(self, path):
        # Open the store in read-only mode (instant, zero memory overhead)
        self._z = zarr.open(str(path), mode="r")

    @property
    def keys(self):
        """See what tensors are inside."""
        return list(self._z.array_keys())

    def __repr__(self):
        """show keys by default"""
        return f"TorchZarrStore keys: {self.keys}"

    def __getitem__(self, key):
        """Allows zarr_store['key'] syntax to instantly return a torch tensor."""
        if key not in self._z:
            raise KeyError(
                f"Tensor '{key}' not found. Available: {self.keys}"
            )
            
        return torch.from_numpy(self._z[key][:])


def load_zarr(zarr_name):
    return TorchZarrStore(zarr_name)

def load_image(image_filename):
    """Load image from file and convert to torch tensor"""
    *iname, iext = image_filename.split(".")
    if iext == "nd2":
        image = nd2.imread(image_filename)
    elif iext == "tif":
        image = tifffile.imread(image_filename)
    elif iext == "tiff":
        image = tifffile.imread(image_filename)
    else:
        print(f"{iext} is an unrecognized format...")
        return

    image = torch.from_numpy(image.astype("float32"))

    if image.dim() == 3:
        print("Single-channel image detected; adding channel dimension...")
        image = image.unsqueeze(1)

    return image


def load_masks(masks_filename):
    """Load masks from file and convert to torch tensor"""
    *mname, mext = masks_filename.split(".")
    if mext == "nd2":
        masks = nd2.imread(masks_filename)
    elif mext == "tif":
        masks = tifffile.imread(masks_filename)
    return torch.from_numpy(masks.astype("int64"))


def write_to_zarr(data, path, chunks=True, compressor="default"):
    """
    Write data (dictionary of tensors, or single tensor/array) to a Zarr store.

    Parameters:
    -----------
    data : dict, torch.Tensor, or np.ndarray
        Data to write. If a dictionary, it will be saved as a Zarr group.
    path : str or Path
        Output path for the Zarr store.
    chunks : bool or tuple, optional
        Chunking strategy for the Zarr arrays.
    compressor : zarr.Codec or str, optional
        Compressor to use. Defaults to Blosc zstd (clevel=5). 
        Set to None for no compression.
    """
    path = str(path)

    # Set up a smart default compressor if the user didn't specify one
    if compressor == "default":
        compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)

    if isinstance(data, (torch.Tensor, np.ndarray)):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        # Use zarr.open with mode='w' to properly handle single arrays with chunks
        z = zarr.open(
            path, mode="w", shape=data.shape, dtype=data.dtype, chunks=chunks, compressor=compressor
        )
        z[:] = data
    elif isinstance(data, dict):
        store = zarr.DirectoryStore(path)
        root = zarr.group(store=store, overwrite=True)
        _write_dict_to_zarr_group(root, data, chunks=chunks, compressor=compressor)
    else:
        raise TypeError(f"Unsupported data type for write_to_zarr: {type(data)}")


def _write_dict_to_zarr_group(group, d, chunks=True, compressor=None):
    """Recursively write a dictionary to a Zarr group."""
    for k, v in d.items():
        key = str(k)
        if isinstance(v, dict):
            subgroup = group.create_group(key)
            _write_dict_to_zarr_group(subgroup, v, chunks=chunks, compressor=compressor)
        elif isinstance(v, (torch.Tensor, np.ndarray)):
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            group.array(key, v, chunks=chunks, compressor=compressor)
        elif isinstance(v, (int, float, str, list, tuple)):
            # Save simple types as attributes (compression doesn't apply here)
            group.attrs[key] = v
        else:
            try:
                arr = np.array(v)
                group.array(key, arr, chunks=chunks, compressor=compressor)
            except Exception:
                print(f"Warning: Could not save key {key} of type {type(v)} to Zarr.")
