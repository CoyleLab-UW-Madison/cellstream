"""
cellstream.io

Loading and saving images, masks, and Zarr stores.
"""

import nd2
import numpy as np
import tifffile
import torch
import zarr

class TorchZarrStore:
    """Wrapper around Zarr group or array to return Torch tensors."""
    def __init__(self, path_or_zarr):
        if isinstance(path_or_zarr, (zarr.hierarchy.Group, zarr.core.Array)):
            self._z = path_or_zarr
        else:
            self._z = zarr.open(str(path_or_zarr), mode="r")

    def keys(self):
        """Returns the keys available in the Zarr store."""
        if isinstance(self._z, zarr.hierarchy.Group):
            return list(self._z.array_keys()) + list(self._z.group_keys())
        # For a bare array, we return a virtual key 'data'
        return ["data"]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        if isinstance(self._z, zarr.hierarchy.Group):
            return f"TorchZarrStore(keys={self.keys()})"
        else:
            return f"TorchZarrStore(shape={self._z.shape}, dtype={str(self._z.dtype)})"

    def __getitem__(self, key):
        if isinstance(self._z, zarr.hierarchy.Group):
            if key not in self._z:
                raise KeyError(f"Key '{key}' not found. Available keys: {self.keys()}")
            item = self._z[key]
            if isinstance(item, zarr.hierarchy.Group):
                return TorchZarrStore(item)
            return torch.from_numpy(item[:])
        else:
            # If it's a bare array and we get a string key, return the whole array
            if isinstance(key, str):
                return torch.from_numpy(self._z[:])
            return torch.as_tensor(self._z[key])

def load_zarr(path):
    """Open a Zarr store as a TorchZarrStore."""
    return TorchZarrStore(path)

def load_image(filename):
    """Load image from .tif or .nd2 and return as (T, C, X, Y) Torch tensor."""
    filename = str(filename)
    filename_lower = filename.lower()
    if filename_lower.endswith(".nd2"):
        data = nd2.imread(filename)
    elif filename_lower.endswith((".tif", ".tiff")):
        data = tifffile.imread(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

    tensor = torch.from_numpy(data.astype("float32"))
    
    # Ensure 4D (T, C, X, Y)
    if tensor.dim() == 3:
        print("Single-channel image detected; adding channel dimension...")
        tensor = tensor.unsqueeze(1)
    
    return tensor

def load_masks(filename):
    """Load masks from .tif or .nd2 and return as Torch tensor."""
    filename = str(filename)
    filename_lower = filename.lower()
    if filename_lower.endswith(".nd2"):
        data = nd2.imread(filename)
    elif filename_lower.endswith((".tif", ".tiff")):
        data = tifffile.imread(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
        
    return torch.from_numpy(data.astype("int64"))


def write_to_zarr(data, path, chunks=True, compressor="default"):
    """
    Write a tensor, array, or dictionary of such to a Zarr store.
    """
    if compressor == "default":
        compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)

    if isinstance(data, (torch.Tensor, np.ndarray)):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        z = zarr.open(path, mode="w", shape=data.shape, dtype=data.dtype, 
                      chunks=chunks, compressor=compressor)
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
            group.attrs[key] = v
        else:
            try:
                arr = np.array(v)
                group.array(key, arr, chunks=chunks, compressor=compressor)
            except Exception:
                print(f"Warning: Could not save key {key} of type {type(v)} to Zarr.")
