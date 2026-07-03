"""
cellstream.io

Loading and saving images, masks, and Zarr stores.
"""

import logging
logger = logging.getLogger(__name__)
import nd2
import numpy as np
import tifffile
import torch
import zarr

class TorchZarrStore:
    """Wrapper around Zarr group or array (v2 or v3) to return Torch tensors."""
    def __init__(self, path_or_zarr):
        if hasattr(path_or_zarr, "attrs"):
            self._z = path_or_zarr
        else:
            self._z = zarr.open(str(path_or_zarr), mode="r")
            
        # Determine if this instance is an Array or a Group
        self._is_array = hasattr(self._z, "shape")

    @property
    def attrs(self):
        """Returns the attributes of the Zarr store."""
        return dict(self._z.attrs)

    def keys(self):
        """Returns the keys available in the Zarr store."""
        if not self._is_array:
           
            base_keys = list(self._z.keys())
        else:
            # For a bare array, we return a virtual key 'data'
            base_keys = ["data"]
        
        if len(self._z.attrs) > 0:
            base_keys.append("_attrs")
        return base_keys

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        if not self._is_array:
            return f"TorchZarrStore(keys={self.keys()})"
        else:
            return f"TorchZarrStore(shape={self._z.shape}, dtype={str(self._z.dtype)})"

    def __getitem__(self, key):
        if key == "_attrs" and len(self._z.attrs) > 0:
            return dict(self._z.attrs)

        if not self._is_array:
            if key not in self._z:
                raise KeyError(f"Key '{key}' not found. Available keys: {self.keys()}")
            
            item = self._z[key]
            # If the retrieved item doesn't have a shape, it's a sub-group
            if not hasattr(item, "shape"):
                return TorchZarrStore(item)
            
            
            return torch.from_numpy(np.asarray(item[:]))
        else:
            # If it's a bare array and we get a string key, return the whole array
            if isinstance(key, str):
                return torch.from_numpy(np.asarray(self._z[:]))
            
            return torch.as_tensor(np.asarray(self._z[key]))

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
        logger.info("Single-channel image detected; adding channel dimension...")
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
    Write a tensor, array, or dictionary of such to a Zarr store (Supports v2 and v3).
    """
    is_v3 = zarr.__version__.startswith("3.")

    if compressor == "default":
        if is_v3:
            # Zarr v3 configuration dictionary
            compressor = {"name": "zstd", "level": 5}
        else:
            # Legacy Zarr v2 Blosc object
            compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)

    #Base case: single Array/Tensor
    if isinstance(data, (torch.Tensor, np.ndarray)):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        z = zarr.open(str(path), mode="w", shape=data.shape, dtype=data.dtype, 
                      chunks=chunks, compressor=compressor)
        z[:] = data

    # Recursive case: Dictionaries
    elif isinstance(data, dict):
        root = zarr.open_group(str(path), mode="w")
        _write_dict_to_zarr_group(root, data, chunks=chunks, compressor=compressor)
    else:
        raise TypeError(f"Unsupported data type for write_to_zarr: {type(data)}")


def _sanitize_metadata(val):
    """Recursively convert values to JSON-serializable types for Zarr attributes."""
    if isinstance(val, dict):
        return {str(k): _sanitize_metadata(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_sanitize_metadata(item) for item in val]
    elif isinstance(val, (int, float, str, bool)) or val is None:
        return val
    else:
        return str(val)


def _write_dict_to_zarr_group(group, d, chunks=True, compressor=None):
    """Recursively write a dictionary to a Zarr group (v2 and v3 compatible)."""
    if "_attrs" in d and isinstance(d["_attrs"], dict):
        for meta_k, meta_v in d["_attrs"].items():
            try:
                group.attrs[str(meta_k)] = _sanitize_metadata(meta_v)
            except Exception as e:
                logger.warning(f"Warning: Could not save attribute {meta_k} to Zarr: {e}")

    for k, v in d.items():
        if k == "_attrs":
            continue
        key = str(k)
        
        if isinstance(v, dict):
            subgroup = group.create_group(key)
            _write_dict_to_zarr_group(subgroup, v, chunks=chunks, compressor=compressor)
        else:
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            elif not isinstance(v, np.ndarray):
                v = np.asarray(v)
                
            if hasattr(group, "create_array"):
                arr = group.create_array(name=key, shape=v.shape, dtype=v.dtype, 
                                         chunks=chunks, compressor=compressor, overwrite=True)
            else:
                arr = group.create_dataset(name=key, shape=v.shape, dtype=v.dtype, 
                                           chunks=chunks, compressor=compressor, overwrite=True)
            arr[:] = v
