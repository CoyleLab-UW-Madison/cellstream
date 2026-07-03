"""
cellstream.large_data

Utilities for handling and processing very large timeseries datasets that exceed system memory.
"""

import logging
import tempfile
import numpy as np
import torch
import zarr
from tqdm.auto import tqdm
import tifffile
import nd2
import shutil
import os

logger = logging.getLogger(__name__)

def convert_to_t_chunked_zarr(input_filename, output_zarr, spatial_chunk=(64, 64), temp_dir=None, compressor="default", dtype=np.float32, normalize=False, downsample_scale=None):
    """
    Converts a large image (ND2 or TIFF) into a Zarr array optimized for time-series analysis.
    The output Zarr will have chunks sized (T, C, block_x, block_y) so that the entire time
    series for a spatial block can be loaded into memory at once without loading the full image.
    
    This function uses a two-pass approach with an intermediate Zarr store to guarantee bounded 
    memory usage even for extremely large files (e.g., 30K+ frames).
    
    Args:
        input_filename: Path to the .nd2 or .tif file.
        output_zarr: Path to write the output Zarr store.
        spatial_chunk: Tuple of (X_chunk_size, Y_chunk_size).
        temp_dir: Optional directory to store the intermediate Zarr.
        compressor: Zarr compressor to use.
        dtype: Output data type (default: float32, which is required for most downstream pipelines).
        normalize: If True, applies histogram normalization to each frame.
        downsample_scale: If provided, downsamples the spatial dimensions by this factor (e.g. 0.5 for 2x).
    """
    input_filename = str(input_filename)
    filename_lower = input_filename.lower()
    
    is_nd2 = filename_lower.endswith(".nd2")
    is_tif = filename_lower.endswith((".tif", ".tiff"))
    
    # 1. Determine dimensions
    if is_nd2:
        file_obj = nd2.ND2File(input_filename)
        shape = file_obj.shape
        if len(shape) == 3:
            T, X, Y = shape
            C = 1
        elif len(shape) == 4:
            T, C, X, Y = shape
        else:
            raise ValueError(f"Unexpected ND2 shape: {shape}")
            
    elif is_tif:
        file_obj = tifffile.memmap(input_filename)
        shape = file_obj.shape
        if len(shape) == 3:
            T, X, Y = shape
            C = 1
        elif len(shape) == 4:
            T, C, X, Y = shape
        else:
            raise ValueError(f"Unexpected TIFF shape: {shape}")
    else:
        raise ValueError(f"Unsupported file format: {input_filename}")
        
    is_v3 = zarr.__version__.startswith("3.")
    if compressor == "default":
        if is_v3:
            compressor = {"name": "zstd", "level": 5}
        else:
            compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
            
    # Calculate dimensions after downsampling
    out_X, out_Y = X, Y
    if downsample_scale is not None:
        out_X = max(1, int(round(X * downsample_scale)))
        out_Y = max(1, int(round(Y * downsample_scale)))
            
    # Create intermediate Zarr store chunked by time (e.g. 100 frames at a time)
    # This makes sequential writing from the source file fast.
    t_chunk_size = min(T, max(1, 100000000 // (C * out_X * out_Y * np.dtype(dtype).itemsize))) # roughly 100MB per chunk
    
    temp_zarr_path = tempfile.mkdtemp(dir=temp_dir, prefix="cellstream_temp_")
    
    from .utils import normalize_histogram, downsample
    
    def process_frame(frame):
        if C == 1 and frame.ndim == 2:
            frame = frame[np.newaxis, :, :]
        # Convert to Torch tensor for processing
        ft = torch.from_numpy(frame.astype(np.float32)).unsqueeze(0) # (1, C, X, Y)
        
        if normalize:
            ft = normalize_histogram(ft)
        if downsample_scale is not None:
            ft = downsample(ft, downsample_scale)
            
        return ft.squeeze(0).cpu().numpy().astype(dtype)
    
    try:
        tmp_z = zarr.open(temp_zarr_path, mode="w", shape=(T, C, out_X, out_Y), dtype=dtype,
                          chunks=(t_chunk_size, C, out_X, out_Y), compressor=compressor)
        
        logger.info(f"Pass 1/2: Copying and pre-processing frames to intermediate store (T={T})...")
        
        with tqdm(total=T, desc="Reading frames") as pbar:
            for t in range(T):
                frame = file_obj.read_frame(t) if hasattr(file_obj, "read_frame") else file_obj[t]
                tmp_z[t, :, :, :] = process_frame(frame)
                pbar.update(1)
        
        if is_nd2:
            file_obj.close()
        
        # Pass 2: Rechunk to output Zarr
        chunk_t = T
        chunk_c = C
        chunk_x, chunk_y = spatial_chunk
        
        out_z = zarr.open(str(output_zarr), mode="w", shape=(T, C, out_X, out_Y), dtype=dtype,
                          chunks=(chunk_t, chunk_c, chunk_x, chunk_y), compressor=compressor)
                          
        logger.info(f"Pass 2/2: Rechunking to spatial blocks {spatial_chunk}...")
        
        x_blocks = (out_X + chunk_x - 1) // chunk_x
        y_blocks = (out_Y + chunk_y - 1) // chunk_y
        total_blocks = x_blocks * y_blocks
        
        with tqdm(total=total_blocks, desc="Writing spatial blocks") as pbar:
            for x in range(0, out_X, chunk_x):
                for y in range(0, out_Y, chunk_y):
                    x_end = min(x + chunk_x, out_X)
                    y_end = min(y + chunk_y, out_Y)
                    
                    # Reading from intermediate zarr is memory-efficient because it only
                    # pulls the needed spatial regions from the temporal chunks.
                    block = tmp_z[:, :, x:x_end, y:y_end]
                    out_z[:, :, x:x_end, y:y_end] = block
                    
                    pbar.update(1)
                    
        logger.info(f"Rechunking complete. Output saved to {output_zarr}.")
        return out_z
        
    finally:
        # Clean up intermediate Zarr
        try:
            shutil.rmtree(temp_zarr_path)
        except OSError as e:
            logger.warning(f"Failed to remove temporary directory {temp_zarr_path}: {e}")

def _init_zarr_structure(group, data_template, full_X, full_Y, chunk_x, chunk_y, compressor):
    from .io import _sanitize_metadata
    if "_attrs" in data_template and isinstance(data_template["_attrs"], dict):
        for k, v in data_template["_attrs"].items():
            group.attrs[str(k)] = _sanitize_metadata(v)
            
    for k, v in data_template.items():
        if k == "_attrs": continue
        if isinstance(v, dict):
            subgroup = group.create_group(str(k))
            _init_zarr_structure(subgroup, v, full_X, full_Y, chunk_x, chunk_y, compressor)
        else:
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            elif not isinstance(v, np.ndarray):
                v = np.asarray(v)
                
            if v.ndim < 2:
                shape = v.shape
                chunks = v.shape
            else:
                shape = list(v.shape)
                shape[-2] = full_X
                shape[-1] = full_Y
                chunks = list(v.shape)
                chunks[-2] = chunk_x
                chunks[-1] = chunk_y
                
            if hasattr(group, "create_array"):
                group.create_array(name=str(k), shape=tuple(shape), dtype=v.dtype, chunks=tuple(chunks), compressor=compressor, overwrite=True)
            else:
                group.create_dataset(name=str(k), shape=tuple(shape), dtype=v.dtype, chunks=tuple(chunks), compressor=compressor, overwrite=True)

def _write_block_to_zarr(group, data_block, x_slice, y_slice):
    for k, v in data_block.items():
        if k == "_attrs": continue
        if isinstance(v, dict):
            _write_block_to_zarr(group[str(k)], v, x_slice, y_slice)
        else:
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            elif not isinstance(v, np.ndarray):
                v = np.asarray(v)
                
            if v.ndim < 2:
                # non-spatial data, assign fully
                group[str(k)][:] = v
            else:
                group[str(k)][..., x_slice, y_slice] = v

def process_t_chunked_zarr(zarr_path, output_zarr_path, process_fn, **kwargs):
    """
    Process a T-chunked Zarr store block by block using a custom processing function (e.g. CWT).
    Supports processing functions that return either a single tensor or a nested dictionary of tensors.
    
    Args:
        zarr_path (str): Path to the input T-chunked Zarr store.
        output_zarr_path (str): Path to write the processed output Zarr.
        process_fn (callable): A function that takes a Torch tensor of shape (T, C, block_x, block_y)
                               and returns a processed Torch tensor or dictionary of tensors.
        kwargs: Additional arguments to pass to the process_fn.
    """
    z_in = zarr.open(str(zarr_path), mode="r")
    
    T, C, X, Y = z_in.shape
    chunk_t, chunk_c, chunk_x, chunk_y = z_in.chunks
    
    if chunk_t != T:
        warnings.warn("The input Zarr does not appear to be T-chunked (chunk_t != T). "
                      "Processing may be extremely slow and memory intensive.")
    
    # Process the first block to determine output dimensions
    first_block_x = min(chunk_x, X)
    first_block_y = min(chunk_y, Y)
    
    test_array = np.asarray(z_in[:, :, 0:first_block_x, 0:first_block_y])
    if test_array.dtype.byteorder not in ('=', '|'):
        test_array = test_array.astype(test_array.dtype.newbyteorder('='))
    test_block = torch.from_numpy(test_array)
    test_out = process_fn(test_block, **kwargs)
    
    is_v3 = zarr.__version__.startswith("3.")
    compressor = {"name": "zstd", "level": 5} if is_v3 else zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
    
    is_dict = isinstance(test_out, dict)
    
    if is_dict:
        z_out = zarr.open_group(str(output_zarr_path), mode="w")
        _init_zarr_structure(z_out, test_out, X, Y, chunk_x, chunk_y, compressor)
    else:
        if not isinstance(test_out, torch.Tensor):
            test_out = torch.as_tensor(test_out)
        T_out, C_out, _, _ = test_out.shape
        z_out = zarr.open(str(output_zarr_path), mode="w", shape=(T_out, C_out, X, Y), 
                          dtype=test_out.numpy().dtype,
                          chunks=(T_out, C_out, chunk_x, chunk_y),
                          compressor=compressor)
                          
    x_blocks = (X + chunk_x - 1) // chunk_x
    y_blocks = (Y + chunk_y - 1) // chunk_y
    total_blocks = x_blocks * y_blocks
    
    with tqdm(total=total_blocks, desc="Processing blocks") as pbar:
        for x in range(0, X, chunk_x):
            for y in range(0, Y, chunk_y):
                x_end = min(x + chunk_x, X)
                y_end = min(y + chunk_y, Y)
                
                if x == 0 and y == 0:
                    out_block = test_out
                else:
                    block_array = np.asarray(z_in[:, :, x:x_end, y:y_end])
                    if block_array.dtype.byteorder not in ('=', '|'):
                        block_array = block_array.astype(block_array.dtype.newbyteorder('='))
                    block = torch.from_numpy(block_array)
                    out_block = process_fn(block, **kwargs)
                    
                x_slice = slice(0, x_end - x)
                y_slice = slice(0, y_end - y)
                full_x_slice = slice(x, x_end)
                full_y_slice = slice(y, y_end)
                
                if is_dict:
                    _write_block_to_zarr(z_out, out_block, full_x_slice, full_y_slice)
                else:
                    if isinstance(out_block, torch.Tensor):
                        out_block = out_block.detach().cpu().numpy()
                    z_out[..., full_x_slice, full_y_slice] = out_block
                pbar.update(1)
                
    return z_out
