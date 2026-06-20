"""
cellstream.cwt.process

High-level Continuous Wavelet Transform (CWT) processing pipelines.
"""

import os
import progressbar
import torch
import pandas as pd
import numpy as np

from ..io import load_image, load_masks
from ..utils import downsample, normalize_dims
from .utils import generate_cwt_image_cellstreams, extract_cwt_cellstreams

def process_cwt_image_cellstreams(
    image,
    masks,
    min_scale=80,
    max_scale=180,
    num_filter_banks=1,
    normalize_amplitudes=False,
    blocks=10,
    use_gpu=False,
    bank_method="max_pool",
    downsample_by=None,
    normalize_histogram=True,
    mean_center=False,
    carrier_channel=0,
    channel_names=None,
    channel_outputs={0: ["amp", "freq", "phase"]},
    sampling=None,
    image_filename=None,
    masks_filename=None,
    **ssqueezepy_cwt_kwargs,
):
    """
    Full CWT-based processing pipeline for a single image stack and mask set.
    Generates a tidy pandas DataFrame containing extracted single-cell trajectories.
    """
    image = normalize_dims(image, 1)
    
    if downsample_by is not None:
        image = downsample(image, downsample_by)
        masks = downsample(masks, downsample_by, is_mask=True)
        
    cwt_features = generate_cwt_image_cellstreams(
        image,
        min_scale=min_scale,
        max_scale=max_scale,
        num_filter_banks=num_filter_banks,
        normalize_amplitudes=normalize_amplitudes,
        blocks=blocks,
        use_gpu=use_gpu,
        bank_method=bank_method,
        normalize_histogram=normalize_histogram,
        mean_center=mean_center,
        carrier_channel=carrier_channel,
        channel_names=channel_names,
        channel_outputs=channel_outputs,
        sampling=sampling,
        **ssqueezepy_cwt_kwargs,
    )
    
    # Compile results into a tidy/long-form dataframe using vectorized ops
    dfs = []
    
    for ch_key, features_dict in cwt_features.items():
        for feat_key, feat_tensor in features_dict.items():
            # Extract trajectories
            means, stds = extract_cwt_cellstreams(feat_tensor, masks)
            
            means_np = means.detach().cpu().numpy()
            stds_np = stds.detach().cpu().numpy()
            
            num_cells, num_banks, T_len = means_np.shape
            
            # Grid of coordinates
            cell_ids, bank_indices, frames = np.meshgrid(
                np.arange(num_cells),
                np.arange(num_banks),
                np.arange(T_len),
                indexing='ij'
            )
            
            df_part = pd.DataFrame({
                "cell_id": cell_ids.ravel(),
                "frame": frames.ravel(),
                "channel": ch_key,
                "feature": feat_key,
                "filter_bank": bank_indices.ravel(),
                "mean": means_np.ravel(),
                "std": stds_np.ravel(),
            })
            if image_filename is not None:
                df_part["image_filename"] = image_filename
            if masks_filename is not None:
                df_part["mask_filename"] = masks_filename
                
            dfs.append(df_part)
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def process_folder_cwt_cellstreams(images_directory, masks_directory, **kwargs):
    """
    Batch process all images and masks in a folder using CWT feature extraction.
    """
    images = sorted(os.listdir(images_directory))
    data = []
    
    for image_filename in progressbar.progressbar(images):
        name, ext = os.path.splitext(image_filename)
        ext = ext.lower().lstrip(".")
        
        if ext in ["nd2", "tif", "tiff"]:
            masks_filename = f"{name}_masks.tif"
            image_path = os.path.join(images_directory, image_filename)
            mask_path = os.path.join(masks_directory, masks_filename)
            
            # Check if mask exists, fallback to standard check
            if not os.path.exists(mask_path):
                # Try .tiff or other extension variants
                masks_filename_alt = f"{name}_masks.tiff"
                mask_path_alt = os.path.join(masks_directory, masks_filename_alt)
                if os.path.exists(mask_path_alt):
                    mask_path = mask_path_alt
                    masks_filename = masks_filename_alt
                    
            if not os.path.exists(mask_path):
                print(f"[warn] Mask file not found: {mask_path}. Skipping.")
                continue
                
            print(f"Processing CWT: {image_path} with {mask_path}")
            image = load_image(image_path)
            masks = load_masks(mask_path)
            
            try:
                pos_data_for_image = process_cwt_image_cellstreams(
                    image,
                    masks,
                    image_filename=image_filename,
                    masks_filename=masks_filename,
                    **kwargs,
                )
                if not pos_data_for_image.empty:
                    data.append(pos_data_for_image)
            except Exception as e:
                print(f"Error processing {image_filename}: {e}")
                
    if not data:
        return pd.DataFrame()
        
    return pd.concat(data, ignore_index=True)