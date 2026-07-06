"""
cellstream.stft.process

High-level Continuous Wavelet Transform (STFT) processing pipelines.
"""

import logging
logger = logging.getLogger(__name__)
import os
from tqdm.auto import tqdm
import torch
import pandas as pd
import numpy as np

from ..io import load_image, load_masks
from ..utils import downsample, normalize_dims
from .utils import generate_stft_image_cellstreams, extract_stft_cellstreams

def process_stft_image_cellstreams(
    image,
    masks,
    min_bin=0,
    max_bin=100,
    num_filter_banks=1,
    normalize_amplitudes=False,
    blocks=10,
    use_gpu=False,
    bank_method="max_pool",
    downsample_by=None,
    normalize_histogram=True,
    mean_center=True,
    carrier_channel=None,
    channel_names=None,
    channel_outputs=None,
    sampling=None,
    image_filename=None,
    masks_filename=None,
    **torch_stft_kwargs,
):
    """
    Full STFT-based processing pipeline for a single image stack and mask set.
    Generates a tidy pandas DataFrame containing extracted single-cell trajectories.
    """
    image = normalize_dims(image, 1)

    if channel_outputs is None:
        channel_outputs = {0: ["amp", "freq", "phase"]}
    
    if downsample_by is not None:
        image = downsample(image, downsample_by)
        masks = downsample(masks, downsample_by, is_mask=True)
        
    stft_features = generate_stft_image_cellstreams(
        image,
        min_bin=min_bin,
        max_bin=max_bin,
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
        **torch_stft_kwargs,
    )
    
    # Compile results into a tidy/long-form dataframe using vectorized ops
    dfs = []
    
    for ch_key, features_dict in stft_features.items():
        for feat_key, feat_tensor in features_dict.items():
            # Extract trajectories
            means, stds = extract_stft_cellstreams(feat_tensor, masks)
            
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

def process_folder_stft_cellstreams(images_directory, masks_directory, **kwargs):
    """
    Batch process all images and masks in a folder using STFT feature extraction.
    """
    images = sorted(os.listdir(images_directory))
    data = []
    
    for image_filename in tqdm(images):
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
                logger.warning(f" Mask file not found: {mask_path}. Skipping.")
                continue
                
            logger.info(f"Processing STFT: {image_path} with {mask_path}")
            image = load_image(image_path)
            masks = load_masks(mask_path)
            
            try:
                pos_data_for_image = process_stft_image_cellstreams(
                    image,
                    masks,
                    image_filename=image_filename,
                    masks_filename=masks_filename,
                    **kwargs,
                )
                if not pos_data_for_image.empty:
                    data.append(pos_data_for_image)
            except Exception as e:
                logger.error(f"Error processing {image_filename}: {e}")
                
    if not data:
        return pd.DataFrame()
        
    return pd.concat(data, ignore_index=True)