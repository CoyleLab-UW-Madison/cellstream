"""
cellstream.cwt.process

High-level Continuous Wavelet Transform (CWT) processing pipelines.
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
    carrier_channel=None,
    channel_names=None,
    channel_outputs=None,
    sampling=None,
    image_filename=None,
    masks_filename=None,
    crop_zarrs=False,
    crop_output_path=None,
    crop_kwargs=None,
    **ssqueezepy_cwt_kwargs,
):
    """
    Full CWT-based processing pipeline for a single image stack and mask set.
    Generates a tidy pandas DataFrame containing extracted single-cell trajectories.
    """
    image = normalize_dims(image, 1)

    if channel_outputs is None:
        channel_outputs = {0: ["amp", "freq", "phase"]}
    
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
        if ch_key == "_attrs":
            continue
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
        df = pd.DataFrame()
    else:
        df = pd.concat(dfs, ignore_index=True)

    # --- Optional cropping ---
    crop_root = None
    if crop_zarrs:
        from ..spatial.crop import crop_zarr_from_masks
        from ..io import _sanitize_metadata

        if crop_output_path is None:
            base = os.path.splitext(image_filename)[0] if image_filename else "image"
            crop_output_path = f"{base}_cwt_crops.zarr"

        ckw = dict(crop_kwargs or {})

        # Ensure mask is 2-D for crop_zarr_from_masks
        masks_2d = masks
        if hasattr(masks_2d, 'dim'):
            while masks_2d.dim() > 2:
                masks_2d = masks_2d[0]
        elif hasattr(masks_2d, 'ndim'):
            while masks_2d.ndim > 2:
                masks_2d = masks_2d[0]

        logger.info(f"Cropping CWT features to per-cell zarr at {crop_output_path}...")
        crop_root = crop_zarr_from_masks(
            cwt_features, masks_2d, crop_output_path, **ckw,
        )

        # Attach per-cell extracted stats from DataFrame to each cell group
        if not df.empty:
            logger.info("Attaching extracted CWT cell data to crop zarr groups...")
            for cell_key in crop_root.group_keys():
                cell_group = crop_root[cell_key]
                label_id = cell_group.attrs.get("label_id", None)
                if label_id is not None and label_id in df["cell_id"].values:
                    cell_rows = df[df["cell_id"] == label_id]
                    # Summarise: for each (channel, feature, filter_bank), store mean and std
                    summary = {}
                    for _, row in cell_rows.iterrows():
                        key = f"ch{row['channel']}_{row['feature']}_bank{row['filter_bank']}"
                        mean_val = row["mean"]
                        std_val = row["std"]
                        if hasattr(mean_val, "item"):
                            mean_val = mean_val.item()
                        if hasattr(std_val, "item"):
                            std_val = std_val.item()
                        summary[f"{key}_mean"] = mean_val
                        summary[f"{key}_std"] = std_val
                    for k, v in _sanitize_metadata(summary).items():
                        try:
                            cell_group.attrs[f"extracted_{k}"] = v
                        except Exception as e:
                            logger.warning(f"Could not attach attr {k} to {cell_key}: {e}")

    # --- Return ---
    if crop_zarrs:
        return (df, crop_root)
    return df

def process_folder_cwt_cellstreams(images_directory, masks_directory, **kwargs):
    """
    Batch process all images and masks in a folder using CWT feature extraction.
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
                
            logger.info(f"Processing CWT: {image_path} with {mask_path}")
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
                df_part = pos_data_for_image[0] if isinstance(pos_data_for_image, tuple) else pos_data_for_image
                if not df_part.empty:
                    data.append(df_part)
            except Exception as e:
                logger.error(f"Error processing {image_filename}: {e}")
                
    if not data:
        return pd.DataFrame()
        
    return pd.concat(data, ignore_index=True)