"""
cellstream.stft.process

High-level Continuous Wavelet Transform (STFT) processing pipelines.
"""

import logging
logger = logging.getLogger(__name__)
import os
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.console import Console
    from rich.tree import Tree
    from rich.table import Table
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
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
    
    save_full_field = torch_stft_kwargs.pop("save_full_field", False)
    save_raw_timeseries = torch_stft_kwargs.pop("save_raw_timeseries", False)
    save_processed_timeseries = torch_stft_kwargs.pop("save_processed_timeseries", False)
    crop_zarrs = torch_stft_kwargs.pop("crop_zarrs", False)
    crop_output_path = torch_stft_kwargs.pop("crop_output_path", None)
    crop_kwargs = torch_stft_kwargs.pop("crop_kwargs", None)
    rich_progress = torch_stft_kwargs.pop("rich_progress", None)
    min_area = torch_stft_kwargs.pop("min_area", None)
    dataframe_output_path = torch_stft_kwargs.pop("dataframe_output_path", None)
    
    if min_area is not None:
        from ..utils import filter_masks_by_area
        masks = filter_masks_by_area(masks, min_area)

    if save_processed_timeseries:
        torch_stft_kwargs["return_timeseries"] = True

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
        if ch_key == "_attrs" or not isinstance(features_dict, dict):
            continue
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
        df = pd.DataFrame()
    else:
        df = pd.concat(dfs, ignore_index=True)
        
    # --- Optional Saving / Cropping ---
    crop_root = None
    if crop_zarrs or save_full_field or save_raw_timeseries or save_processed_timeseries:
        from ..io import write_unified_zarr

        if crop_output_path is None:
            base = os.path.splitext(image_filename)[0] if image_filename else "image"
            crop_output_path = f"{base}_stft_crops.zarr"

        ckw = dict(crop_kwargs or {})
        if rich_progress is not None:
            ckw["rich_progress"] = rich_progress
            
        masks_2d = masks
        if hasattr(masks_2d, 'dim'):
            while masks_2d.dim() > 2:
                masks_2d = masks_2d.max(dim=0).values if hasattr(masks_2d.max(dim=0), 'values') else masks_2d.max(dim=0)[0] if isinstance(masks_2d.max(dim=0), tuple) else masks_2d.max(dim=0)
        elif hasattr(masks_2d, 'ndim'):
            while masks_2d.ndim > 2:
                masks_2d = masks_2d.max(axis=0)

        # Restructure features
        parent_key = "timeseries" if "timeseries" in stft_features else "raw_timeseries"
        features_dict = {
            "stft": {
                parent_key: {},
                "_attrs": stft_features.get("_attrs", {})
            }
        }
            
        for k, v in stft_features.items():
            if k in ["raw_timeseries", "timeseries", "_attrs"]:
                continue
            features_dict["stft"][parent_key][f"channel_{k}"] = v

        processed_data = stft_features.get("timeseries", None)

        logger.info(f"Writing unified Zarr to {crop_output_path}...")
        crop_root = write_unified_zarr(
            output_path=crop_output_path,
            raw_data=image,
            processed_data=processed_data,
            features_dict=features_dict,
            masks=masks_2d,
            save_full_field=save_full_field,
            save_raw_timeseries=save_raw_timeseries,
            save_processed_timeseries=save_processed_timeseries,
            crop_zarrs=crop_zarrs,
            crop_kwargs=ckw,
        )
        
    # --- Save DataFrame ---
    if dataframe_output_path is not None:
        ext = os.path.splitext(dataframe_output_path)[1].lower()
        if ext == ".parquet":
            df.to_parquet(dataframe_output_path)
        elif ext in [".csv", ".txt"]:
            df.to_csv(dataframe_output_path, index=False)
        else:
            logger.warning(f"Unsupported extension {ext} for dataframe_output_path. Defaulting to CSV.")
            df.to_csv(f"{dataframe_output_path}.csv", index=False)
            
    if crop_zarrs:
        return (df, crop_root)
    return df

def process_folder_stft_cellstreams(images_directory, masks_directory, dataframe_output_path=None, **kwargs):
    """
    Batch process all images and masks in a folder using STFT feature extraction.
    """
    images = sorted(os.listdir(images_directory))
    data = []
    progress_cb = kwargs.get("progress_callback")
    show_progress = kwargs.get("show_progress", True)
    
    import contextlib
    if progress_cb is None and show_progress and RICH_AVAILABLE:
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        )
    else:
        progress_ctx = contextlib.nullcontext()
        
    with progress_ctx as progress:
        if progress_cb is not None:
            iterator = images
        elif show_progress:
            if RICH_AVAILABLE:
                task = progress.add_task("[bold green]Processing STFT folders...", total=len(images))
                iterator = images
                kwargs["rich_progress"] = progress
            else:
                from tqdm.auto import tqdm
                iterator = tqdm(images, desc="Processing STFT folders")
        else:
            iterator = images
            
        for image_filename in iterator:
            if progress_cb is not None:
                progress_cb()
            if RICH_AVAILABLE and progress_cb is None and show_progress:
                progress.update(task, description=f"[bold green]Processing STFT: {image_filename}")
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
                    
                if RICH_AVAILABLE and progress_cb is None and show_progress:
                    progress.advance(task)
                
    if not data:
        return pd.DataFrame()
        
    df = pd.concat(data, ignore_index=True)
    if dataframe_output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(dataframe_output_path)), exist_ok=True)
        ext = os.path.splitext(dataframe_output_path)[1].lower()
        if ext == ".parquet":
            df.to_parquet(dataframe_output_path, index=False)
        else:
            df.to_csv(dataframe_output_path, index=False)

    return df
