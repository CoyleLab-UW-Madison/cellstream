"""
cellstream.cwt.process

High-level Continuous Wavelet Transform (CWT) processing pipelines.
"""

import logging
logger = logging.getLogger(__name__)
import os
import re
import time
from tqdm.auto import tqdm
import torch
import pandas as pd
import numpy as np
import zarr

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.tree import Tree
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

from ..io import load_image, load_masks
from ..utils import downsample, normalize_dims, filter_masks_by_area
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
    dataframe_output_path=None,
    min_area=None,
    **ssqueezepy_cwt_kwargs,
):
    """
    Full CWT-based processing pipeline for a single image stack and mask set.
    Generates a tidy pandas DataFrame containing extracted single-cell trajectories.
    """
    # Support 'min_mask_size' alias for 'min_area'
    if "min_mask_size" in ssqueezepy_cwt_kwargs:
        val = ssqueezepy_cwt_kwargs.pop("min_mask_size")
        if min_area is None:
            min_area = val

    save_full_field = ssqueezepy_cwt_kwargs.pop("save_full_field", False)
    save_raw_timeseries = ssqueezepy_cwt_kwargs.pop("save_raw_timeseries", False)
    save_processed_timeseries = ssqueezepy_cwt_kwargs.pop("save_processed_timeseries", False)
    rich_progress = ssqueezepy_cwt_kwargs.get("rich_progress", None)
            
    image = normalize_dims(image, 1)
    
    if min_area is not None:
        masks = filter_masks_by_area(masks, min_area)

    if channel_outputs is None:
        channel_outputs = {0: ["amp", "freq", "phase"]}
    
    if downsample_by is not None:
        image = downsample(image, downsample_by)
        masks = downsample(masks, downsample_by, is_mask=True)
        
    if save_processed_timeseries:
        ssqueezepy_cwt_kwargs["return_timeseries"] = True

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
    fast_cell_summary = {}
    
    for ch_key, features_dict in cwt_features.items():
        if ch_key == "_attrs" or not isinstance(features_dict, dict):
            continue
        for feat_key, feat_tensor in features_dict.items():
            # Extract trajectories
            means, stds = extract_cwt_cellstreams(feat_tensor, masks)
            
            means_np = means.detach().cpu().numpy()
            stds_np = stds.detach().cpu().numpy()
            
            num_cells, num_banks, T_len = means_np.shape
            
            # --- FAST METADATA EXTRACTION ---
            # Pre-compute temporal means instantly from the tensors for Zarr .attrs
            temp_mean = means_np.mean(axis=-1)
            temp_std = stds_np.mean(axis=-1)
            for c_idx in range(num_cells):
                for b_idx in range(num_banks):
                    key = f"ch{ch_key}_{feat_key}_bank{b_idx}"
                    fast_cell_summary.setdefault(c_idx, {})[f"{key}_mean"] = temp_mean[c_idx, b_idx].item()
                    fast_cell_summary[c_idx][f"{key}_std"] = temp_std[c_idx, b_idx].item()
            
            # --- LONG DATAFRAME BUILD ---
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
        from ..io import write_unified_zarr, _sanitize_metadata

        if crop_output_path is None:
            base = os.path.splitext(image_filename)[0] if image_filename else "image"
            crop_output_path = f"{base}_cwt_crops.zarr"

        ckw = dict(crop_kwargs or {})
        if rich_progress is not None:
            ckw["rich_progress"] = rich_progress
        # Ensure mask is 2-D for crop_zarr_from_masks
        masks_2d = masks
        if hasattr(masks_2d, 'dim'):
            while masks_2d.dim() > 2:
                masks_2d = masks_2d.max(dim=0).values if hasattr(masks_2d.max(dim=0), 'values') else masks_2d.max(dim=0)[0] if isinstance(masks_2d.max(dim=0), tuple) else masks_2d.max(dim=0)
        elif hasattr(masks_2d, 'ndim'):
            while masks_2d.ndim > 2:
                masks_2d = masks_2d.max(axis=0)
                
        # Restructure features
        parent_key = "timeseries" if "timeseries" in cwt_features else "raw_timeseries"
        features_dict = {
            "cwt": {
                parent_key: {},
                "_attrs": cwt_features.get("_attrs", {})
            }
        }
            
        for k, v in cwt_features.items():
            if k in ["raw_timeseries", "timeseries", "_attrs"]:
                continue
            features_dict["cwt"][parent_key][f"channel_{k}"] = v

        processed_data = cwt_features.get("timeseries", None)

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
        
        # Calculate raw expression means
        if crop_zarrs and crop_root is not None and "cells" in crop_root:
            import scipy.ndimage as ndi
            img_np = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
            masks_2d_np = masks_2d.detach().cpu().numpy() if hasattr(masks_2d, "detach") else np.asarray(masks_2d)
            
            raw_means = {}
            cell_ids = df["cell_id"].unique() if not df.empty else []
            if len(cell_ids) > 0:
                if img_np.ndim == 4: # T, C, Y, X
                    time_avg = img_np.mean(axis=0) # C, Y, X
                    for c in range(time_avg.shape[0]):
                        means_c = ndi.mean(time_avg[c], labels=masks_2d_np, index=cell_ids)
                        for i, cid in enumerate(cell_ids):
                            raw_means.setdefault(cid, {})[f"raw_ch{c}_mean"] = float(means_c[i])
                elif img_np.ndim == 3: # T, Y, X
                    time_avg = img_np.mean(axis=0) # Y, X
                    means_c = ndi.mean(time_avg, labels=masks_2d_np, index=cell_ids)
                    for i, cid in enumerate(cell_ids):
                        raw_means.setdefault(cid, {})["raw_ch0_mean"] = float(means_c[i])

            # Attach per-cell extracted stats from DataFrame to each cell group
            if len(fast_cell_summary) > 0:
                keys = list(crop_root["cells"].group_keys())
                
                rich_progress = ssqueezepy_cwt_kwargs.get("rich_progress")
                attach_task = None
                if ckw.get("show_progress", False):
                    if rich_progress is not None:
                        attach_task = rich_progress.add_task("[bold green]Attaching CWT metadata...", total=len(keys))
                        iterable = keys
                    elif RICH_AVAILABLE:
                        from rich.progress import track
                        iterable = track(keys, description="[bold green]Attaching CWT metadata...", console=console)
                    else:
                        from tqdm.auto import tqdm
                        iterable = tqdm(keys, desc="Attaching CWT metadata", leave=False)
                else:
                    iterable = keys
                
                for cell_key in iterable:
                    if rich_progress is not None and attach_task is not None:
                        rich_progress.advance(attach_task)
                    cell_group = crop_root["cells"][cell_key]
                    label_id = cell_group.attrs.get("label_id", None)
                    if label_id is not None and label_id in fast_cell_summary:
                        summary = fast_cell_summary[label_id].copy()
                            
                        # Add raw expression means
                        if label_id in raw_means:
                            summary.update(raw_means[label_id])
                            
                        try:
                            # Batch update the Zarr attributes in a single disk IO write
                            cell_group.attrs.update({
                                f"extracted_{k}": v for k, v in _sanitize_metadata(summary).items()
                            })
                        except Exception as e:
                            logger.warning(f"Could not attach attributes to {cell_key}: {e}")

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

    # --- Return ---
    if crop_zarrs:
        return (df, crop_root)
    return df

def process_folder_cwt_cellstreams(images_directory, masks_directory, dataframe_output_path=None, min_area=None, **kwargs):
    """
    Batch process all images and masks in a folder using CWT feature extraction.
    """
    # Support 'min_mask_size' alias for 'min_area'
    if "min_mask_size" in kwargs:
        val = kwargs.pop("min_mask_size")
        if min_area is None:
            min_area = val
            
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
                task = progress.add_task("[bold green]Processing CWT folders...", total=len(images))
                iterator = images
                kwargs["rich_progress"] = progress
            else:
                from tqdm.auto import tqdm
                iterator = tqdm(images, desc="Processing CWT folders")
        else:
            iterator = images
            
        for image_filename in iterator:
            if progress_cb is not None:
                progress_cb()
            if RICH_AVAILABLE and progress_cb is None and show_progress:
                progress.update(task, description=f"[bold green]Processing CWT: {image_filename}")
            
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
                
                img_kwargs = kwargs.copy()
                if img_kwargs.get("crop_zarrs") and "crop_output_path" not in img_kwargs:
                    crop_dir = img_kwargs.pop("crop_output_dir", images_directory)
                    img_kwargs["crop_output_path"] = os.path.join(crop_dir, f"{name}_cwt_crops.zarr")
                else:
                    img_kwargs.pop("crop_output_dir", None)
                    
                try:
                    pos_data_for_image = process_cwt_image_cellstreams(
                        image,
                        masks,
                        image_filename=image_filename,
                        masks_filename=masks_filename,
                        min_area=min_area,
                        **img_kwargs,
                    )
                    df_part = pos_data_for_image[0] if isinstance(pos_data_for_image, tuple) else pos_data_for_image
                    if not df_part.empty:
                        data.append(df_part)
                except Exception as e:
                    logger.error(f"Error processing {image_filename}: {e}")
                    
                if RICH_AVAILABLE and progress_cb is None and show_progress:
                    progress.advance(task)
                
    if not data:
        df = pd.DataFrame()
    else:
        df = pd.concat(data, ignore_index=True)
        
    if dataframe_output_path is not None:
        ext = os.path.splitext(dataframe_output_path)[1].lower()
        if ext == ".parquet":
            df.to_parquet(dataframe_output_path)
        elif ext in [".csv", ".txt"]:
            df.to_csv(dataframe_output_path, index=False)
        else:
            logger.warning(f"Unsupported extension {ext} for dataframe_output_path. Defaulting to CSV.")
            df.to_csv(f"{dataframe_output_path}.csv", index=False)
            
    return df

def process_cell(
    cell_group: zarr.Group,
    device: str = None,
    force_recompute: bool = False,
    parent_mask: torch.Tensor = None,
    **kwargs
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw_keys = [k for k, v in cell_group.items() if hasattr(v, 'shape') and len(v.shape) >= 3 and k not in ['phase', 'mask', 'thumbnail']]
    
    if not raw_keys:
        top_mask = None
        if 'mask' in cell_group:
            top_mask = torch.from_numpy(cell_group['mask'][:].astype('float32'))
            
        found_subchannels = False
        any_processed = False
        for key in list(cell_group.keys()):
            subgroup = cell_group[key]
            if isinstance(subgroup, zarr.Group):
                processed = process_cell(subgroup, force_recompute=force_recompute, device=device, parent_mask=top_mask, **kwargs)
                if processed:
                    any_processed = True
                found_subchannels = True
                
        if not found_subchannels:
            logger.info(f"Skipping {cell_group.name}: No raw timeseries found.")
        return any_processed

    cwt_group = cell_group.require_group('cwt')
    
    if not force_recompute and len(cwt_group.keys()) > 0:
        return False

    mask = None
    if 'mask' in cell_group:
        mask = torch.from_numpy(cell_group['mask'][:].astype('float32'))
    elif parent_mask is not None:
        mask = parent_mask
        
    if mask is not None:
        mask = mask.squeeze()

    rich_progress = kwargs.pop('rich_progress', None)
    rich_cell_name = kwargs.pop('rich_cell_name', cell_group.name)
    image_name = kwargs.pop('image_name', "")
    if rich_progress is not None:
        kwargs['rich_progress'] = rich_progress
        kwargs['rich_cell_name'] = rich_cell_name
        
    processed_any = False
    
    cell_id_match = re.search(r'cell_(\d+)', cell_group.name)
    label_id = int(cell_id_match.group(1)) if cell_id_match else cell_group.name.split('/')[-1]
    
    df_rows = []
    
    for rk in raw_keys:
        raw_arr = torch.from_numpy(cell_group[rk][:].astype('float32'))
        if raw_arr.ndim == 3:
            raw_arr = raw_arr.unsqueeze(1)
            
        features = generate_cwt_image_cellstreams(raw_arr, use_gpu=(device=='cuda'), **kwargs)
        
        target_group = cwt_group.require_group(rk)
        
        for ch_key, ch_features in features.items():
            if ch_key == '_attrs': continue
            
            ch_group = target_group.require_group(f"channel_{ch_key}")
            
            for feat_name, feat_val in ch_features.items():
                if feat_name in ch_group:
                    del ch_group[feat_name]
                    
                feat_val = feat_val.detach().cpu().numpy() if hasattr(feat_val, 'detach') else np.asarray(feat_val)
                    
                chunks = (1, 1, feat_val.shape[-2], feat_val.shape[-1]) if feat_val.ndim >= 3 else True
                if feat_val.ndim == 3:
                    feat_val = np.expand_dims(feat_val, axis=1) # match CWT axis convention
                    chunks = (1, 1, feat_val.shape[2], feat_val.shape[3])
                    
                ch_group.create_dataset(
                    feat_name, data=feat_val, chunks=chunks,
                    compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
                )
                
                # --- Inline Aggregation ---
                if mask is not None:
                    mask_np = mask.numpy() if hasattr(mask, 'numpy') else np.asarray(mask)
                    valid_pixels = mask_np > 0
                    if valid_pixels.sum() > 0:
                        # feat_val is (T, num_banks, Y, X)
                        spatial_mean = feat_val[..., valid_pixels].mean(axis=-1)
                        spatial_std = feat_val[..., valid_pixels].std(axis=-1)
                        
                        T_len, num_banks = spatial_mean.shape
                        
                        # Add to long-form dataframe
                        for t in range(T_len):
                            for b in range(num_banks):
                                df_rows.append({
                                    "image_filename": image_name,
                                    "cell_id": label_id,
                                    "frame": t,
                                    "channel": ch_key,
                                    "feature": feat_name,
                                    "filter_bank": b,
                                    "mean": float(spatial_mean[t, b]),
                                    "std": float(spatial_std[t, b])
                                })
                        
                        # Quick temporal mean for .attrs
                        temp_mean = spatial_mean.mean(axis=0)
                        temp_std = spatial_std.mean(axis=0)
                        
                        attrs_update = {}
                        for b in range(num_banks):
                            key = f"ch{ch_key}_{feat_name}_bank{b}"
                            attrs_update[f"{key}_mean"] = float(temp_mean[b])
                            attrs_update[f"{key}_std"] = float(temp_std[b])
                        
                        cwt_stats = dict(cell_group.attrs.get('cwt_stats', {}))
                        cwt_stats.update(attrs_update)
                        cell_group.attrs.update({'cwt_stats': cwt_stats})
            
        if '_attrs' in features:
            target_group.attrs.update({'cwt_processing': features['_attrs']})
            
        processed_any = True

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return processed_any, df_rows

def process_zarr_store(zarr_path: str, force: bool = False, process_full_field: bool = False, **kwargs):
    logger.info(f"Opening Zarr store: {zarr_path}")
    store = zarr.open(zarr_path, mode='a')
    
    if process_full_field:
        if "processed" in store:
            full_field_arr = store["processed"]
            parent_key = "processed"
        elif "raw_timeseries" in store:
            full_field_arr = store["raw_timeseries"]
            parent_key = "raw_timeseries"
        else:
            logger.error("No raw_timeseries or processed found for full field processing.")
            return False
            
        logger.info(f"Processing full-field for {zarr_path}")
        image_tensor = torch.from_numpy(np.asarray(full_field_arr[:]).astype("float32"))
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(1)
            
        if "channel_outputs" not in kwargs:
            channel_outputs = {0: ["amp", "freq", "phase", "z_score"]}
            # default to all channels if image has more
            if image_tensor.shape[1] > 1:
                for i in range(1, image_tensor.shape[1]):
                    channel_outputs[i] = ["amp", "phase", "z_score", "phase_difference"]
            kwargs["channel_outputs"] = channel_outputs
            
        use_gpu = kwargs.get("use_gpu", kwargs.get("device", 'cuda' if torch.cuda.is_available() else 'cpu') == 'cuda')
        kwargs_no_device = {k: v for k, v in kwargs.items() if k not in [
            "device", "use_gpu", "process_full_field", "crop_zarrs", 
            "images", "masks", "output", "input", "crop_output_dir", "dataframe_output_path",
            "min_area", "min_mask_size"
        ]}
        
        import contextlib
        if RICH_AVAILABLE:
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
            if RICH_AVAILABLE:
                kwargs_no_device['rich_progress'] = progress
            cwt_features = generate_cwt_image_cellstreams(image_tensor, use_gpu=use_gpu, **kwargs_no_device)
        
            if RICH_AVAILABLE:
                write_task = progress.add_task("[bold blue]Writing full-field CWT to Zarr...", total=None)
                
            cwt_group = store.require_group("cwt")
            features_group = cwt_group.require_group(parent_key)
            
            if "_attrs" in cwt_features:
                features_group.attrs.update(cwt_features["_attrs"])
                
            for ch_key, ch_features in cwt_features.items():
                if ch_key in ["raw_timeseries", "timeseries", "_attrs"]:
                    continue
                ch_group = features_group.require_group(f"ch_{ch_key}")
                for feat_name, feat_val in ch_features.items():
                    feat_val_np = feat_val.detach().cpu().numpy() if hasattr(feat_val, 'detach') else np.asarray(feat_val)
                    chunks = (1, 1, feat_val_np.shape[-2], feat_val_np.shape[-1]) if feat_val_np.ndim >= 3 else True
                    if feat_val_np.ndim == 3:
                        feat_val_np = np.expand_dims(feat_val_np, axis=1) # match CWT axis convention
                        chunks = (1, 1, feat_val_np.shape[2], feat_val_np.shape[3])
                    
                    if feat_name in ch_group:
                        del ch_group[feat_name]
                    ch_group.create_dataset(
                        feat_name, data=feat_val_np, chunks=chunks,
                        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
                    )
            
            if RICH_AVAILABLE:
                progress.update(write_task, total=1, completed=1, description="[bold green]Wrote full-field CWT to Zarr")

                
        if kwargs.get("crop_zarrs", False):
            if "masks" in store:
                masks = store["masks"][:]
                from ..spatial.crop import crop_zarr_from_masks
                
                features_for_crop = {
                    f"ch_{k}": v for k, v in cwt_features.items() if k not in ["raw_timeseries", "timeseries", "_attrs"]
                }
                if "_attrs" in cwt_features:
                    features_for_crop["_attrs"] = cwt_features["_attrs"]
                    
                crop_dict = {
                    "cwt": {
                        parent_key: features_for_crop
                    }
                }
                ckw = kwargs.get("crop_kwargs", {})
                for k in ["min_mask_size", "padding_fraction", "min_padding_px", "min_area"]:
                    if k in kwargs:
                        ckw[k if k != "min_area" else "min_mask_size"] = kwargs[k]
                
                crop_zarr_from_masks(
                    features=crop_dict,
                    label_image=masks,
                    output_path=store.require_group("cells"),
                    **ckw
                )
                
                # --- Fast Metadata Extraction ---
                try:
                    from .utils import extract_cwt_cellstreams
                    masks_tensor = torch.from_numpy(masks) if not isinstance(masks, torch.Tensor) else masks
                    if hasattr(masks_tensor, 'dim'):
                        while masks_tensor.dim() > 2:
                            masks_tensor = masks_tensor.max(dim=0).values if hasattr(masks_tensor.max(dim=0), 'values') else masks_tensor.max(dim=0)[0] if isinstance(masks_tensor.max(dim=0), tuple) else masks_tensor.max(dim=0)
                    elif hasattr(masks_tensor, 'ndim'):
                        while masks_tensor.ndim > 2:
                            masks_tensor = masks_tensor.max(axis=0)

                    fast_cell_summary = {}
                    for ch_key, features_dict in cwt_features.items():
                        if ch_key in ["_attrs", "raw_timeseries", "timeseries"]: continue
                        for feat_key, feat_tensor in features_dict.items():
                            means, stds = extract_cwt_cellstreams(feat_tensor, masks_tensor)
                            means_np = means.detach().cpu().numpy()
                            stds_np = stds.detach().cpu().numpy()
                            num_cells, num_banks, T_len = means_np.shape
                            temp_mean = means_np.mean(axis=-1)
                            temp_std = stds_np.mean(axis=-1)
                            for c_idx in range(num_cells):
                                for b_idx in range(num_banks):
                                    key = f"ch{ch_key}_{feat_key}_bank{b_idx}"
                                    fast_cell_summary.setdefault(c_idx, {})[f"{key}_mean"] = temp_mean[c_idx, b_idx].item()
                                    fast_cell_summary[c_idx][f"{key}_std"] = temp_std[c_idx, b_idx].item()
                                    
                    cells_group = store.require_group("cells")
                    from ..io import _sanitize_metadata
                    for cell_key in cells_group.keys():
                        cell_group = cells_group[cell_key]
                        label_id = cell_group.attrs.get("label_id", None)
                        if label_id is not None and label_id in fast_cell_summary:
                            cell_group.attrs.update({
                                f"extracted_{k}": v for k, v in _sanitize_metadata(fast_cell_summary[label_id]).items()
                            })
                except Exception as e:
                    logger.warning(f"Failed to attach fast cell stats: {e}")
            else:
                logger.warning("No masks found, cannot crop.")
                
        return True
    
    if 'cells' in store:
        cells_group = store['cells']
        cell_ids = list(cells_group.keys())
    else:
        cell_ids = [k for k in store.keys() if k.startswith('cell_')]
        if not cell_ids:
            logger.error("Store does not contain a 'cells' group or any 'cell_' root groups.")
            return
        cells_group = store

    if RICH_AVAILABLE:
        tree = Tree(f"[bold blue]{zarr_path}[/bold blue]")
        num_cells = len(cell_ids)
        num_subchannels = 0
        
        for idx, cell_id in enumerate(cell_ids):
            if idx < 5:
                cell_node = tree.add(f"[cyan]{cell_id}[/cyan]")
            
            if isinstance(cells_group[cell_id], zarr.Group):
                raw_keys = [k for k, v in cells_group[cell_id].items() if hasattr(v, 'shape') and len(v.shape) >= 3 and k not in ['phase', 'mask', 'thumbnail']]
                if raw_keys:
                    if idx < 5:
                        for rk in raw_keys:
                            cell_node.add(f"[blue]{rk}[/blue] [green](Timeseries found)[/green]")
                    num_subchannels += len(raw_keys)
                        
        if num_cells > 5:
            tree.add(f"... and {num_cells - 5} more cells")
            
        tree.add(f"Found [bold cyan]{num_cells}[/bold cyan] cells, [bold cyan]{num_subchannels}[/bold cyan] channels ready for processing.")
        console.print(tree)
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        )
        
        stats = {'processed': 0, 'skipped': 0, 'errors': 0}
        start_time = time.time()
        
        all_df_rows = []
        image_name = os.path.basename(zarr_path).replace("_crops.zarr", "")
        
        with progress:
            main_task = progress.add_task("[bold green]Processing CWT features...", total=len(cell_ids))
            sub_task = progress.add_task("[cyan]Computing CWT...", total=100)
            kwargs['rich_progress'] = progress
            kwargs['rich_sub_task'] = sub_task
            kwargs['image_name'] = image_name
            
            for cell_id in cell_ids:
                try:
                    kwargs['rich_cell_name'] = cell_id
                    res = process_cell(cells_group[cell_id], force_recompute=force, **kwargs)
                    if isinstance(res, tuple):
                        processed, rows = res
                    else:
                        processed, rows = res, []
                        
                    if processed:
                        stats['processed'] += 1
                        all_df_rows.extend(rows)
                    else:
                        stats['skipped'] += 1
                except Exception as e:
                    stats['errors'] += 1
                    console.print(f"[red][ ERROR ][/red] {cell_id}: {e}")
                progress.advance(main_task)
                
        elapsed = time.time() - start_time
        table = Table(title="CWT Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Total Cells", str(num_cells))
        table.add_row("Processed", str(stats['processed']))
        table.add_row("Skipped", str(stats['skipped']))
        table.add_row("Errors", str(stats['errors']))
        table.add_row("Time Taken", f"{elapsed:.1f}s")
        console.print(table)
        
        if all_df_rows:
            df = pd.DataFrame(all_df_rows)
            out_pq = zarr_path.replace('.zarr', '_cwt_features.parquet')
            df.to_parquet(out_pq)
            logger.info(f"Saved aggregated dataframe to {out_pq}")
    else:
        all_df_rows = []
        image_name = os.path.basename(zarr_path).replace("_crops.zarr", "")
        kwargs['image_name'] = image_name
        
        for cell_id in tqdm(cell_ids, desc="Processing CWT features", position=0, leave=True):
            try:
                res = process_cell(cells_group[cell_id], force_recompute=force, **kwargs)
                if isinstance(res, tuple):
                    all_df_rows.extend(res[1])
            except Exception as e:
                logger.error(f"Error processing {cell_id}: {e}")
                
        if all_df_rows:
            df = pd.DataFrame(all_df_rows)
            out_pq = zarr_path.replace('.zarr', '_cwt_features.parquet')
            df.to_parquet(out_pq)
            logger.info(f"Saved aggregated dataframe to {out_pq}")
            
    return True
