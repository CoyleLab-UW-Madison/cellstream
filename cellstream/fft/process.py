"""
cellstream.fft.process

High-Level FFT-Based Image Processing Pipelines

@authors: coylelab

This module defines the high-level interface for processing time-resolved
microscopy images into per-cell frequency-domain feature summaries using the FFT.

Functions:
- process_image_cellstreams:
    Main entry point for analyzing a single image and its segmentation masks.
    Performs FFT extraction, frequency peak querying, mask thresholding,
    and single-cell aggregation.

- create_dataframe:
    Converts aggregated feature statistics into a structured `pandas.DataFrame`
    for export or modeling.

- reshape_to_longform`:
    Transforms wide-form DataFrame into tidy/long-form format for plotting or
    statistical analysis.

- process_folder_cellstreams`:
    Batch-processes all compatible image/mask pairs in a directory.

"""

import logging
logger = logging.getLogger(__name__)
import os
from tqdm.auto import tqdm
import torch
import pandas as pd

import zarr
import numpy as np
import time
import re
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.tree import Tree
    from rich.console import Console
    from rich.table import Table
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..io import load_image, load_masks, _sanitize_metadata
from ..utils import downsample, normalize_dims, filter_masks_by_area
from ..analysis import extract_single_cell_data, create_dataframe, reshape_to_longform
from .utils import generate_fft_features, query_fft_features


def process_image_cellstreams(
    image,
    masks,
    cutoff_frequency_bin=0,
    carrier_index=0,
    channel_names=None,
    threshold_cutoffs=None,
    return_fft_features=False,
    image_filename=None,
    masks_filename=None,
    downsample_by=None,
    crop_zarrs=False,
    crop_output_path=None,
    crop_kwargs=None,
    dataframe_output_path=None,
    min_area=None,
    **kwargs,
):
    """
    Full FFT-based processing pipeline for a single image and mask set.

    Steps:
    - Optional downsampling
    - FFT feature extraction
    - Peak frequency querying
    - Thresholding to create additional masks
    - Per-cell feature extraction
    - DataFrame generation

    Parameters:
    -----------
    image : torch.Tensor
        Time-resolved input image, shape (T, C, X, Y).
    masks : torch.Tensor or dict
        Either a single mask (X, Y) or a dictionary of masks.
    cutoff_frequency_bin : int
        Ignore frequencies below this bin when locating peaks.
    carrier_index : int
        Reference channel for phase comparisons and peak sharing.
    channel_names : list of str
        Optional names for the image channels.
    threshold_cutoffs : dict, optional
        Mapping of {feature_name: threshold_value} to generate new thresholded masks.
    return_fft_features : bool
        Whether to return the raw FFT feature tensors as well.
    image_filename : str, optional
        Name of source image file (for record keeping).
    masks_filename : str, optional
        Name of masks file.
    downsample_by : float or None
        If set, spatially downsample image and masks by this factor.
    min_area : int, optional
        If set, filter masks smaller than this area.
    kwargs : dict
        Additional arguments passed to `generate_fft_features` and `query_fft_features`.

    Returns:
    --------
    df : pandas.DataFrame
        Table of per-cell FFT-derived features.
    fft_features : dict (optional)
        Raw FFT features dictionary (if ``return_fft_features=True``).
    crop_root : zarr.Group (optional)
        The root zarr group of the cropped store (if ``crop_zarrs=True``).
    """

    image = normalize_dims(image, 1)
    
    if min_area is not None:
        masks = filter_masks_by_area(masks, min_area)

    T, C, X, Y = image.shape

    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(C)]
    elif len(channel_names) != C:
        raise ValueError(f"Expected {C} channel names, got {len(channel_names)}")

    if downsample_by is not None:
        image = downsample(image, downsample_by)
        masks = downsample(masks, downsample_by, is_mask=True)

    mean_image = image.mean(axis=0)

    logger.info("Generating FFT features...")
    fft_features = generate_fft_features(image, **kwargs)

    logger.info(f"Querying FFT features using channel {carrier_index} as carrier...")
    queried_fft_features = query_fft_features(
        fft_features, cutoff_frequency_bin, carrier_index, **kwargs
    )

    # --- Normalize input mask(s) to a dictionary ---
    if isinstance(masks, dict):
        masks_dict = {k: v.clone() for k, v in masks.items()}
    else:
        masks_dict = {"all": masks.clone()}

    if threshold_cutoffs is not None:
        for feature_name, threshold in threshold_cutoffs.items():
            queried_feature_key = f"queried_{feature_name}"
            if queried_feature_key in queried_fft_features.keys():
                feature_vals = queried_fft_features[queried_feature_key][carrier_index]
                mask = (feature_vals > threshold).int() * masks.clone()
                masks_dict[f"thresh_{queried_feature_key}_at_{threshold}"] = mask.to(
                    dtype=torch.int64
                )
            else:
                logger.warning(f" Feature '{queried_feature_key}' not found in queried_fft_features. Skipping threshold.")

        logger.info("Extracting single-cell data...")
    results = extract_single_cell_data(masks_dict, queried_fft_features, mean_image)

    logger.info("making dataframe...")
    df = create_dataframe(results, channel_names, image_filename, masks_filename)

    # --- Optional cropping ---
    crop_root = None
    if crop_zarrs:
        from ..spatial.crop import crop_zarr_from_masks
        from ..io import _sanitize_metadata

        if crop_output_path is None:
            base = os.path.splitext(image_filename)[0] if image_filename else "image"
            crop_output_path = f"{base}_crops.zarr"

        ckw = dict(crop_kwargs or {})
        if "rich_progress" in kwargs:
            ckw["rich_progress"] = kwargs["rich_progress"]

        # Get the primary mask (the "all" key in the masks_dict)
        primary_mask = masks_dict.get("all", next(iter(masks_dict.values())))

        # Ensure mask is 2-D for crop_zarr_from_masks
        if primary_mask.dim() > 2:
            while primary_mask.dim() > 2:
                primary_mask = primary_mask.max(dim=0).values if hasattr(primary_mask.max(dim=0), 'values') else primary_mask.max(dim=0)[0] if isinstance(primary_mask.max(dim=0), tuple) else primary_mask.max(dim=0)

        # Restructure features to match the hierarchy produced by process_cell
        parent_key = "timeseries" if "timeseries" in fft_features else "raw_timeseries"
        structured_features = {
            "raw_timeseries": image,
            "fft": {
                parent_key: {},
                "_attrs": fft_features.get("_attrs", {})
            }
        }
        if "timeseries" in fft_features:
            structured_features["timeseries"] = fft_features["timeseries"]
            
        for k, v in fft_features.items():
            if k in ["raw_timeseries", "timeseries", "_attrs"]:
                continue
            structured_features["fft"][parent_key][f"channel_{k}"] = v

        logger.info(f"Cropping features to per-cell zarr at {crop_output_path}...")
        crop_root = crop_zarr_from_masks(
            structured_features, primary_mask, crop_output_path, **ckw,
        )

        # Attach per-cell extracted stats from DataFrame to each cell group
        keys = list(crop_root.group_keys())
        
        rich_progress = kwargs.get("rich_progress")
        attach_task = None
        if ckw.get("show_progress", False):
            if rich_progress is not None:
                attach_task = rich_progress.add_task("[bold green]Attaching FFT metadata...", total=len(keys))
                iterable = keys
            elif RICH_AVAILABLE:
                from rich.progress import track
                iterable = track(keys, description="[bold green]Attaching FFT metadata...", console=console)
            else:
                from tqdm.auto import tqdm
                iterable = tqdm(keys, desc="Attaching FFT metadata", leave=False)
        else:
            iterable = keys
            
        for cell_key in iterable:
            if rich_progress is not None and attach_task is not None:
                rich_progress.advance(attach_task)
            cell_group = crop_root[cell_key]
            label_id = cell_group.attrs.get("label_id", None)
            if label_id is not None and label_id in df["cell_id"].values:
                row = df[df["cell_id"] == label_id].iloc[0]
                extracted = {}
                for col in row.index:
                    if col in ("cell_id", "image_filename", "mask_filename"):
                        continue
                    val = row[col]
                    # Convert numpy types to Python builtins for zarr attrs
                    if hasattr(val, "item"):
                        val = val.item()
                    extracted[col] = val
                try:
                    # Batch update the Zarr attributes in a single disk IO write
                    cell_group.attrs.update({
                        f"extracted_{k}": v for k, v in _sanitize_metadata(extracted).items()
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
    out = [df]
    if return_fft_features:
        out.append(fft_features)
    if crop_zarrs:
        out.append(crop_root)
    return out[0] if len(out) == 1 else tuple(out)


def process_folder_cellstreams(images_directory, masks_directory, dataframe_output_path=None, min_area=None, **kwargs):
    """
    Batch process all images and masks in a folder using FFT feature extraction.

    Parameters:
    -----------
    images_directory : str
        Path to directory containing input image files (.tif, .nd2).
    masks_directory : str
        Path to directory containing corresponding mask files.

    kwargs : dict
        Passed to `process_image_cellstreams`.

    Returns:
    --------
    all_data : pandas.DataFrame
        Combined DataFrame of all per-cell results from the folder.
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
                task = progress.add_task("[bold green]Processing FFT folders...", total=len(images))
                iterator = images
                kwargs["rich_progress"] = progress
            else:
                from tqdm.auto import tqdm
                iterator = tqdm(images, desc="Processing FFT folders")
        else:
            iterator = images
            
        for image_filename in iterator:
            if progress_cb is not None:
                progress_cb()
            if RICH_AVAILABLE and progress_cb is None and show_progress:
                progress.update(task, description=f"[bold green]Processing FFT: {image_filename}")
            name, ext = os.path.splitext(image_filename)
            ext = ext.lower().lstrip(".")
    
            if ext in ["nd2", "tif", "tiff"]:
                masks_filename = f"{name}_masks.tif"
                image_path = os.path.join(images_directory, image_filename)
                mask_path = os.path.join(masks_directory, masks_filename)
                
                if not os.path.exists(mask_path):
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
                    pos_data_for_image = process_image_cellstreams(
                        image,
                        masks,
                        image_filename=image_filename,
                        masks_filename=masks_filename,
                        min_area=min_area,
                        **kwargs,
                    )
                    df_part = pos_data_for_image[0] if isinstance(pos_data_for_image, tuple) else pos_data_for_image
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
    parent_mask = None,
    **kwargs
):
    import torch
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
                res = process_cell(subgroup, force_recompute=force_recompute, device=device, parent_mask=top_mask, **kwargs)
                if isinstance(res, tuple):
                    processed, _ = res
                else:
                    processed = res
                    
                if processed:
                    any_processed = True
                found_subchannels = True
                
        if not found_subchannels:
            logger.info(f"Skipping {cell_group.name}: No raw timeseries found.")
        return any_processed, []

    fft_group = cell_group.require_group('fft')
    
    if not force_recompute and len(fft_group.keys()) > 0:
        return False, []

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
            
        features = generate_fft_features(raw_arr, device=device, **kwargs)
        
        target_group = fft_group.require_group(rk)
        
        for feat_name, feat_val in features.items():
            if feat_name == '_attrs': continue
            
            if feat_name in target_group:
                del target_group[feat_name]
                
            feat_val = feat_val.detach().cpu().numpy() if hasattr(feat_val, 'detach') else np.asarray(feat_val)
                
            chunks = (1, 1, feat_val.shape[-2], feat_val.shape[-1]) if feat_val.ndim >= 3 else True
            if feat_val.ndim == 3:
                feat_val = np.expand_dims(feat_val, axis=1) # match CWT axis convention
                chunks = (1, 1, feat_val.shape[2], feat_val.shape[3])
                
            target_group.create_dataset(
                feat_name, data=feat_val, chunks=chunks,
                compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
            )
            
            if mask is not None:
                mask_np = mask.numpy() if hasattr(mask, 'numpy') else np.asarray(mask)
                valid_pixels = mask_np > 0
                if valid_pixels.sum() > 0:
                    spatial_mean = feat_val[..., valid_pixels].mean(axis=-1)
                    spatial_std = feat_val[..., valid_pixels].std(axis=-1)
                    
                    df_row = {"image_filename": image_name, "cell_id": label_id}
                    
                    if spatial_mean.ndim == 2:
                        F_len, C_len = spatial_mean.shape
                        for f in range(F_len):
                            for c in range(C_len):
                                key = f"ch{c}_{feat_name}_f{f}"
                                df_row[f"{key}_mean"] = float(spatial_mean[f, c])
                                df_row[f"{key}_std"] = float(spatial_std[f, c])
                                
                        df_rows.append(df_row)
                        
                        freq_mean = spatial_mean.mean(axis=0)
                        freq_std = spatial_std.mean(axis=0)
                        attrs_update = {}
                        for c in range(C_len):
                            key = f"ch{c}_{feat_name}_avgfreq"
                            attrs_update[f"{key}_mean"] = float(freq_mean[c])
                            attrs_update[f"{key}_std"] = float(freq_std[c])
                            
                        fft_stats = dict(cell_group.attrs.get('fft_stats', {}))
                        fft_stats.update(attrs_update)
                        cell_group.attrs.update({'fft_stats': fft_stats})
                        
                    elif spatial_mean.ndim == 1:
                        C_len = spatial_mean.shape[0]
                        attrs_update = {}
                        for c in range(C_len):
                            key = f"ch{c}_{feat_name}"
                            df_row[f"{key}_mean"] = float(spatial_mean[c])
                            df_row[f"{key}_std"] = float(spatial_std[c])
                            attrs_update[f"{key}_mean"] = float(spatial_mean[c])
                            attrs_update[f"{key}_std"] = float(spatial_std[c])
                            
                        df_rows.append(df_row)
                        fft_stats = dict(cell_group.attrs.get('fft_stats', {}))
                        fft_stats.update(attrs_update)
                        cell_group.attrs.update({'fft_stats': fft_stats})
            
        if '_attrs' in features:
            target_group.attrs.update({'fft_processing': features['_attrs']})
            
        processed_any = True

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return processed_any, df_rows

def process_zarr_store(zarr_path: str, force: bool = False, **kwargs):
    logger.info(f"Opening Zarr store: {zarr_path}")
    store = zarr.open(zarr_path, mode='a')
    
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
    else:
        progress = None
        
    all_dfs = []
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    with progress or tqdm(total=len(cell_ids), desc="Processing FFT") as pbar:
        task = progress.add_task("[cyan]Processing FFT features...", total=len(cell_ids)) if progress else None
        
        t0 = time.time()
        for cell_id in cell_ids:
            cell_group = cells_group[cell_id]
            if progress:
                progress.update(task, description=f"[cyan]Processing {cell_id}...")
            
            try:
                processed, dfs = process_cell(
                    cell_group, 
                    force_recompute=force, 
                    rich_progress=progress,
                    rich_cell_name=cell_id,
                    image_name=os.path.basename(zarr_path),
                    **kwargs
                )
                if processed:
                    processed_count += 1
                else:
                    skipped_count += 1
                    
                if dfs:
                    all_dfs.extend(dfs)
                    
            except Exception as e:
                logger.error(f"Error processing {cell_id}: {e}")
                if progress:
                    progress.console.print(f"[red][ ERROR ][/red] {cell_id}: {e}")
                error_count += 1
                
            if progress:
                progress.advance(task)
            else:
                pbar.update(1)
                
        t1 = time.time()

    if all_dfs:
        df = pd.DataFrame(all_dfs)
        df_path = zarr_path.replace('.zarr', '_fft_features.parquet')
        try:
            df.to_parquet(df_path)
            if progress:
                progress.console.print(f"[green]Saved summary to {df_path}[/green]")
        except ImportError:
            df_path = zarr_path.replace('.zarr', '_fft_features.csv')
            df.to_csv(df_path, index=False)
            if progress:
                progress.console.print(f"[yellow]Saved summary to {df_path} (install pyarrow for parquet)[/yellow]")

    if RICH_AVAILABLE:
        table = Table(title="FFT Processing Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Cells", str(len(cell_ids)))
        table.add_row("Processed", str(processed_count))
        table.add_row("Skipped", str(skipped_count))
        table.add_row("Errors", str(error_count))
        table.add_row("Time Taken", f"{t1-t0:.1f}s")
        
        console.print(table)
