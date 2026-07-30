import os
import logging
import contextlib
import torch
import pandas as pd
from tqdm.auto import tqdm

try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.console import Console
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..io import load_image, load_masks, write_unified_zarr, TorchZarrStore
from ..utils import downsample, normalize_histogram, normalize_dims
from ..registration import register_and_transform_image_timeseries

logger = logging.getLogger(__name__)

def preprocess_image(image, downsample_by=None, norm_histogram=False, register=False, reg_channel=0, mean_center=False):
    """
    Applies common preprocessing steps to an image tensor.
    """
    image = normalize_dims(image, 1)
    
    if downsample_by is not None:
        image = downsample(image, downsample_by)
        
    if norm_histogram:
        image = normalize_histogram(image)
        
    if register:
        image = register_and_transform_image_timeseries(image, reg_channel=reg_channel)
        
    if mean_center:
        image = image - image.mean(dim=0, keepdim=True)
        
    return image

def process_folder_preprocess(
    images_directory, 
    masks_directory, 
    downsample_by=None, 
    normalize_histogram_flag=False, 
    register=False, 
    reg_channel=0, 
    mean_center=False,
    crop_zarrs=False,
    crop_output_dir=None,
    output=None,
    crop_output_path=None,
    **kwargs
):
    """
    Batch preprocess all images and masks in a folder.
    """
    # handle alias
    if "normalize_histogram" in kwargs:
        normalize_histogram_flag = kwargs.pop("normalize_histogram")

    images = sorted(os.listdir(images_directory))
    data = []
    
    target_dir = crop_output_dir or output or images_directory
    
    import contextlib
    
    valid_images = []
    for image_filename in images:
        name, ext = os.path.splitext(image_filename)
        if ext.lower().lstrip(".") in ["nd2", "tif", "tiff"]:
            valid_images.append(image_filename)
            
    if RICH_AVAILABLE:
        console.print(f"Found {len(valid_images)} images to preprocess.")
        
    for idx, image_filename in enumerate(valid_images):
        name, ext = os.path.splitext(image_filename)
        
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
            
        if RICH_AVAILABLE:
            console.print(f"[bold cyan]──▶ Processing {idx + 1}/{len(valid_images)}:[/bold cyan] {image_filename}")
            status = console.status("[blue]Loading...")
            status.start()
        else:
            status = None
            
        try:
            raw_image = load_image(image_path)
            masks = load_masks(mask_path)
            
            preprocessed_image = normalize_dims(raw_image, 1)
            
            if downsample_by is not None:
                if status:
                    status.update("[blue]Downsampling...")
                preprocessed_image = downsample(preprocessed_image, downsample_by)
                masks = downsample(masks, downsample_by, is_mask=True)
                
            # for cropping the raw_timeseries, we downsample but crop before further processing
            raw_image = preprocessed_image.clone()
                
            if normalize_histogram_flag:
                if status:
                    status.update("[blue]Normalizing histogram...")
                preprocessed_image = normalize_histogram(preprocessed_image)
                
            if register:
                if status:
                    status.update("[blue]Registering...")
                preprocessed_image = register_and_transform_image_timeseries(preprocessed_image, reg_channel=reg_channel)
                
            if mean_center:
                if status:
                    status.update("[blue]Mean centering...")
                preprocessed_image = preprocessed_image - preprocessed_image.mean(dim=0, keepdim=True)
            
            if crop_output_path:
                out_path = crop_output_path
            else:
                out_path = os.path.join(target_dir, f"{name}.zarr")
                
            ckw = kwargs.get("crop_kwargs", {})
            for k in ["min_mask_size", "padding_fraction", "min_padding_px", "min_area"]:
                if k in kwargs:
                    ckw[k if k != "min_area" else "min_mask_size"] = kwargs[k]
                    
            if status:
                status.update("[blue]Writing Zarr...")
                # Stop the status before calling write_unified_zarr so it doesn't fight with the cropping progress bar
                status.stop()
                
            crop_root = write_unified_zarr(
                output_path=out_path,
                raw_data=raw_image,
                processed_data=preprocessed_image,
                masks=masks,
                save_raw_timeseries=True,
                save_processed_timeseries=True,
                crop_zarrs=crop_zarrs,
                crop_kwargs=ckw
            )
        
            if crop_zarrs and crop_root is not None and "cells" in crop_root:
                if status:
                    status.update("[blue]Extracting cell stats...")
                    status.start()
                    
                import scipy.ndimage as ndi
                import numpy as np
                
                img_np = raw_image.detach().cpu().numpy() if hasattr(raw_image, "detach") else np.asarray(raw_image)
                masks_np = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
                
                cells_group = crop_root["cells"]
                cell_ids = []
                for k in cells_group.keys():
                    lbl = cells_group[k].attrs.get("label_id")
                    if lbl is not None:
                        cell_ids.append(lbl)
                        
                if len(cell_ids) > 0:
                    raw_means = {}
                    if img_np.ndim == 4:
                        time_avg = img_np.mean(axis=0)
                        for c in range(time_avg.shape[0]):
                            means_c = ndi.mean(time_avg[c], labels=masks_np, index=cell_ids)
                            for i, cid in enumerate(cell_ids):
                                raw_means.setdefault(cid, {})[f"raw_ch{c}_mean"] = float(means_c[i])
                    elif img_np.ndim == 3:
                        time_avg = img_np.mean(axis=0)
                        means_c = ndi.mean(time_avg, labels=masks_np, index=cell_ids)
                        for i, cid in enumerate(cell_ids):
                            raw_means.setdefault(cid, {})["raw_ch0_mean"] = float(means_c[i])
                            
                    for cell_key in cells_group.keys():
                        cell_group = cells_group[cell_key]
                        label_id = cell_group.attrs.get("label_id")
                        if label_id in raw_means:
                            cell_group.attrs.update(raw_means[label_id])
            
            data.append({"image_filename": image_filename, "status": "preprocessed", "output": out_path})
        finally:
            if status:
                status.stop()
        
    df = pd.DataFrame(data) if data else pd.DataFrame()
    return df

def process_zarr_store(
    zarr_path,
    downsample_by=None, 
    normalize_histogram_flag=False, 
    register=False, 
    reg_channel=0, 
    mean_center=False,
    **kwargs
):
    """
    Run preprocessing on an existing Zarr store containing raw_timeseries.
    Writes preprocessed_timeseries back to the store.
    """
    if "normalize_histogram" in kwargs:
        normalize_histogram_flag = kwargs.pop("normalize_histogram")
        
    import zarr
    logger.info(f"Opening Zarr store for preprocessing: {zarr_path}")
    
    # We will read from TorchZarrStore but write using raw zarr to append
    store = TorchZarrStore(zarr_path)
    if "raw_timeseries" not in store:
        logger.error(f"Store {zarr_path} does not contain 'raw_timeseries'. Cannot preprocess.")
        return False
        
    raw_image = store["raw_timeseries"]
    
    preprocessed_image = preprocess_image(
        raw_image, 
        downsample_by=downsample_by, 
        norm_histogram=normalize_histogram_flag, 
        register=register, 
        reg_channel=reg_channel, 
        mean_center=mean_center
    )
    
    # Write back to the same store
    root = zarr.open_group(zarr_path, mode="a")
    
    # Write processed
    preprocessed_np = preprocessed_image.detach().cpu().numpy()
    
    if "processed" in root:
        del root["processed"]
        
    arr = root.create_dataset(
        name="processed", 
        shape=preprocessed_np.shape, 
        dtype=preprocessed_np.dtype, 
        compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE),
        overwrite=True
    )
    arr[:] = preprocessed_np
    
    return True
