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

from ..io import load_image, load_masks
from ..utils import downsample, normalize_dims
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

        # Get the primary mask (the "all" key in the masks_dict)
        primary_mask = masks_dict.get("all", next(iter(masks_dict.values())))

        # Ensure mask is 2-D for crop_zarr_from_masks
        if primary_mask.dim() > 2:
            primary_mask = primary_mask[0]

        logger.info(f"Cropping features to per-cell zarr at {crop_output_path}...")
        crop_root = crop_zarr_from_masks(
            fft_features, primary_mask, crop_output_path, **ckw,
        )

        # Attach per-cell extracted stats from DataFrame to each cell group
        logger.info("Attaching extracted cell data to crop zarr groups...")
        for cell_key in crop_root.group_keys():
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
                for k, v in _sanitize_metadata(extracted).items():
                    try:
                        cell_group.attrs[f"extracted_{k}"] = v
                    except Exception as e:
                        logger.warning(f"Could not attach attr {k} to {cell_key}: {e}")

    # --- Return ---
    out = [df]
    if return_fft_features:
        out.append(fft_features)
    if crop_zarrs:
        out.append(crop_root)
    return out[0] if len(out) == 1 else tuple(out)


def process_folder_cellstreams(images_directory, masks_directory, **kwargs):
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
    for image_filename in tqdm(images):
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

            logger.info(f"Processing FFT: {image_path} with {mask_path}")
            image = load_image(image_path)
            masks = load_masks(mask_path)

            try:
                pos_data_for_image = process_image_cellstreams(
                    image,
                    masks,
                    image_filename=image_filename,
                    masks_filename=masks_filename,
                    **kwargs,
                )
                df_part = pos_data_for_image[0] if isinstance(pos_data_for_image, tuple) else pos_data_for_image
                data.append(df_part)

            except Exception as e:
                logger.error(f"Error processing {image_filename}: {e}")
    if not data:
        return pd.DataFrame()
    data = pd.concat(data, ignore_index=True)
    return data

