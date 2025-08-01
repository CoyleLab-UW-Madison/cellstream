# -*- coding: utf-8 -*-
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

import torch
import pandas as pd
import os
import progressbar
import re
    
from .utils import generate_fft_features
from .utils import query_fft_features
from .utils import extract_single_cell_data
from ..image.utils import downsample
from ..image.loaders import load_image
from ..image.loaders import load_masks

def create_dataframe(
    results, 
    channel_names=None, 
    image_filename=None, 
    masks_filename=None
):
    """
    Convert the dictionary of per-cell results into a structured DataFrame.

    Parameters:
    -----------
    results : dict
        Dictionary mapping mask names to dicts of per-cell channel-level features.
    channel_names : list of str, optional
        Channel names to label columns; otherwise defaults to "Channel i".
    image_filename : str, optional
        Name of the original image file (for tracking).
    masks_filename : str, optional
        Name of the corresponding mask file.

    Returns:
    --------
    df : pandas.DataFrame
        Wide-form table with per-cell stats (mean, sd) for each channel and feature.
    """
    
    # Pick one entry to get number of masks
    first_key = next(iter(results))
    num_cells = results[first_key][next(iter(results[first_key]))].shape[1]

    if channel_names==None:
        first_result=results[first_key]
        first_results_key = next(iter(first_result))
        num_channels = first_result[first_results_key].shape[0]
        channel_names=[]
        for i in range(num_channels):
            channel_names.append(f"Channel {i}")

    df_data = {
        "mask_index": torch.arange(num_cells).detach().cpu().numpy(),
        "image_filename": image_filename,
        "mask_filename": masks_filename,
    }

    def add_to_df(result_dict, suffix=""):
        for key, tensor in result_dict.items():
            is_sd = key.endswith("_sd")
            base = key[:-3] if is_sd else key
            stat_suffix = "_sd" if is_sd else "_mean"
            for ch_idx, ch_name in enumerate(channel_names):
                colname = f"{ch_name}_{base}{stat_suffix}{suffix}"
                df_data[colname] = tensor[ch_idx].detach().cpu().numpy()

    for mask_name, result in results.items():
        suffix = "" if mask_name == "all" else f"___{mask_name}"
        add_to_df(result, suffix=suffix)

    return pd.DataFrame(df_data)

def reshape_to_longform(df):
    
    """
    Reshape wide-form FFT feature DataFrame to tidy/long-form.

    Parameters:
    -----------
    df : pandas.DataFrame
        Wide-form output from `create_dataframe`.

    Returns:
    --------
    long_df : pandas.DataFrame
        Long-form DataFrame with columns:
        ['mask_index', 'image_filename', 'mask_filename',
         'channel', 'feature', 'stat', 'mask_type', 'value']
    """

    id_vars = ["mask_index", "image_filename", "mask_filename"]
    value_vars = [col for col in df.columns if col not in id_vars]

    # Melt into long form
    long_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="measurement", value_name="value")

    # Extract structured fields using regex
    pattern = re.compile(r"(?P<channel>[^_]+)_(?P<feature>[^_]+)_(?P<stat>mean|sd)(?:_(?P<mask_type>.*))?")

    extracted = long_df["measurement"].str.extract(pattern)
    long_df = pd.concat([long_df, extracted], axis=1).drop(columns="measurement")

    # Fill NaNs in mask_type with 'all' (default mask)
    long_df["mask_type"] = long_df["mask_type"].fillna("all")

    return long_df

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
    **kwargs
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
        Raw FFT features dictionary (if `return_fft_features=True`).
    """
    
    T, C, X, Y = image.shape
    
    if channel_names is None:
        channel_names = [f'channel_{i}' for i in range(C)]
    elif len(channel_names) != C:
        raise ValueError(f"Expected {C} channel names, got {len(channel_names)}")
    
    if downsample_by is not None:
        image = downsample(image, downsample_by)
        masks = downsample(masks, downsample_by, is_mask=True)

    mean_image = image.mean(axis=0)
    
    print("Generating FFT features...")
    fft_features = generate_fft_features(image,**kwargs)

    print(f"Querying FFT features using channel {carrier_index} as carrier...")
    queried_fft_features = query_fft_features(fft_features, cutoff_frequency_bin, carrier_index,**kwargs)

    # --- Normalize input mask(s) to a dictionary ---
    if isinstance(masks, dict):
        masks_dict = {k: v.clone() for k, v in masks.items()}
    else:
        masks_dict = {'all': masks.clone()}
    
    if threshold_cutoffs is not None:
        for feature_name, threshold in threshold_cutoffs.items():
            queried_feature_key=f"queried_{feature_name}"
            if queried_feature_key in queried_fft_features.keys():
                feature_vals = queried_fft_features[queried_feature_key][carrier_index]
                mask = (feature_vals > threshold).int() * masks.clone()
                masks_dict[f"thresh_{queried_feature_key}_at_{threshold}"] = mask.to(dtype=torch.int64)
            else:
                print(f"[warn] Feature '{queried_feature_key}' not found in queried_fft_features. Skipping threshold.")

    
    # if cutoff_power is not None:
    #     carrier_amp = queried_fft_features['queried_norm_amplitudes'][carrier_index]
    #     masks_th = (carrier_amp > cutoff_power).int() * masks.clone()
    #     masks_dict['thresholded'] = masks_th.to(dtype=torch.int64)
    
    print("Extracting single-cell data...")
    results = extract_single_cell_data(
        masks_dict, queried_fft_features, mean_image
    )
    
    print("making dataframe...")
    df = create_dataframe(
        results, channel_names, image_filename, masks_filename
    )

    return (df, fft_features) if return_fft_features else df

def process_folder_cellstreams(
        images_directory,
        masks_directory,
        **kwargs
    ):

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
    
    images = os.listdir(images_directory)
    
    # Process the positive images
    data = []
    for image_filename in progressbar.progressbar(images):
        name, ext = image_filename.split('.')
        
        if ext in ['nd2', 'tif']:
            masks_filename = f"{name}_masks.tif"
            image_path = os.path.join(images_directory, image_filename)
            mask_path = os.path.join(masks_directory, masks_filename)
    
        
            print(f"Processing: {image_path} with {mask_path}")
            print("Loading images...")
            image=load_image(image_path)
            masks=load_masks(mask_path)
            
            try: 
                pos_data_for_image = process_image_cellstreams(
                    image,
                    masks,
                    image_filename=image_filename,
                    masks_filename=masks_filename,
                    **kwargs
                )
                data.append(pos_data_for_image)
                
            except Exception as e:
                print(f"Error processing {image}: {e}")
    data = pd.concat(data, ignore_index=True)
    return data