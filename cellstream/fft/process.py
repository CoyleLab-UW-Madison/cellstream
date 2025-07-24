# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 14:52:43 2025

@author: smcoyle
"""


import torch
import pandas as pd
import os
import progressbar

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
    Create pandas DataFrame from results dict.
    Will add '_th' or other suffixes based on mask names.
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
        suffix = "" if mask_name == "all" else f"_{mask_name}"
        add_to_df(result, suffix=suffix)

    return pd.DataFrame(df_data)

def reshape_to_longform(df):
    """
    Reshape the wide-form dataframe into tidy/long-form format.
    Assumes feature columns follow pattern: <channel>_<feature>_<stat>[_<mask_type>]
    """
    import re

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
    cutoff_power=None, 
    cutoff_phases=None, 
    return_fft_features=False,
    image_filename=None,
    masks_filename=None,
    downsample_by=None,
    **kwargs
):
    """
    Main processing function for per-cell features.
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
    
    if cutoff_power is not None:
        carrier_amp = queried_fft_features['queried_norm_amplitudes'][carrier_index]
        masks_th = (carrier_amp > cutoff_power).int() * masks.clone()
        masks_dict['thresholded'] = masks_th.to(dtype=torch.int64)
    
    print("Extracting single-cell data...")
    results = extract_single_cell_data(
        masks_dict, queried_fft_features, mean_image
    )

    df = create_dataframe(
        results, channel_names, image_filename, masks_filename
    )

    return (df, fft_features) if return_fft_features else df

def process_folder_cellstreams(
        images_directory,
        masks_directory,
        **kwargs
    ):

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