# -*- coding: utf-8 -*-
"""
cellstream.fft.utils
FFT Feature Extraction and Analysis Module

@authors: coylelab

This module provides tools to extract per-pixel frequency-domain features from
time-resolved microscopy data using FFT, and to map these features to single-cell
summaries via segmentation masks. The extracted features can include amplitude,
normalized amplitude, z-scored amplitude, and phase, enabling a rich downstream
analysis of cell oscillatory behavior in the frequency domain.

Main Components:
----------------
1. `generate_fft_features`:
    Computes per-pixel FFT features from 4D image stacks (T x C x X x Y).

2. `query_fft_features`:
    Extracts the dominant frequency features per pixel or channel and computes
    phase differences relative to a reference ("carrier") channel.

3. `extract_single_cell_data`:
    Aggregates FFT-derived features at the single-cell level using segmentation masks.
"""


import torch
from torch_scatter import scatter_mean, scatter_std
from ..image.utils import normalize_histogram as norm_hist
import progressbar

def generate_fft_features(
    image, 
    normalize_histogram=True,
    max_bin=None,
    batch_size=None,
    device=None,
    fft_features_to_process=['full_amplitude', 'normalized_amplitude', 'z_score', 'phase'],
    **kwargs       
):

    """
    Compute FFT-based feature maps from a time-resolved image stack.
    
    Parameters:
    -----------
    image : torch.Tensor
        4D tensor of shape (T, C, X, Y) representing the input image stack.
    normalize_histogram : bool
        Whether to histogram-equalize the input image before processing.
    max_bin : int or None
        Maximum number of frequency bins to keep from the FFT.
        If None, keeps all bins.
    batch_size : int or None
        If set, computes FFTs in flattened spatial batches to save memory.
    device : torch.device
        Device to use for batched FFT computation.
    fft_features_to_process : list of str
        List of which FFT features to compute. Options:
        - 'full_amplitude'
        - 'normalized_amplitude'
        - 'z_score'
        - 'phase'
    Returns:
    --------
    feature_map : dict
        Dictionary of FFT features keyed by feature type, each of shape (F, C, X, Y).
    """
    
    T, C, X, Y = image.shape
    F = T // 2 + 1
    
    if max_bin is None:
        max_bin = F
    if normalize_histogram:
        image = norm_hist(image)
     
    mean_image = image.mean(axis=0)
    centered_image = image - mean_image
     
    feature_map = {}
     
    def allocate(shape):
        return torch.empty(shape if batch_size else None)
    
     
    if batch_size is not None:
         centered_image = centered_image.reshape(T, C, X * Y)
         bar = progressbar.ProgressBar(max_value=X * Y)
     
         # Allocate
         buffers = {}
         for f in fft_features_to_process:
             buffers[f] = allocate((max_bin, C, X * Y))
     
         for start in range(0, X * Y, batch_size):
             end = min(start + batch_size, X * Y)
             batch = centered_image[:, :, start:end].to(device)
             fft_chunk = torch.fft.rfft(batch, axis=0)
             amp = fft_chunk.abs()
     
             if 'full_amplitude' in fft_features_to_process:
                 buffers['full_amplitude'][:, :, start:end] = amp[:max_bin].cpu()
     
             if 'normalized_amplitude' in fft_features_to_process:
                 norm_amp = amp / amp.sum(axis=0)
                 buffers['normalized_amplitude'][:, :, start:end] = norm_amp[:max_bin].cpu()
     
             if 'z_score' in fft_features_to_process:
                 z = (amp - amp.mean(dim=0, keepdims=True)) / amp.std(dim=0, keepdims=True)
                 buffers['z_score'][:, :, start:end] = z[:max_bin].cpu()
     
             if 'phase' in fft_features_to_process:
                 phase = fft_chunk.angle()
                 buffers['phase'][:, :, start:end] = phase[:max_bin].cpu()
     
             bar.update(end)
     
         for key in buffers:
             feature_map[key] = buffers[key].reshape(max_bin, C, X, Y)
             
    else:
        
         fft = torch.fft.rfft(centered_image, axis=0)
         amp = fft.abs()
     
         if 'full_amplitude' in fft_features_to_process:
             feature_map['full_amplitude'] = amp[:max_bin]
     
         if 'normalized_amplitude' in fft_features_to_process:
             norm_amp = amp / amp.sum(axis=0)
             feature_map['normalized_amplitude'] = norm_amp[:max_bin]
     
         if 'z_score' in fft_features_to_process:
             z = (amp - amp.mean(dim=0, keepdims=True)) / amp.std(dim=0, keepdims=True)
             feature_map['z_score'] = z[:max_bin]
     
         if 'phase' in fft_features_to_process:
             phase = fft.angle()
             feature_map['phase'] = phase[:max_bin]
     
    return feature_map


def query_fft_features(
    fft_features,
    cutoff_frequency_bin,
    carrier_index,
    sampling=None,
    peak_method='normalized_amplitude',
    fft_features_to_process=['full_amplitude', 'normalized_amplitude', 'z_score', 'phase'],
    **kwargs
):
   
    """
    Identify dominant frequency bins and extract corresponding FFT features.
    
    Parameters:
    -----------
    fft_features : dict
        Dictionary of FFT feature tensors from `generate_fft_features`.
    cutoff_frequency_bin : int
        Index to start searching for peaks to avoid low-frequency bias.
    carrier_index : int
        Reference channel index for computing phase differences.
    sampling : dict or None
        Optional sampling info: {'fs': float, 'N': int}
        Used to convert FFT bins into frequency units.
    peak_method : str
        Which feature to use for locating the dominant frequency bin.
    fft_features_to_process : list of str
        Which features to return at the dominant frequency.
    
    Returns:
    --------
    queried_features : dict
        Contains queried amplitude, z-score, normalized amplitude, and phase difference
        at the peak frequency per pixel, along with the raw peak frequency index and optionally Hz.
    """

    if peak_method not in fft_features_to_process:
        if 'normalized_amplitude' in fft_features_to_process:
            peak_method='normalized_amplitude'
            print(f"{peak_method} not in features; defaulting to normalized amplitude...")
        elif 'z_score' in fft_features_to_process:
            peak_method='z_score'
            print(f"{peak_method} not in features; defaulting to z-score amplitude...")
    
    num_channels = fft_features[peak_method].shape[1]
    maxes, argmaxes = torch.max(fft_features[peak_method][cutoff_frequency_bin:], dim=0)
    
    # Set all non-carrier channels to use carrier's argmax
    query_image = argmaxes.unsqueeze(0).clone().contiguous()
    carrier_argmax = query_image[:, carrier_index, :, :].unsqueeze(1)
    non_carrier_mask = torch.ones(num_channels, dtype=torch.bool)
    non_carrier_mask[carrier_index] = False
    query_image_clone = query_image.clone()
    query_image_clone[:, non_carrier_mask, :, :] = carrier_argmax.expand(-1, non_carrier_mask.sum().item(), -1, -1)
    
    query_image = query_image_clone + cutoff_frequency_bin
    
    queried_features = {}
    
    if 'full_amplitude' in fft_features_to_process and 'full_amplitude' in fft_features:
        queried_features['queried_amplitude'] = torch.gather(fft_features['full_amplitude'], 0, query_image).squeeze(0)
    
    if 'normalized_amplitude' in fft_features_to_process and 'normalized_amplitude' in fft_features:
        queried_features['queried_normalized_amplitude'] = torch.gather(fft_features['normalized_amplitude'], 0, query_image).squeeze(0)
    
    if 'z_score' in fft_features_to_process and 'z_score' in fft_features:
        queried_features['queried_z_score'] = torch.gather(fft_features['z_score'], 0, query_image).squeeze(0)
    
    if 'phase' in fft_features_to_process and 'phase' in fft_features:
        phase_diff = ((fft_features['phase'] - fft_features['phase'][:, carrier_index, :, :].unsqueeze(1)) % (2 * torch.pi)).abs()
        queried_features['queried_phase_difference'] = torch.gather(phase_diff, 0, query_image).squeeze(0)
    
    adjusted_argmaxes = argmaxes + cutoff_frequency_bin
    queried_features['argmaxes'] = adjusted_argmaxes
    
    if sampling is not None:
        fs = sampling['fs']
        N = sampling['N']
        freqs = torch.fft.rfftfreq(N, d=1.0/fs).to(adjusted_argmaxes.device)
        queried_features['frequencies'] = freqs[adjusted_argmaxes]
    
    return queried_features

def extract_single_cell_data(
    masks_dict,  # {'all': mask, 'thresholded': mask_th, ...}
    queried_features, 
    mean_levels_image=None
):
    
    """
    Aggregate per-pixel FFT features into per-cell statistics using segmentation masks.
    
    Parameters:
    -----------
    masks_dict : dict
        Dictionary mapping mask variant names to 2D mask tensors.
        Each mask tensor should have shape (X, Y) with integer label IDs.
    queried_features : dict
        Dictionary of per-pixel FFT features, typically from `query_fft_features`.
    mean_levels_image : torch.Tensor or None
        Optional per-pixel expression image (C, X, Y) to include in output.
    
    Returns:
    --------
    results : dict
        Dictionary keyed by mask name, each value is a dict of per-cell feature statistics.
    """
    
    #Get shape from first feature image in the dictioanry
    C,X,Y = queried_features[list(queried_features.keys())[0]].shape
    
    #Flatten X,Y
    reshaped_features = {
        key: val.reshape(C, X * Y) for key, val in queried_features.items()
    }
    
    if mean_levels_image is not None:
        reshaped_mean_levels_image = mean_levels_image.reshape(C, X * Y)
    
    def compute_stats(mask_flat, dim_size):
        stats = {}
        for key, val in reshaped_features.items():
            stats[key] = scatter_mean(val, mask_flat, dim=-1, dim_size=dim_size)
            stats[f"{key}_sd"] = scatter_std(val, mask_flat, dim=-1, dim_size=dim_size)
        
        if mean_levels_image is not None:
           stats['levels'] = scatter_mean(reshaped_mean_levels_image, mask_flat, dim=-1, dim_size=dim_size)  
        return stats
    
    results = {}
    num_labels = max(int(mask.max().item()) for mask in masks_dict.values()) + 1
    for name, mask in masks_dict.items():
        mask_flat = mask.reshape(X * Y)
        results[name] = compute_stats(mask_flat, num_labels)
    
    return results

