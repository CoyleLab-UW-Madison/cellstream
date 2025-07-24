# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 14:52:43 2025

@author: smcoyle
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
    **kwargs       
):
    """
    Compute FFT analysis including:
    - Mean centering
    - Optional histogram normalization
    - FFT computation
    - Amplitude, normalized amplitude, and phase
    """
    T, C, X, Y = image.shape
    F = T // 2 + 1
    
    if max_bin is None:
        max_bin = F

    if normalize_histogram:
        image = norm_hist(image)

    mean_image = image.mean(axis=0)
    centered_image = image - mean_image

    if batch_size is not None:
        print(f"Processing in batch mode with block size {batch_size}")
        centered_image = centered_image.reshape(T, C, X * Y)

        # Allocate result arrays
        full_amplitude = torch.empty(max_bin, C, X * Y)
        normalized = torch.empty(max_bin, C, X * Y)
        phase = torch.empty(max_bin, C, X * Y)

        start = 0
        bar = progressbar.ProgressBar(max_value= X * Y)

        while start < X * Y:
            end = min(start + batch_size, X * Y)
            batch = centered_image[:, :, start:end].to(device)

            fft_chunk = torch.fft.rfft(batch, axis=0)
            amp = fft_chunk.abs()
            norm_amp = amp / amp.sum(axis=0)
            ph = fft_chunk.angle()

            full_amplitude[:, :, start:end] = amp[:max_bin].cpu()
            normalized[:, :, start:end] = norm_amp[:max_bin].cpu()
            phase[:, :, start:end] = ph[:max_bin].cpu()

            bar.update(end)
            start = end

        # Reshape back to spatial layout
        full_amplitude = full_amplitude.reshape(max_bin, C, X, Y)
        normalized = normalized.reshape(max_bin, C, X, Y)
        phase = phase.reshape(max_bin, C, X, Y)

    else:
        full = torch.fft.rfft(centered_image, axis=0)
        amp = full.abs()
        norm_amp = amp / amp.sum(axis=0)
        ph = full.angle()

        full_amplitude = amp[:max_bin]
        normalized = norm_amp[:max_bin]
        phase = ph[:max_bin]

    return {
        'full_amplitude': full_amplitude,
        'normalized_amplitude': normalized,
        'phase': phase
    }

def query_fft_features(
        fft_features, 
        cutoff_frequency_bin, 
        carrier_index,
        sampling=None,
        **kwargs
    ):
    
    """
    Process FFT results to extract relevant frequency information
    """
    # Select carrier peak frequencies
    
    num_channels=fft_features['normalized_amplitude'].shape[1]
    
    
    maxes, argmaxes = torch.max(fft_features['normalized_amplitude'][cutoff_frequency_bin:], dim=0)
    
    # Set all non-carrier channels to use carrier's argmax
    query_image = argmaxes.unsqueeze(0).clone().contiguous()
    carrier_argmax = query_image[:, carrier_index, :, :].unsqueeze(1)
    non_carrier_mask = torch.ones(num_channels, dtype=torch.bool)
    non_carrier_mask[carrier_index] = False
    query_image_clone = query_image.clone()
    query_image_clone[:, non_carrier_mask, :, :] = carrier_argmax.expand(-1, non_carrier_mask.sum().item(), -1, -1)
    
    query_image = query_image_clone + cutoff_frequency_bin
    
    # Gather amplitudes using query image indices
    queried_amplitudes = torch.gather(fft_features['full_amplitude'], 0, query_image).squeeze(0)
    queried_norm_amplitudes = torch.gather(fft_features['normalized_amplitude'], 0, query_image).squeeze(0)
    
    # Compute and gather phase-differences using query image indices
    phase_difference = ((fft_features['phase'] - ((fft_features['phase'][:, carrier_index, :, :]).unsqueeze(1))) % (2*torch.pi)).abs()
    queried_phase_differences = torch.gather(phase_difference, 0, query_image).squeeze(0)
    
    adjusted_argmaxes = argmaxes + cutoff_frequency_bin
    if sampling is not None:
        fs = sampling['fs']
        N = sampling['N']
        freqs = torch.fft.rfftfreq(N, d=1.0/fs).to(adjusted_argmaxes.device)
        # Map bin index to frequency
        frequency_estimates = freqs[adjusted_argmaxes]
    else:
        frequency_estimates = None

    queried_fft_features = {
        'queried_amplitudes': queried_amplitudes,
        'queried_norm_amplitudes': queried_norm_amplitudes,
        'queried_phase_differences': queried_phase_differences,
        'argmaxes': adjusted_argmaxes
    }

    if frequency_estimates is not None:
        queried_fft_features['frequencies'] = frequency_estimates

    return queried_fft_features

def extract_single_cell_data(
    masks_dict,  # {'all': mask, 'thresholded': mask_th, ...}
    queried_features, 
    mean_levels_image=None
):
    """
    Extract single-cell features for each mask variant.
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
    for name, mask in masks_dict.items():
        mask_flat = mask.reshape(X * Y)
        num_labels = int(mask_flat.max().item()) + 1
        results[name] = compute_stats(mask_flat, num_labels)

    return results

