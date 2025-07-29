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
    fft_features_to_process=['full_amplitude', 'normalized_amplitude', 'z_score', 'phase'],
    **kwargs       
):

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
       return torch.empty(shape, device=device if batch_size else None)
   

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
    Extracts FFT features from peak frequency bins.
    Parameters:
        features_to_query: which FFT features to extract at the peak frequency.
            Options: any subset of: 'full_amplitude', 'normalized_amplitude', 'z_score', 'phase'
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

