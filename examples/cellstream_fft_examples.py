# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 09:21:07 2025

@author: smcoyle
"""

import torch
import progressbar
import matplotlib.pyplot as plt

import cellstream
import napari

images_dir = 'images'
masks_dir = 'masks'

data=cellstream.fft.process_folder_cellstreams(
    images_dir,
    masks_dir,
    carrier_index=1,
    threshold_cutoffs={'normalized_amplitude': .02},
    channel_names=['minE', 'minD'],
    cutoff_frequency_bin=6,
    peak_method='normalized_amplitude',
    fft_features_to_process=['z_score','normalized_amplitude']
    
    #downsample_by=0.25
)

plt.scatter(
    data['minD_queried_normalized_amplitude_mean___thresh_queried_normalized_amplitude_at_0.02'], 
    data['minE_queried_normalized_amplitude_mean___thresh_queried_normalized_amplitude_at_0.02'], 
    c=data['minD_argmaxes_mean___thresh_queried_normalized_amplitude_at_0.02'], 
    cmap='turbo', 
    vmin=6, 
    vmax=26
)

plt.figure()
plt.scatter(
    data['minD_levels_mean___thresh_queried_normalized_amplitude_at_0.02'], 
    data['minE_levels_mean___thresh_queried_normalized_amplitude_at_0.02'], 
    c=data['minD_argmaxes_mean___thresh_queried_normalized_amplitude_at_0.02'], 
    cmap='turbo', 
    vmin=6, 
    vmax=26
)