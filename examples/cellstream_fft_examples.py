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
    cutoff_power=0.02,
    channel_names=['minE', 'minD'],
    cutoff_frequency_bin=6,
    #downsample_by=0.25
)

plt.scatter(
    data['minD_levels_mean_thresholded'], 
    data['minE_levels_mean_thresholded'], 
    c=data['minD_argmaxes_mean_thresholded'], 
    cmap='turbo', 
    vmin=6, 
    vmax=26
)

plt.figure()
plt.scatter(
    data['minD_queried_norm_amplitudes_mean_thresholded'], 
    data['minE_queried_norm_amplitudes_mean_thresholded'], 
    c=data['minD_argmaxes_mean_thresholded'], 
    cmap='turbo', 
    vmin=6, 
    vmax=26
)