# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 12:41:33 2025

@author: smcoyle
"""

import torch
import cellstream
import matplotlib.pyplot as plt

timeseries_image=cellstream.image.load_image("timeseries_for_cwt.tif")

cwt_features = cellstream.cwt.generate_cwt_image_cellstreams(
    
    ###image file
    timeseries_image,

    ###cwt parameters
    min_scale=25,
    max_scale=150,
    num_filter_banks=1,
    blocks=50,
    use_gpu=True,
    bank_method='sort',
    normalize_amplitudes=False,

    ###pre-processing
    #downsample_by=0.25,
    normalize_histogram=True,
 
    ###channel information
    channel_names=['minD','pkc_activity','pka_activity'],
    carrier_channel=0, 
    channel_outputs={
        0: ['amp','freq'],
        1: ['amp','phase_difference'],
        2: ['amp','phase_difference']
        },
    
    ###sampling parameters
    # sampling={
    #     'fs': 2,
    #     'N' : 361
    #     }
    )

#consolidate amplitude features across lines
amp_features=torch.cat(
        [
         cwt_features['minD']['amp'],
         cwt_features['pka_activity']['amp'],
         cwt_features['pkc_activity']['amp']
        ],
        dim=1
    )

phase_features=torch.cat(
        [
         cwt_features['pka_activity']['phase_difference'],
         cwt_features['pkc_activity']['phase_difference']
        ],
        dim=1
    )

#load track-masks
track_masks=cellstream.image.load_masks("timeseries_masks_for_cwt.tif")

#extract single-cell trajectories
amp_signaling_cellstreams,_=cellstream.cwt.extract_cwt_cellstreams(amp_features, track_masks)
phase_signaling_cellstreams,_=cellstream.cwt.extract_cwt_cellstreams(phase_features, track_masks)



#visualize example cell PKA and PKC activity
plt.plot(amp_signaling_cellstreams[6][1])
plt.plot(amp_signaling_cellstreams[6][2])

plt.figure()
#visualize example cell PKA and PKC activity
plt.plot(phase_signaling_cellstreams[6][0])
plt.plot(phase_signaling_cellstreams[6][1])


