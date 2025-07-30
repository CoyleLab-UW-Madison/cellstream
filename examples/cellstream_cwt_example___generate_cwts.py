# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 12:41:33 2025

@author: smcoyle
"""

import torch
import cellstream
import napari
import matplotlib.pyplot as plt

timeseries_image=cellstream.image.load_image("images/example_timeseries_mini_0.tif")

results = cellstream.cwt.generate_cwt_image_cellstreams(
    
    ###image file
    timeseries_image,

    ###cwt parameters
    min_scale=80,
    max_scale=180,
    num_filter_banks=20,
    blocks=50,
    use_gpu=True,
    bank_method='max_pool',
    normalize_amplitudes=False,

    ###pre-processing
    #downsample_by=0.25,
    normalize_histogram=True,
 
    ###channel information
    channel_names=['MinE','MinD'],
    carrier_channel=1, 
    channel_outputs={
        0: ['amp','phase_difference'],
        1: ['amp','freq']
        },
    
    ###sampling parameters
    # sampling={
    #     'fs': 2,
    #     'N' : 361
    #     }
    )


###napari visualization
viewer = napari.Viewer()
viewer.add_image(
    results['MinD']['amp'].detach().numpy(), 
    name='my_volume', 
    rendering='mip',
    scale=[20,1,1]
    )  # rendering='mip', 'translucent', etc.


##create 3d rock
outs=[]
for i in torch.linspace(-2.5,2.5,25):
    viewer.camera.angles = (0, i,90)
    out=viewer.screenshot(canvas_only=True)
    outs.append(torch.from_numpy(out))
outs=torch.stack(outs,dim=0)


#extraction protoype
masks=cellstream.image.load_masks("masks/example_timeseries_mini_0_masks.tif")
from torch_scatter import scatter_mean, scatter_std
feature=results['MinD']['amp']
masks_bc=masks.broadcast_to(361,20,250,250)
feature=feature.reshape(361,20,-1)
masks_bc=masks_bc.reshape(361,20,-1)
out=scatter_mean(feature,masks_bc,dim=-1)