# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 12:41:33 2025

@author: smcoyle
"""

import cellstream

timeseries_image=cellstream.image.load_image("images/example_timeseries_mini_0.tif")


## perform CWT filter-banking, using 20 banks caross scales 80-180

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
    
    ##sampling parameters
    sampling={
        'fs': 2,
        'N' : 361
        }
    
    )


###napari visualization
# viewer = napari.Viewer()
# viewer.add_image(
#     results['MinD']['amp'].detach().numpy(), 
#     name='my_volume', 
#     rendering='mip',
#     scale=[20,1,1]
#     )  # rendering='mip', 'translucent', etc.


###create 3d rock
# outs=[]
# for i in torch.linspace(-2.5,2.5,25):
#     viewer.camera.angles = (0, i,90)
#     out=viewer.screenshot(canvas_only=True)
#     outs.append(torch.from_numpy(out))
# outs=torch.stack(outs,dim=0)


