"""
Created on Wed Jul 23 12:41:33 2025

@author: smcoyle
"""

import cellstream

timeseries_image = cellstream.image.load_image("images/example_timeseries_mini_0.tif")

## perform CWT filter-banking, using 20 banks caross scales 80-180

results = cellstream.cwt.generate_cwt_image_cellstreams(
    ###image file
    timeseries_image,
    ###cwt parameters
    min_scale=80,
    max_scale=180,
    num_filter_banks=1,
    blocks='auto',
    use_gpu=True,
    bank_method="max_pool",
    normalize_amplitudes=False,
    ###pre-processing
    # downsample_by=0.25,
    normalize_histogram=True,
    ###channel information
    channel_names=["MinE", "MinD"],
    carrier_channel=1,
    channel_outputs={0: ["amp", "phase_difference"], 1: ["amp", "freq"]},
    #channel_outputs={0: ["amp"], 1: ["amp"]},
    ##sampling parameters
    sampling={"fs": 2, "N": 361},
)