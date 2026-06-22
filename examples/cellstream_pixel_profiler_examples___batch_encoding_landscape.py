# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:54:44 2026

@author: scoyl
"""

import cellstream

pixel_features=cellstream.experimental.pixel_profiler.batch_profile_pixels(
        #experiment params
        "images",
        channel_names={0: 'E', 1: 'D'},
        #filename_label="test",
        
        #FFT params
        min_bin=4,
        max_bin=40,
        max_fft_bin=50,

        #filter params
        filter_method='product',
        c_val=35.0,
        filter_channel=0,
        peak_constraint='exactly_one',
        
        #acceleration
        device='cuda',
        fft_batch_size=2500,

    )

#generate waveform encoding landscape from pixel features
pixel_landscape=cellstream.experimental.pixel_profiler.compute_2d_landscape(
        pixel_features,
        x_col='D',
        y_col='E',
        z_col='F_bin',
        bins=25,
        x_range=None,
        y_range=None,
        min_count=10,
        percentiles=(5.0, 95.0)
    )

cellstream.experimental.pixel_profiler.plot_2d_landscape(
    pixel_landscape,
    cmap='turbo'
    )


