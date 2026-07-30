"""
Created on Wed Jul 23 12:41:33 2025

@author: smcoyle
"""

import matplotlib.pyplot as plt
import torch

import cellstream

timeseries_image = cellstream.image.load_image("../cwt/timeseries_for_cwt.tif")

features = cellstream.stft.generate_stft_image_cellstreams(
    ###image file
    timeseries_image,
    ###stft parameters
    min_bin=25,
    max_bin=110,
    n_fft=128,
    win_length=32,
    window=torch.hann_window(32),
    num_filter_banks=1,
    blocks='auto',
    use_gpu=True,
    bank_method="max_pool",
    normalize_histogram=True,
    mean_center=True, 
    ###channel information
    channel_names=["minD", "pkc_activity", "pka_activity"],
    carrier_channel=0,
    channel_outputs={
        0: ["amp","z_score","normalized_amp", "freq"],
        1: ["amp","z_score","normalized_amp", "phase_difference"],
        2: ["amp","z_score","normalized_amp", "phase_difference"],
    },
    ##sampling parameters
    sampling={"fs": 2, "N": 361}
)


# consolidate normalized amplitude features across lines
amp_features = torch.cat(
    [
        features["minD"]["normalized_amp"],
        features["pka_activity"]["normalized_amp"],
        features["pkc_activity"]["normalized_amp"],
    ],
    dim=1,
)

phase_features = torch.cat(
    [
        features["pka_activity"]["phase_difference"],
        features["pkc_activity"]["phase_difference"],
    ],
    dim=1,
)

# load track-masks
track_masks = cellstream.image.load_masks("../cwt/timeseries_masks_for_cwt.tif")

# extract single-cell trajectories
amp_signaling_cellstreams, _ = cellstream.stft.extract_stft_cellstreams(
    amp_features, track_masks
)
phase_signaling_cellstreams, _ = cellstream.stft.extract_stft_cellstreams(
    phase_features, track_masks
)


# visualize example cell (#6) PKA and PKC activity
plt.plot(amp_signaling_cellstreams[6][1], label="PKA (Norm Amp)")
plt.plot(amp_signaling_cellstreams[6][2], label="PKC (Norm Amp)")
plt.legend()
plt.show()
