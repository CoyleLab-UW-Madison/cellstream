# -*- coding: utf-8 -*-
"""
cellstream.fft.utils
FFT Feature Extraction and Analysis Module

@authors: coylelab

This module provides tools to extract per-pixel frequency-domain features from
time-resolved microscopy data using FFT, and to map these features to single-cell
summaries via segmentation masks. The extracted features can include amplitude,
normalized amplitude, z-scored amplitude, and phase, enabling a rich downstream
analysis of cell oscillatory behavior in the frequency domain.

Main Components:
----------------
1. `generate_fft_features`:
    Computes per-pixel FFT features from 4D image stacks (T x C x X x Y).

2. `query_fft_features`:
    Extracts the dominant frequency features per pixel or channel and computes
    phase differences relative to a reference ("carrier") channel.

3. `extract_single_cell_data`:
    Aggregates FFT-derived features at the single-cell level using segmentation masks.
"""


import torch
import progressbar

def map_data_onto_mask(mask_image, dataframe, property):

    # ensure df has index = cell_id
    if "cell_id" in df.columns:
        df = df.set_index("cell_id")

    # values array aligned to index
    cell_ids = df.index.to_numpy()
    values = df[prop].to_numpy()

    # torch tensors on correct device
    lut = torch.zeros(mask.max().item() + 1, dtype=torch.as_tensor(values).dtype)
    lut[cell_ids] = torch.as_tensor(values)

    return lut[mask]
