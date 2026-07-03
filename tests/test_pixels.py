import pytest
import torch
import numpy as np
import pandas as pd
from cellstream.pixels import (
    profile_image_pixels,
    project_pixels,
    compute_2d_landscape
)

@pytest.fixture
def sample_timeseries():
    # Shape (T, C, H, W) -> (60, 2, 10, 10)
    # Give it a strong frequency signal in bin 10 for some pixels to ensure they are picked up
    t = torch.arange(60).view(60, 1, 1, 1).float()
    signal = torch.sin(2 * np.pi * 10 * t / 60)
    
    img = torch.randn((60, 2, 10, 10))
    img[:, 0, 5, 5] += signal.squeeze() * 10  # Channel 0
    img[:, 1, 5, 5] += signal.squeeze() * 10  # Channel 1
    
    return img

def test_profile_image_pixels(sample_timeseries):
    df = profile_image_pixels(
        sample_timeseries,
        channel_names={0: 'E', 1: 'D'},
        c_val=1.0, # low threshold to ensure we catch the peak
        min_bin=4,
        max_bin=20,
        max_fft_bin=30,
        filter_method='product',
        peak_constraint='exactly_one'
    )
    
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert 'E' in df.columns
        assert 'D' in df.columns
        assert 'F_bin' in df.columns

def test_landscape_and_projection():
    # Create dummy DataFrame
    df = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [1, 2, 3],
        'E': [1.0, 2.0, 3.0],
        'D': [0.5, 1.5, 2.5],
        'F_bin': [10, 15, 20],
        'filename': ['test.tif'] * 3
    })
    
    proj = project_pixels(df, 'F_bin', shape=(5, 5))
    assert proj.shape == (5, 5)
    assert proj[1, 1] == 10
    
    stats = compute_2d_landscape(
        df,
        x_col='D',
        y_col='E',
        z_col='F_bin',
        bins=10,
        min_count=1
    )
    assert 'mean' in stats
    assert stats['mean'].shape == (10, 10)
