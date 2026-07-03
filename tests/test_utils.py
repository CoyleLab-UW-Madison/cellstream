import torch
from cellstream.utils import normalize_dims, downsample, corr_along_axis

def test_normalize_dims():
    img_3d = torch.randn(20, 8, 8)
    img_4d = normalize_dims(img_3d, channel_dim=1)
    assert img_4d.dim() == 4
    assert img_4d.shape == (20, 1, 8, 8)

def test_downsample():
    img = torch.randn(20, 2, 8, 8)
    down = downsample(img, scale=0.5)
    assert down.shape == (20, 2, 4, 4)

def test_corr_along_axis():
    series_a = torch.randn(20, 1, 8, 8)
    series_b = torch.randn(20, 1, 8, 8)
    corr = corr_along_axis(series_a, series_b, window=5, norm_histogram=False)
    assert corr.shape == (16, 1, 8, 8) # 20 - 5 + 1
