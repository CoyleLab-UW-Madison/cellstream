import torch
from cellstream.fft.utils import generate_fft_features, query_fft_features

def test_generate_fft_features(synthetic_image):
    features = generate_fft_features(synthetic_image)
    assert "full_amplitude" in features
    assert features["full_amplitude"].shape == (11, 2, 8, 8) # (T//2 + 1) = 11

def test_query_fft_features(synthetic_image):
    features = generate_fft_features(synthetic_image)
    queried = query_fft_features(features, cutoff_frequency_bin=1, carrier_index=0)
    assert "queried_amplitude" in queried
    assert queried["queried_amplitude"].shape == (2, 8, 8)
