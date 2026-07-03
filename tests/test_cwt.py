import pytest
from cellstream.cwt.utils import generate_cwt_image_cellstreams
import os
import torch

@pytest.mark.skipif("CI" in os.environ, reason="ssqueezepy might be heavy for basic CI")
def test_generate_cwt_image_cellstreams(synthetic_image):
    # Minimal test for CWT processing
    try:
        import ssqueezepy
    except ImportError:
        pytest.skip("ssqueezepy not installed")

    result = generate_cwt_image_cellstreams(
        synthetic_image, 
        min_scale=4, 
        max_scale=8, 
        num_filter_banks=1,
        blocks=1
    )
    assert 0 in result
    assert "amp" in result[0]
