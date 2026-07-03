import torch
import pytest
from cellstream.analysis import extract_single_cell_data

def test_extract_single_cell_data(synthetic_masks):
    # Require torch-scatter
    try:
        import torch_scatter
    except ImportError:
        pytest.skip("torch_scatter not installed")

    # mock queried features: 2 channels, 8x8 spatial
    features = {
        "queried_amplitude": torch.randn(2, 8, 8),
        "queried_phase": torch.randn(2, 8, 8)
    }
    masks_dict = {"cell": synthetic_masks[0]}

    result = extract_single_cell_data(masks_dict, features)
    assert "cell" in result
    assert result["cell"]["queried_amplitude"].shape[1] == 3 # background (0) + 2 cells
