import pytest
import torch
import numpy as np

@pytest.fixture
def synthetic_image():
    """Returns a synthetic image of shape (T, C, X, Y)."""
    return torch.randn(20, 2, 8, 8)

@pytest.fixture
def synthetic_masks():
    """Returns a synthetic mask of shape (T, X, Y) with 2 unique cells."""
    masks = torch.zeros(20, 8, 8, dtype=torch.int64)
    masks[:, 2:4, 2:4] = 1
    masks[:, 5:7, 5:7] = 2
    return masks
