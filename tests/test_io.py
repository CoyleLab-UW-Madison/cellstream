import os
import torch
import pytest
import tempfile
from cellstream.io import write_to_zarr, load_zarr

def test_zarr_io(synthetic_image):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.zarr")
        write_to_zarr(synthetic_image, path)
        store = load_zarr(path)
        assert store is not None
        loaded = store["data"]
        assert torch.allclose(synthetic_image, loaded)
