"""
Tests for cellstream.spatial.crop — crop_zarr_from_masks
"""

import pytest
import numpy as np
import torch
import zarr
import shutil
from pathlib import Path

from cellstream.spatial.crop import crop_zarr_from_masks, _compute_padded_bbox


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_label_image():
    """Static 2D label image (X=8, Y=8) with two cells, matching conftest masks spatially."""
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[2:4, 2:4] = 1   # cell 1: 2x2 block
    labels[5:7, 5:7] = 2   # cell 2: 2x2 block
    return labels


@pytest.fixture
def synthetic_fft_features():
    """Mimics output of generate_fft_features: flat dict with (F, C, X, Y) tensors."""
    F, C, X, Y = 11, 2, 8, 8
    return {
        "full_amplitude": torch.randn(F, C, X, Y),
        "normalized_amplitude": torch.randn(F, C, X, Y),
        "z_score": torch.randn(F, C, X, Y),
        "phase": torch.randn(F, C, X, Y),
        "timeseries": torch.randn(20, C, X, Y),
        "_attrs": {
            "normalize_histogram": True,
            "max_bin": 11,
            "batch_size": None,
            "device": None,
            "fft_features_to_process": ["full_amplitude", "normalized_amplitude", "z_score", "phase"],
            "downsample_by": None,
            "return_timeseries": True,
        },
    }


@pytest.fixture
def synthetic_cwt_features():
    """Mimics output of generate_cwt_image_cellstreams: nested dict with channel keys."""
    T, banks, X, Y = 20, 1, 8, 8
    return {
        "channel_0": {
            "amp": torch.randn(T, banks, X, Y),
            "freq": torch.randn(T, banks, X, Y),
            "phase": torch.randn(T, banks, X, Y),
        },
        "channel_1": {
            "amp": torch.randn(T, banks, X, Y),
            "phase_difference": torch.randn(T, banks, X, Y),
        },
        "timeseries": torch.randn(T, 2, X, Y),
        "_attrs": {
            "min_scale": 80,
            "max_scale": 180,
            "num_filter_banks": 1,
            "return_timeseries": True,
        },
    }


@pytest.fixture
def output_zarr_path(tmp_path):
    """Provides a temporary path for output zarr."""
    return str(tmp_path / "test_crop_output.zarr")


# ---------------------------------------------------------------------------
# Tests: _compute_padded_bbox
# ---------------------------------------------------------------------------

class TestComputePaddedBbox:
    def test_basic_padding(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True  # 10x10 cell
        y_min_p, y_max_p, x_min_p, x_max_p, y_min, y_max, x_min, x_max = _compute_padded_bbox(
            mask, padding_fraction=0.1, min_padding_px=1, image_shape=(20, 20)
        )
        # bbox is (5,15,5,15), size 10x10, 10% padding = 1px each side
        assert y_min_p == 4
        assert y_max_p == 16
        assert x_min_p == 4
        assert x_max_p == 16

    def test_min_padding_enforced(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:7, 5:7] = True  # 2x2 cell
        y_min_p, y_max_p, x_min_p, x_max_p, y_min, y_max, x_min, x_max = _compute_padded_bbox(
            mask, padding_fraction=0.1, min_padding_px=3, image_shape=(20, 20)
        )
        # bbox is (5,7,5,7), size 2x2, 10% = 0.2 -> rounds to 1, but min_padding=3
        assert y_min_p == 2
        assert y_max_p == 10
        assert x_min_p == 2
        assert x_max_p == 10

    def test_clamped_to_image_bounds(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:3, 0:3] = True  # cell at corner
        y_min_p, y_max_p, x_min_p, x_max_p, y_min, y_max, x_min, x_max = _compute_padded_bbox(
            mask, padding_fraction=0.5, min_padding_px=2, image_shape=(10, 10)
        )
        assert y_min_p == 0  # clamped
        assert x_min_p == 0  # clamped
        assert y_max_p <= 10
        assert x_max_p <= 10


# ---------------------------------------------------------------------------
# Tests: crop_zarr_from_masks with FFT-style features (flat dict)
# ---------------------------------------------------------------------------

class TestCropFromFftFeatures:
    def test_creates_cell_groups(self, synthetic_fft_features, synthetic_label_image, output_zarr_path):
        root = crop_zarr_from_masks(
            synthetic_fft_features, synthetic_label_image, output_zarr_path
        )
        assert "cell_1" in root
        assert "cell_2" in root

    def test_cell_has_mask_array(self, synthetic_fft_features, synthetic_label_image, output_zarr_path):
        root = crop_zarr_from_masks(
            synthetic_fft_features, synthetic_label_image, output_zarr_path
        )
        assert "mask" in root["cell_1"]
        mask_data = np.asarray(root["cell_1"]["mask"][:])
        assert mask_data.ndim == 2
        assert mask_data.sum() > 0  # should contain the cell pixels

    def test_feature_arrays_are_cropped(self, synthetic_fft_features, synthetic_label_image, output_zarr_path):
        root = crop_zarr_from_masks(
            synthetic_fft_features, synthetic_label_image, output_zarr_path
        )
        # Original shape is (11, 2, 8, 8). Crop should be smaller in last 2 dims.
        full_amp = np.asarray(root["cell_1"]["full_amplitude"][:])
        assert full_amp.ndim == 4
        assert full_amp.shape[0] == 11   # F preserved
        assert full_amp.shape[1] == 2    # C preserved
        assert full_amp.shape[2] < 8     # X cropped
        assert full_amp.shape[3] < 8     # Y cropped

    def test_timeseries_is_cropped(self, synthetic_fft_features, synthetic_label_image, output_zarr_path):
        root = crop_zarr_from_masks(
            synthetic_fft_features, synthetic_label_image, output_zarr_path
        )
        ts = np.asarray(root["cell_1"]["timeseries"][:])
        assert ts.ndim == 4
        assert ts.shape[0] == 20   # T preserved
        assert ts.shape[2] < 8     # X cropped

    def test_root_attrs_preserved(self, synthetic_fft_features, synthetic_label_image, output_zarr_path):
        root = crop_zarr_from_masks(
            synthetic_fft_features, synthetic_label_image, output_zarr_path
        )
        attrs = dict(root.attrs)
        assert attrs["num_cells"] == 2
        assert attrs["padding_fraction"] == 0.1
        assert "source_attrs" in attrs
        assert attrs["source_attrs"]["max_bin"] == 11

    def test_cell_attrs_have_traceability(self, synthetic_fft_features, synthetic_label_image, output_zarr_path):
        root = crop_zarr_from_masks(
            synthetic_fft_features, synthetic_label_image, output_zarr_path
        )
        cell_attrs = dict(root["cell_1"].attrs)
        assert cell_attrs["label_id"] == 1
        assert "bbox_original" in cell_attrs
        assert "bbox_padded" in cell_attrs
        assert "centroid_yx" in cell_attrs
        assert "area_pixels" in cell_attrs
        assert "crop_shape" in cell_attrs
        # Verify bbox_original is within image bounds
        y_min, y_max, x_min, x_max = cell_attrs["bbox_original"]
        assert 0 <= y_min < y_max <= 8
        assert 0 <= x_min < x_max <= 8

    def test_no_attrs_key_in_arrays(self, synthetic_fft_features, synthetic_label_image, output_zarr_path):
        """_attrs should be in .attrs metadata, NOT as an array."""
        root = crop_zarr_from_masks(
            synthetic_fft_features, synthetic_label_image, output_zarr_path
        )
        cell_keys = list(root["cell_1"].keys())
        assert "_attrs" not in cell_keys

    def test_thumbnail_is_generated(self, synthetic_fft_features, synthetic_label_image, output_zarr_path):
        root = crop_zarr_from_masks(
            synthetic_fft_features, synthetic_label_image, output_zarr_path
        )
        assert "thumbnail" in root["cell_1"]
        thumb = np.asarray(root["cell_1"]["thumbnail"][:])
        crop_shape = root["cell_1"].attrs["crop_shape"]
        assert thumb.shape == (2, crop_shape[0], crop_shape[1])



# ---------------------------------------------------------------------------
# Tests: crop_zarr_from_masks with CWT-style features (nested dict)
# ---------------------------------------------------------------------------

class TestCropFromCwtFeatures:
    def test_nested_structure_preserved(self, synthetic_cwt_features, synthetic_label_image, output_zarr_path):
        root = crop_zarr_from_masks(
            synthetic_cwt_features, synthetic_label_image, output_zarr_path
        )
        cell = root["cell_1"]
        assert "channel_0" in cell
        assert "channel_1" in cell
        assert "amp" in cell["channel_0"]
        assert "phase_difference" in cell["channel_1"]

    def test_nested_arrays_cropped(self, synthetic_cwt_features, synthetic_label_image, output_zarr_path):
        root = crop_zarr_from_masks(
            synthetic_cwt_features, synthetic_label_image, output_zarr_path
        )
        amp = np.asarray(root["cell_1"]["channel_0"]["amp"][:])
        assert amp.ndim == 4
        assert amp.shape[0] == 20   # T
        assert amp.shape[1] == 1    # banks
        assert amp.shape[2] < 8     # X cropped
        assert amp.shape[3] < 8     # Y cropped


# ---------------------------------------------------------------------------
# Tests: Zarr round-trip (write features to zarr, then crop from zarr path)
# ---------------------------------------------------------------------------

class TestCropFromZarrPath:
    def test_crop_from_zarr_store(self, synthetic_fft_features, synthetic_label_image, tmp_path):
        from cellstream.io import write_to_zarr
        
        zarr_source = str(tmp_path / "source_features.zarr")
        write_to_zarr(synthetic_fft_features, zarr_source)
        
        output_path = str(tmp_path / "crop_output.zarr")
        root = crop_zarr_from_masks(
            zarr_source, synthetic_label_image, output_path
        )
        assert "cell_1" in root
        assert "cell_2" in root
        # Verify arrays exist and have correct dimensionality
        full_amp = np.asarray(root["cell_1"]["full_amplitude"][:])
        assert full_amp.ndim == 4


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_pixel_cell(self, output_zarr_path):
        """A cell that is just 1 pixel should still get min_padding."""
        labels = np.zeros((20, 20), dtype=np.int64)
        labels[10, 10] = 1
        features = {
            "data": torch.randn(5, 2, 20, 20),
            "_attrs": {"test": True},
        }
        root = crop_zarr_from_masks(features, labels, output_zarr_path, min_padding_px=3)
        assert "cell_1" in root
        crop = np.asarray(root["cell_1"]["data"][:])
        # 1px cell + 3px padding each side = 7x7 crop
        assert crop.shape[-2] == 7
        assert crop.shape[-1] == 7

    def test_custom_background_label(self, output_zarr_path):
        """Non-zero background label should be skipped."""
        labels = np.full((8, 8), 255, dtype=np.int64)  # background = 255
        labels[2:4, 2:4] = 1
        features = {
            "data": torch.randn(5, 2, 8, 8),
            "_attrs": {},
        }
        root = crop_zarr_from_masks(
            features, labels, output_zarr_path, background_label=255
        )
        assert "cell_1" in root
        assert "cell_255" not in root

    def test_torch_label_image(self, synthetic_fft_features, output_zarr_path):
        """Label image as torch tensor should work."""
        labels = torch.zeros(8, 8, dtype=torch.int64)
        labels[3:6, 3:6] = 1
        root = crop_zarr_from_masks(synthetic_fft_features, labels, output_zarr_path)
        assert "cell_1" in root

    def test_crop_data_matches_source(self, output_zarr_path):
        """Verify the cropped values actually match the source data at the right location."""
        X, Y = 16, 16
        labels = np.zeros((X, Y), dtype=np.int64)
        labels[4:8, 4:8] = 1  # 4x4 cell
        
        # Use a tensor with known values
        data = torch.arange(X * Y, dtype=torch.float32).reshape(1, 1, X, Y)
        features = {"data": data, "_attrs": {}}
        
        root = crop_zarr_from_masks(
            features, labels, output_zarr_path,
            padding_fraction=0.0, min_padding_px=0
        )
        
        cropped = np.asarray(root["cell_1"]["data"][:])
        expected = data[0, 0, 4:8, 4:8].numpy()
        np.testing.assert_array_equal(cropped[0, 0], expected)
