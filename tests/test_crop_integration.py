"""Tests for crop_zarrs integration in FFT and CWT process pipelines."""
import os
import shutil
import tempfile

import numpy as np
import pytest
import torch

# -- helpers -------------------------------------------------------------------

@pytest.fixture
def small_image():
    """Small 4-D image (T, C, X, Y) with a clean oscillating signal."""
    T, C, X, Y = 20, 2, 16, 16
    t = torch.arange(T).float().view(T, 1, 1, 1)
    img = torch.sin(2 * 3.14159 * t / 10).expand(T, C, X, Y).clone()
    # Add some noise
    img += torch.randn_like(img) * 0.1
    return img


@pytest.fixture
def small_masks():
    """2-D label mask with 2 cells."""
    masks = torch.zeros(16, 16, dtype=torch.int64)
    masks[2:6, 2:6] = 1
    masks[9:13, 9:13] = 2
    return masks


@pytest.fixture
def tmp_zarr_dir(tmp_path):
    """Return a temporary directory for zarr output."""
    return str(tmp_path)


# -- FFT integration -----------------------------------------------------------

class TestFFTCropIntegration:

    def test_default_no_crop(self, small_image, small_masks):
        """crop_zarrs=False (default) returns DataFrame only."""
        from cellstream.fft.process import process_image_cellstreams

        result = process_image_cellstreams(small_image, small_masks)
        import pandas as pd
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_crop_zarrs_produces_store(self, small_image, small_masks, tmp_zarr_dir):
        """crop_zarrs=True returns (df, crop_root) and writes zarr on disk."""
        from cellstream.fft.process import process_image_cellstreams

        out_path = os.path.join(tmp_zarr_dir, "fft_test.zarr")
        result = process_image_cellstreams(
            small_image,
            small_masks,
            crop_zarrs=True,
            crop_output_path=out_path,
            image_filename="test_image.tif",
        )

        # Should return a tuple (df, crop_root)
        assert isinstance(result, tuple)
        df, crop_root = result[0], result[-1]

        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Zarr store should exist
        assert os.path.exists(out_path)

        # Should have cell_1 and cell_2 groups
        cell_keys = list(crop_root.group_keys())
        assert "cell_1" in cell_keys
        assert "cell_2" in cell_keys

    def test_extracted_attrs_present(self, small_image, small_masks, tmp_zarr_dir):
        """Each cell group should have extracted_ attributes from the DataFrame."""
        from cellstream.fft.process import process_image_cellstreams

        out_path = os.path.join(tmp_zarr_dir, "fft_attrs.zarr")
        result = process_image_cellstreams(
            small_image,
            small_masks,
            crop_zarrs=True,
            crop_output_path=out_path,
            image_filename="test_image.tif",
        )
        _, crop_root = result[0], result[-1]

        cell_group = crop_root["cell_1"]
        attr_keys = list(cell_group.attrs)
        extracted_keys = [k for k in attr_keys if k.startswith("extracted_")]
        assert len(extracted_keys) > 0, "Expected extracted_ attributes on cell_1"

    def test_return_with_fft_features_and_crop(self, small_image, small_masks, tmp_zarr_dir):
        """return_fft_features=True + crop_zarrs=True returns (df, fft_features, crop_root)."""
        from cellstream.fft.process import process_image_cellstreams

        out_path = os.path.join(tmp_zarr_dir, "fft_both.zarr")
        result = process_image_cellstreams(
            small_image,
            small_masks,
            return_fft_features=True,
            crop_zarrs=True,
            crop_output_path=out_path,
            image_filename="test_image.tif",
        )
        assert isinstance(result, tuple)
        assert len(result) == 3  # (df, fft_features, crop_root)

    def test_auto_output_path(self, small_image, small_masks, tmp_path, monkeypatch):
        """When crop_output_path is None, auto-generates from image_filename."""
        from cellstream.fft.process import process_image_cellstreams

        # Change cwd so auto-generated path lands in temp dir
        monkeypatch.chdir(tmp_path)

        result = process_image_cellstreams(
            small_image,
            small_masks,
            crop_zarrs=True,
            image_filename="my_sample.tif",
        )
        df, crop_root = result[0], result[-1]
        assert os.path.exists("my_sample_crops.zarr")


# -- CWT integration ----------------------------------------------------------

class TestCWTCropIntegration:

    @pytest.fixture(autouse=True)
    def _skip_if_no_ssqueezepy(self):
        try:
            import ssqueezepy
        except ImportError:
            pytest.skip("ssqueezepy not installed")

    def test_default_no_crop(self, small_image, small_masks):
        """crop_zarrs=False (default) returns DataFrame only."""
        from cellstream.cwt.process import process_cwt_image_cellstreams

        result = process_cwt_image_cellstreams(
            small_image,
            small_masks,
            min_scale=4,
            max_scale=8,
            num_filter_banks=1,
            blocks=1,
        )
        import pandas as pd
        assert isinstance(result, pd.DataFrame)

    def test_crop_zarrs_produces_store(self, small_image, small_masks, tmp_zarr_dir):
        """crop_zarrs=True returns (df, crop_root)."""
        from cellstream.cwt.process import process_cwt_image_cellstreams

        out_path = os.path.join(tmp_zarr_dir, "cwt_test.zarr")
        result = process_cwt_image_cellstreams(
            small_image,
            small_masks,
            min_scale=4,
            max_scale=8,
            num_filter_banks=1,
            blocks=1,
            crop_zarrs=True,
            crop_output_path=out_path,
            image_filename="test_image.tif",
        )

        assert isinstance(result, tuple)
        df, crop_root = result
        assert os.path.exists(out_path)

        cell_keys = list(crop_root.group_keys())
        assert "cell_1" in cell_keys
        assert "cell_2" in cell_keys

    def test_extracted_cwt_attrs_present(self, small_image, small_masks, tmp_zarr_dir):
        """Each cell group should have extracted_ CWT attributes."""
        from cellstream.cwt.process import process_cwt_image_cellstreams

        out_path = os.path.join(tmp_zarr_dir, "cwt_attrs.zarr")
        result = process_cwt_image_cellstreams(
            small_image,
            small_masks,
            min_scale=4,
            max_scale=8,
            num_filter_banks=1,
            blocks=1,
            crop_zarrs=True,
            crop_output_path=out_path,
            image_filename="test_image.tif",
        )
        _, crop_root = result

        cell_group = crop_root["cell_1"]
        attr_keys = list(cell_group.attrs)
        extracted_keys = [k for k in attr_keys if k.startswith("extracted_")]
        assert len(extracted_keys) > 0, "Expected extracted_ attributes on cell_1"
