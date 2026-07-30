"""
tests/test_runner.py

Unit and integration tests for cellstream.runner TOML parser and execution engine.
"""

import pytest
from pathlib import Path
from cellstream.runner import load_job_config, resolve_job_params, run_pipeline


class TestLoadJobConfig:
    """Test TOML configuration parsing and validation."""

    def test_valid_single_job(self, tmp_path):
        config_file = tmp_path / "single_job.toml"
        config_file.write_text(
            """
            [defaults]
            images = "path/to/images"
            masks = "path/to/masks"

            [[jobs]]
            type = "cwt"
            min_scale = 50
            """
        )
        config = load_job_config(config_file)
        assert "jobs" in config
        assert len(config["jobs"]) == 1
        assert config["jobs"][0]["type"] == "cwt"
        assert config["jobs"][0]["min_scale"] == 50

    def test_valid_chained_jobs(self, tmp_path):
        config_file = tmp_path / "chained_job.toml"
        config_file.write_text(
            """
            [defaults]
            images = "path/to/images"
            masks = "path/to/masks"

            [[jobs]]
            type = "cwt"
            crop_zarrs = true

            [[jobs]]
            type = "phase"
            smooth_sigma = 1.5

            [[jobs]]
            type = "fft"
            cutoff_frequency_bin = 2
            """
        )
        config = load_job_config(config_file)
        assert len(config["jobs"]) == 3
        assert [j["type"] for j in config["jobs"]] == ["cwt", "phase", "fft"]

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_job_config("nonexistent_config_file_xyz.toml")

    def test_missing_jobs_section_raises(self, tmp_path):
        config_file = tmp_path / "invalid.toml"
        config_file.write_text(
            """
            [defaults]
            images = "path/to/images"
            """
        )
        with pytest.raises(ValueError, match="must contain a non-empty \\[\\[jobs\\]\\] array"):
            load_job_config(config_file)

    def test_unknown_job_type_raises(self, tmp_path):
        config_file = tmp_path / "unknown_type.toml"
        config_file.write_text(
            """
            [[jobs]]
            type = "invalid_job_type_123"
            """
        )
        with pytest.raises(ValueError, match="Unknown job type"):
            load_job_config(config_file)


class TestResolveJobParams:
    """Test parameter resolution, defaults merging, and chaining rules."""

    def test_job_overrides_defaults(self):
        defaults = {"images": "img_dir", "masks": "mask_dir", "use_gpu": True, "min_scale": 100}
        job = {"type": "cwt", "min_scale": 80}
        jtype, mode, params = resolve_job_params(job, defaults, previous_output=None)
        assert jtype == "cwt"
        assert mode == "folder"
        assert params["use_gpu"] is True
        assert params["min_scale"] == 80  # Job overridden value
        assert "type" not in params


    def test_folder_mode_resolution(self):
        defaults = {"images": "img_dir", "masks": "mask_dir"}
        job = {"type": "fft"}
        jtype, mode, params = resolve_job_params(job, defaults, previous_output=None)
        assert mode == "folder"
        assert params["images"] == "img_dir"
        assert params["masks"] == "mask_dir"

    def test_zarr_mode_explicit_input(self):
        defaults = {}
        job = {"type": "phase", "input": "cells.zarr"}
        jtype, mode, params = resolve_job_params(job, defaults, previous_output=None)
        assert mode == "zarr"
        assert params["input"] == "cells.zarr"

    def test_chaining_sets_input_from_previous_output(self):
        defaults = {}
        job = {"type": "phase"}
        jtype, mode, params = resolve_job_params(job, defaults, previous_output="prev_output_crops.zarr")
        assert mode == "zarr"
        assert params["input"] == "prev_output_crops.zarr"

    def test_phase_first_job_without_input_raises(self):
        defaults = {}
        job = {"type": "phase"}
        with pytest.raises(ValueError, match="operates on Zarr stores"):
            resolve_job_params(job, defaults, previous_output=None)

    def test_stft_zarr_mode_raises(self):
        defaults = {"input": "crops.zarr"}
        job = {"type": "stft"}
        with pytest.raises(ValueError, match="STFT does not yet support Zarr store processing"):
            resolve_job_params(job, defaults, previous_output=None)


class TestDryRun:
    """Test pipeline dry-run mode."""

    def test_dry_run_execution(self, tmp_path, capsys):
        config_file = tmp_path / "dry_run_test.toml"
        config_file.write_text(
            """
            [defaults]
            images = "mock/images"
            masks = "mock/masks"
            output = "mock/output"

            [[jobs]]
            type = "cwt"
            crop_zarrs = true

            [[jobs]]
            type = "phase"
            smooth_sigma = 1.0
            """
        )
        run_pipeline(config_file, dry_run=True)
        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out
        assert "cwt" in captured.out
        assert "phase" in captured.out
