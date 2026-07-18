"""
cellstream.phase

Phase-field analysis: topological defect detection, phase velocity,
FTLE computation, and streamline generation.
"""

from .utils import winding_number, generate_phase_features
from .analytic import (
    phase_velocity,
    compute_ftle,
    generate_streamlines,
    generate_instantaneous_streamlines,
    generate_phase_colored_streamlines,
)
from .process import process_cell, process_zarr_store

__all__ = [
    # Low-level math
    "winding_number",
    "phase_velocity",
    "compute_ftle",
    "generate_streamlines",
    "generate_instantaneous_streamlines",
    "generate_phase_colored_streamlines",
    # Feature generation
    "generate_phase_features",
    # Batch processing
    "process_cell",
    "process_zarr_store",
]
