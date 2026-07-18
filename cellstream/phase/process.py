"""
cellstream.phase.process

Batch processing pipelines for phase-field feature extraction.

Provides cell-level and store-level Zarr I/O orchestration that calls
``generate_phase_features`` from ``phase.utils`` and writes the results
into the appropriate Zarr groups.

Expected Zarr layout written by ``process_cell``::

    cell_group/
        phase           # (T, Y, X) input — must already exist
        mask            # (T, Y, X) optional input
        defects/
            winding_number   # (T, Y, X)
            positive_coords  # (N, 3) — [t, y, x]
            negative_coords  # (N, 3) — [t, y, x]
        flow/
            velocity         # (T, 2, Y, X)
            speed            # (T, Y, X)
            ftle_forward     # (T, Y, X)
            ftle_backward    # (T, Y, X)
"""

import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np
import zarr
from tqdm.auto import tqdm

from .utils import generate_phase_features


def process_cell(
    cell_group: zarr.Group,
    device: str = None,
    force_recompute: bool = False,
    **kwargs
):
    """
    Process a single cell's phase data and write derived features into
    the cell's Zarr group.

    Args:
        cell_group: Zarr group containing at least a 'phase' array.
        device: 'cuda', 'cpu', or None (auto-detect).
        force_recompute: If True, overwrite existing results.
        **kwargs: Forwarded to ``generate_phase_features``
                  (ftle_integration_time, smooth_sigma, defect_window_size).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'phase' not in cell_group:
        logger.info(f"Skipping {cell_group.name}: No 'phase' array found.")
        return
        
    flow_group = cell_group.require_group('flow')
    defects_group = cell_group.require_group('defects')
    
    # Check if we need to run
    if not force_recompute and 'velocity' in flow_group and 'winding_number' in defects_group:
        return
        
    # Load data
    phase = torch.from_numpy(cell_group['phase'][:].astype('float32'))
    mask = None
    if 'mask' in cell_group:
        mask = torch.from_numpy(cell_group['mask'][:].astype('float32'))

    # Validate shape — process_cell expects (T, Y, X)
    if phase.ndim != 3:
        logger.warning(
            f"Skipping {cell_group.name}: expected 3-D phase (T, Y, X), "
            f"got shape {tuple(phase.shape)}"
        )
        return

    # Generate unified features
    features = generate_phase_features(phase, mask=mask, device=device, **kwargs)
    
    # --- Write Defects ---
    wn_np = features['winding_number']

    if 'winding_number' in defects_group:
        del defects_group['winding_number']
    defects_group.create_dataset(
        'winding_number', 
        data=wn_np, 
        chunks=(1, wn_np.shape[1], wn_np.shape[2]),
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    )
    
    # Extract defect coordinates as a convenience table
    T = wn_np.shape[0]
    t_indices = np.arange(T)
    pos_lists = []
    neg_lists = []
    for t in t_indices:
        pos_y, pos_x = np.where(wn_np[t] > 0.5)
        neg_y, neg_x = np.where(wn_np[t] < -0.5)
        if len(pos_y):
            pos_lists.append(np.column_stack([np.full(len(pos_y), t), pos_y, pos_x]))
        if len(neg_y):
            neg_lists.append(np.column_stack([np.full(len(neg_y), t), neg_y, neg_x]))
        
    pos_defects = np.concatenate(pos_lists, axis=0) if pos_lists else np.empty((0, 3), dtype=np.intp)
    neg_defects = np.concatenate(neg_lists, axis=0) if neg_lists else np.empty((0, 3), dtype=np.intp)
    
    for key in ('positive_coords', 'negative_coords'):
        if key in defects_group:
            del defects_group[key]
        
    defects_group.create_dataset('positive_coords', data=pos_defects)
    defects_group.create_dataset('negative_coords', data=neg_defects)
    
    # --- Write Flow ---
    v_np = features['velocity']
    speed_np = features['speed']
    ftle_fwd_np = features['ftle_forward']
    ftle_bwd_np = features['ftle_backward']

    for name in ('velocity', 'speed', 'ftle_forward', 'ftle_backward'):
        if name in flow_group:
            del flow_group[name]
            
    flow_group.create_dataset('velocity',      data=v_np,        chunks=(1, 2, v_np.shape[2], v_np.shape[3]))
    flow_group.create_dataset('speed',          data=speed_np,    chunks=(1, speed_np.shape[1], speed_np.shape[2]))
    flow_group.create_dataset('ftle_forward',   data=ftle_fwd_np, chunks=(1, ftle_fwd_np.shape[1], ftle_fwd_np.shape[2]))
    flow_group.create_dataset('ftle_backward',  data=ftle_bwd_np, chunks=(1, ftle_bwd_np.shape[1], ftle_bwd_np.shape[2]))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_zarr_store(zarr_path: str, force: bool = False, **kwargs):
    """
    Iterate over all cells in a Zarr store and attach phase features.
    
    Args:
        zarr_path: Path to the Zarr store (directory or .zip).
        force: If True, recompute features even if they already exist.
        **kwargs: Forwarded to ``process_cell`` / ``generate_phase_features``.
    """
    logger.info(f"Opening Zarr store: {zarr_path}")
    store = zarr.open(zarr_path, mode='a')
    
    if 'cells' not in store:
        logger.error("Store does not contain a 'cells' group.")
        return
        
    cells_group = store['cells']
    cell_ids = list(cells_group.keys())
    logger.info(f"Found {len(cell_ids)} cells.")
    
    for cell_id in tqdm(cell_ids, desc="Processing phase features"):
        try:
            process_cell(cells_group[cell_id], force_recompute=force, **kwargs)
        except Exception as e:
            logger.error(f"Error processing {cell_id}: {e}")
