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
    parent_mask: torch.Tensor = None,
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
        # Check if phase exists in any subchannels (like minD, minE)
        # Load mask from top level if it exists
        top_mask = None
        if 'mask' in cell_group:
            top_mask = torch.from_numpy(cell_group['mask'][:].astype('float32'))
            
        found_subchannels = False
        for key in list(cell_group.keys()):
            subgroup = cell_group[key]
            if isinstance(subgroup, zarr.Group) and 'phase' in subgroup:
                process_cell(subgroup, force_recompute=force_recompute, device=device, parent_mask=top_mask, **kwargs)
                found_subchannels = True
                
        if not found_subchannels:
            logger.info(f"Skipping {cell_group.name}: No 'phase' array found.")
        return
        
    flow_group = cell_group.require_group('flow')
    defects_group = cell_group.require_group('defects')
    
    # Check if we need to run
    if not force_recompute and 'velocity' in flow_group and 'winding_number' in defects_group:
        return
        
    # Load data
    phase = torch.from_numpy(cell_group['phase'][:].astype('float32'))
    
    # Try to find a mask (either in this group or passed from parent)
    mask = None
    if 'mask' in cell_group:
        mask = torch.from_numpy(cell_group['mask'][:].astype('float32'))
    elif parent_mask is not None:
        mask = parent_mask
        
    if mask is not None:
        mask = mask.squeeze()

    # Handle CWT output format (T, 1, Y, X)
    phase = phase.squeeze()
    
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
    wn_np = features.get('winding_number')

    if wn_np is not None:
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
    v_np = features.get('velocity')
    speed_np = features.get('speed')
    ftle_fwd_np = features.get('ftle_forward')
    ftle_bwd_np = features.get('ftle_backward')
    streams_np = features.get('streamlines')
    pstreams_np = features.get('phase_streamlines')

    for name in ('velocity', 'speed', 'ftle_forward', 'ftle_backward', 'streamlines', 'phase_streamlines'):
        if name in flow_group and features.get(name) is not None:
            del flow_group[name]
            
    if v_np is not None:
        flow_group.create_dataset('velocity',      data=v_np,        chunks=(1, 2, v_np.shape[2], v_np.shape[3]))
    if speed_np is not None:
        flow_group.create_dataset('speed',          data=speed_np,    chunks=(1, speed_np.shape[1], speed_np.shape[2]))
    if ftle_fwd_np is not None:
        flow_group.create_dataset('ftle_forward',   data=ftle_fwd_np, chunks=(1, ftle_fwd_np.shape[1], ftle_fwd_np.shape[2]))
    if ftle_bwd_np is not None:
        flow_group.create_dataset('ftle_backward',  data=ftle_bwd_np, chunks=(1, ftle_bwd_np.shape[1], ftle_bwd_np.shape[2]))
    if streams_np is not None:
        flow_group.create_dataset('streamlines',    data=streams_np,  chunks=(1, 3, streams_np.shape[2], streams_np.shape[3]))
    if pstreams_np is not None:
        flow_group.create_dataset('phase_streamlines', data=pstreams_np, chunks=(1, 3, pstreams_np.shape[2], pstreams_np.shape[3]))

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
    
    if 'cells' in store:
        cells_group = store['cells']
        cell_ids = list(cells_group.keys())
    else:
        # Fallback to checking if cells are at the root
        cell_ids = [k for k in store.keys() if k.startswith('cell_')]
        if not cell_ids:
            logger.error("Store does not contain a 'cells' group or any 'cell_' root groups.")
            return
        cells_group = store

    logger.info(f"Found {len(cell_ids)} cells.")
    
    for cell_id in tqdm(cell_ids, desc="Processing phase features"):
        try:
            process_cell(cells_group[cell_id], force_recompute=force, **kwargs)
        except Exception as e:
            logger.error(f"Error processing {cell_id}: {e}")
