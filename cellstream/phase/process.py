"""
cellstream.phase.process

Batch processing pipelines for phase-field feature extraction.

Provides cell-level and store-level Zarr I/O orchestration that calls
``generate_phase_features`` from ``phase.utils`` and writes the results
into the appropriate Zarr groups.

Expected Zarr layout written by ``process_cell``::

    cell_group/
        phase             # (T, 1, Y, X) input — must already exist
        mask              # (Y, X) optional input
        defects/
            winding_number   # (T, 1, Y, X)  — matches CWT sibling shape
            positive_coords  # (N, 3) — [t, y, x]
            negative_coords  # (N, 3) — [t, y, x]
        flow/
            velocity         # (T, 2, Y, X)
            speed            # (T, 1, Y, X)  — matches CWT sibling shape
            wavenumber       # (T, 1, Y, X)  — matches CWT sibling shape
            ftle_forward     # (T, 1, Y, X)  — matches CWT sibling shape
            ftle_backward    # (T, 1, Y, X)  — matches CWT sibling shape
            streamlines      # (T, 1, Y, X)  — matches CWT sibling shape
            phase_streamlines # (T, 3, Y, X) — RGB
"""

import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np
import zarr
from tqdm.auto import tqdm

from .utils import generate_phase_features


def _expand_to_4d(arr):
    """Insert a singleton dim at axis=1 so (T, Y, X) → (T, 1, Y, X).
    
    This ensures phase-derived features share the same axis convention as
    the CWT sibling arrays (T, num_banks, Y, X) in the same Zarr group,
    so Napari can overlay them without shape mismatch.
    """
    if arr.ndim == 3:
        return np.expand_dims(arr, axis=1)
    return arr


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
    
    # --- P4: Smart skip logic — check which requested features are missing ---
    requested = kwargs.get('phase_features_to_process', None)
    if requested is None:
        requested = ['velocity', 'speed', 'ftle_forward', 'ftle_backward', 'winding_number']
    
    if not force_recompute:
        # Map feature names to the group + key where they're stored
        existing = set()
        for k in flow_group.keys():
            existing.add(k)
        for k in defects_group.keys():
            existing.add(k)
        
        missing = [f for f in requested if f not in existing]
        if not missing:
            return  # All requested features already present
        
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
    
    # --- Write _attrs metadata for reproducibility (P3) ---
    attrs_dict = features.get('_attrs', {})
    if attrs_dict:
        cell_group.attrs.update({'phase_processing': attrs_dict})
    
    # --- Write Defects ---
    wn_np = features.get('winding_number')

    if wn_np is not None:
        # P0: Expand (T, Y, X) → (T, 1, Y, X) to match CWT axis convention
        wn_4d = _expand_to_4d(wn_np)
        if 'winding_number' in defects_group:
            del defects_group['winding_number']
        defects_group.create_dataset(
            'winding_number', 
            data=wn_4d, 
            chunks=(1, 1, wn_4d.shape[2], wn_4d.shape[3]),
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
    # Features that keep their native shape (already have a non-spatial dim at axis 1)
    v_np = features.get('velocity')             # (T, 2, Y, X)
    pstreams_np = features.get('phase_streamlines')  # (T, 3, Y, X)
    
    # Features that need (T, Y, X) → (T, 1, Y, X) expansion for CWT compatibility
    speed_np = features.get('speed')            # (T, Y, X) → (T, 1, Y, X)
    wavenumber_np = features.get('wavenumber')  # (T, Y, X) → (T, 1, Y, X)
    ftle_fwd_np = features.get('ftle_forward')  # (T, Y, X) → (T, 1, Y, X)
    ftle_bwd_np = features.get('ftle_backward') # (T, Y, X) → (T, 1, Y, X)
    streams_np = features.get('streamlines')    # (T, Y, X) → (T, 1, Y, X)

    all_flow_keys = ('velocity', 'speed', 'wavenumber', 'ftle_forward', 'ftle_backward', 'streamlines', 'phase_streamlines')
    for name in all_flow_keys:
        if name in flow_group and features.get(name) is not None:
            del flow_group[name]
            
    if v_np is not None:
        flow_group.create_dataset('velocity', data=v_np, chunks=(1, 2, v_np.shape[2], v_np.shape[3]))
    if pstreams_np is not None:
        flow_group.create_dataset('phase_streamlines', data=pstreams_np, chunks=(1, 3, pstreams_np.shape[2], pstreams_np.shape[3]))
    
    # P0: Write scalar fields with singleton bank dim to match CWT convention
    for name, arr in [('speed', speed_np), ('wavenumber', wavenumber_np),
                      ('ftle_forward', ftle_fwd_np), ('ftle_backward', ftle_bwd_np),
                      ('streamlines', streams_np)]:
        if arr is not None:
            arr_4d = _expand_to_4d(arr)
            flow_group.create_dataset(name, data=arr_4d, chunks=(1, 1, arr_4d.shape[2], arr_4d.shape[3]))

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
