"""
cellstream.features

High-Level Feature Extraction Pipelines

This module provides unified interfaces for extracting mathematical
features (flow, defects, FTLE) from phase fields, cleanly separating
the core math from Zarr data pipeline logic.
"""

import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np
import zarr
from tqdm.auto import tqdm

from .flow.analytic import phase_velocity, compute_ftle
from .phase import winding_number

def generate_phase_features(
    phase: torch.Tensor,
    mask: torch.Tensor = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ftle_integration_time: int = 20,
    smooth_sigma: float = 1.0,
    defect_window_size: int = 5,
):
    """
    Generate all relevant features from a phase field (defects, flow, FTLE).
    
    Args:
        phase: (T, Y, X) tensor of phase values
        mask: Optional (T, Y, X) or (Y, X) mask
        device: 'cuda' or 'cpu'
        ftle_integration_time: frames for FTLE integration
        smooth_sigma: smoothing for phase velocity
        defect_window_size: kernel size for winding number computation
        
    Returns:
        dict containing the generated feature tensors (moved to CPU as numpy arrays)
    """
    phase = phase.to(device)
    if mask is not None:
        mask = mask.to(device)
        
    features = {}
    
    # 1. Defects (Winding Number Field)
    # Winding number mathematically produces blocky "fields" around singularities
    wn = winding_number(phase, n=defect_window_size, device=device)
    features['winding_number'] = wn.cpu().numpy()
    
    # 2. Phase Velocity
    v, speed, _ = phase_velocity(phase, smooth_sigma=smooth_sigma, device=device)
    features['velocity'] = v.cpu().numpy()
    features['speed'] = speed.cpu().numpy()
    
    # 3. FTLE (Forward and Backward)
    ftle_fwd = compute_ftle(
        v, 
        integration_time=ftle_integration_time, 
        device=device, 
        mask=mask, 
        backward=False
    )
    features['ftle_forward'] = ftle_fwd.cpu().numpy()
    
    ftle_bwd = compute_ftle(
        v, 
        integration_time=ftle_integration_time, 
        device=device, 
        mask=mask, 
        backward=True
    )
    features['ftle_backward'] = ftle_bwd.cpu().numpy()
    
    return features


def process_cell(
    cell_group: zarr.Group,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    force_recompute: bool = False,
    **kwargs
):
    """
    Process a single cell's phase data, attaching generated features
    back into the cell's Zarr group.
    """
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
        
    # Generate unified features
    features = generate_phase_features(phase, mask=mask, device=device, **kwargs)
    
    # Write Defects
    if 'winding_number' in defects_group:
        del defects_group['winding_number']
    wn_np = features['winding_number']
    defects_group.create_dataset(
        'winding_number', 
        data=wn_np, 
        chunks=(1, wn_np.shape[1], wn_np.shape[2]),
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    )
    
    # Extract coordinates as a convenience
    T = wn_np.shape[0]
    pos_defects_list = []
    neg_defects_list = []
    for t in range(T):
        pos_y, pos_x = np.where(wn_np[t] > 0.5)
        neg_y, neg_x = np.where(wn_np[t] < -0.5)
        pos_defects_list.append(np.stack([np.full_like(pos_y, t), pos_y, pos_x], axis=1))
        neg_defects_list.append(np.stack([np.full_like(neg_y, t), neg_y, neg_x], axis=1))
        
    pos_defects = np.concatenate(pos_defects_list, axis=0) if pos_defects_list else np.empty((0, 3))
    neg_defects = np.concatenate(neg_defects_list, axis=0) if neg_defects_list else np.empty((0, 3))
    
    if 'positive_coords' in defects_group:
        del defects_group['positive_coords']
    if 'negative_coords' in defects_group:
        del defects_group['negative_coords']
        
    defects_group.create_dataset('positive_coords', data=pos_defects)
    defects_group.create_dataset('negative_coords', data=neg_defects)
    
    # Write Flow
    for name in ['velocity', 'speed', 'ftle_forward', 'ftle_backward']:
        if name in flow_group:
            del flow_group[name]
            
    v_np = features['velocity']
    speed_np = features['speed']
    ftle_fwd_np = features['ftle_forward']
    ftle_bwd_np = features['ftle_backward']
    
    flow_group.create_dataset('velocity', data=v_np, chunks=(1, 2, v_np.shape[2], v_np.shape[3]))
    flow_group.create_dataset('speed', data=speed_np, chunks=(1, speed_np.shape[1], speed_np.shape[2]))
    flow_group.create_dataset('ftle_forward', data=ftle_fwd_np, chunks=(1, ftle_fwd_np.shape[1], ftle_fwd_np.shape[2]))
    flow_group.create_dataset('ftle_backward', data=ftle_bwd_np, chunks=(1, ftle_bwd_np.shape[1], ftle_bwd_np.shape[2]))

    torch.cuda.empty_cache()


def process_zarr_store(zarr_path: str, force: bool = False, **kwargs):
    """
    Iterate over all cells in a Zarr store and attach phase features.
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
