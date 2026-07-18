"""
cellstream.flow.process

High-Level Flow Processing Pipelines

This module defines the high-level interface for processing phase images
into flow fields, topological defects, and FTLE ridges.
"""

import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np
import zarr
from tqdm.auto import tqdm

from .analytic import phase_velocity, compute_ftle
from ..phase import winding_number

def process_cell_flow(
    cell_group: zarr.Group,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ftle_integration_time: int = 20,
    smooth_sigma: float = 1.0,
    defect_window_size: int = 5,
    force_recompute: bool = False
):
    """
    Process a single cell's phase data to compute its dense flow features
    and topological defects, storing them back into the cell's Zarr group.
    
    Args:
        cell_group: zarr.Group for a specific cell (e.g. store['cells']['cell_123'])
        device: 'cuda' or 'cpu'
        ftle_integration_time: frames for FTLE integration
        smooth_sigma: smoothing for phase velocity
        defect_window_size: kernel size for winding number computation
        force_recompute: whether to overwrite existing features
    """
    if 'phase' not in cell_group:
        logger.info(f"Skipping {cell_group.name}: No 'phase' array found.")
        return
        
    # Create or get groups
    flow_group = cell_group.require_group('flow')
    defects_group = cell_group.require_group('defects')
    
    # Check if we need to run
    has_velocity = 'velocity' in flow_group
    has_defects = 'winding_number' in defects_group
    if has_velocity and has_defects and not force_recompute:
        return
        
    # Load data
    phase = torch.from_numpy(cell_group['phase'][:].astype('float32')).to(device)
    
    mask = None
    if 'mask' in cell_group:
        mask = torch.from_numpy(cell_group['mask'][:].astype('float32')).to(device)
        
    # --- 1. Compute Phase Defects (Winding Number) ---
    if not has_defects or force_recompute:
        wn = winding_number(phase, n=defect_window_size, device=device)
        wn_np = wn.cpu().numpy()
        
        if 'winding_number' in defects_group:
            del defects_group['winding_number']
        defects_group.create_dataset(
            'winding_number', 
            data=wn_np, 
            chunks=(1, wn_np.shape[1], wn_np.shape[2]),
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        )
        
        # Extract precise coordinates of +1 and -1 defects
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

    # --- 2. Compute Phase Velocity and FTLE ---
    if not has_velocity or force_recompute:
        # Phase velocity
        v, speed, wavenumber = phase_velocity(phase, smooth_sigma=smooth_sigma, device=device)
        
        # Forward FTLE
        ftle_fwd = compute_ftle(
            v, 
            integration_time=ftle_integration_time, 
            device=device, 
            mask=mask, 
            backward=False
        )
        
        # Backward FTLE
        ftle_bwd = compute_ftle(
            v, 
            integration_time=ftle_integration_time, 
            device=device, 
            mask=mask, 
            backward=True
        )
        
        # Move to CPU and save
        v_np = v.cpu().numpy()
        speed_np = speed.cpu().numpy()
        ftle_fwd_np = ftle_fwd.cpu().numpy()
        ftle_bwd_np = ftle_bwd.cpu().numpy()
        
        for name in ['velocity', 'speed', 'ftle_forward', 'ftle_backward']:
            if name in flow_group:
                del flow_group[name]
                
        # Chunking: (1, 2, Y, X) for velocity, (1, Y, X) for others
        flow_group.create_dataset('velocity', data=v_np, chunks=(1, 2, v_np.shape[2], v_np.shape[3]))
        flow_group.create_dataset('speed', data=speed_np, chunks=(1, speed_np.shape[1], speed_np.shape[2]))
        flow_group.create_dataset('ftle_forward', data=ftle_fwd_np, chunks=(1, ftle_fwd_np.shape[1], ftle_fwd_np.shape[2]))
        flow_group.create_dataset('ftle_backward', data=ftle_bwd_np, chunks=(1, ftle_bwd_np.shape[1], ftle_bwd_np.shape[2]))

    # Clean up GPU memory
    torch.cuda.empty_cache()


def process_zarr_store_flow(zarr_path: str, force: bool = False, **kwargs):
    """
    Iterate over all cells in a Zarr store and compute flow features.
    
    Args:
        zarr_path: Path to the cellstream Zarr store
        force: Force recomputation of existing features
        kwargs: Passed to process_cell_flow
    """
    logger.info(f"Opening Zarr store: {zarr_path}")
    store = zarr.open(zarr_path, mode='a')
    
    if 'cells' not in store:
        logger.error("Store does not contain a 'cells' group.")
        return
        
    cells_group = store['cells']
    cell_ids = list(cells_group.keys())
    logger.info(f"Found {len(cell_ids)} cells.")
    
    for cell_id in tqdm(cell_ids, desc="Processing flow features"):
        try:
            process_cell_flow(cells_group[cell_id], force_recompute=force, **kwargs)
        except Exception as e:
            logger.error(f"Error processing {cell_id}: {e}")
