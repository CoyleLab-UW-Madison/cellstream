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
import os
import re
import zarr
import pandas as pd
import time
from tqdm.auto import tqdm

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.tree import Tree
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

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
    root_group: zarr.Group = None,
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

    if root_group is None:
        root_group = cell_group

    if 'phase' not in cell_group:
        # Check if phase exists in any subchannels (like minD, minE)
        # Load mask from top level if it exists, otherwise inherit from parent
        top_mask = parent_mask
        if 'mask' in cell_group:
            top_mask = torch.from_numpy(cell_group['mask'][:].astype('float32'))
            
        any_processed = False
        for key in list(cell_group.keys()):
            subgroup = cell_group[key]
            if isinstance(subgroup, zarr.Group):
                processed = process_cell(subgroup, force_recompute=force_recompute, device=device, parent_mask=top_mask, root_group=root_group, **kwargs)
                if processed:
                    any_processed = True
                
        return any_processed
        
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
            return False # All requested features already present
        
    # Load data
    phase = torch.from_numpy(cell_group['phase'][:].astype('float32'))
    
    # Try to find a mask (either in this group or passed from parent)
    mask = None
    if kwargs.get('use_mask', True):
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
        logger.debug(
            f"Skipping {cell_group.name}: expected 3-D phase (T, Y, X), "
            f"got shape {tuple(phase.shape)}"
        )
        return False

    # Extract rich_progress if present so we can pass it down
    rich_progress = kwargs.pop('rich_progress', None)
    rich_cell_name = kwargs.pop('rich_cell_name', cell_group.name)
    image_name = kwargs.pop('image_name', "")
    if rich_progress is not None:
        kwargs['rich_progress'] = rich_progress
        kwargs['rich_cell_name'] = rich_cell_name
        
    cell_id_match = re.search(r'cell_(\d+)', cell_group.name)
    label_id = int(cell_id_match.group(1)) if cell_id_match else cell_group.name.split('/')[-1]
    
    df_rows = []

    # Generate unified features
    features = generate_phase_features(phase, mask=mask, device=device, **kwargs)
    
    # --- Write _attrs metadata for reproducibility (P3) ---
    attrs_dict = features.get('_attrs', {})
    if attrs_dict:
        root_group.attrs.update({'phase_processing': attrs_dict})
    
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
            
    # --- Inline Aggregation ---
    if mask is not None:
        mask_np = mask.numpy() if hasattr(mask, 'numpy') else np.asarray(mask)
        valid_pixels = mask_np > 0
        if valid_pixels.sum() > 0:
            attrs_update = {}
            for name, arr in [('winding_number', wn_np), ('velocity', v_np), ('speed', speed_np), 
                              ('wavenumber', wavenumber_np), ('ftle_forward', ftle_fwd_np), 
                              ('ftle_backward', ftle_bwd_np)]:
                if arr is not None:
                    arr_4d = arr if arr.ndim == 4 else _expand_to_4d(arr)
                    spatial_mean = arr_4d[..., valid_pixels].mean(axis=-1)
                    spatial_std = arr_4d[..., valid_pixels].std(axis=-1)
                    
                    T_len, C_len = spatial_mean.shape
                    
                    for t in range(T_len):
                        for c in range(C_len):
                            df_rows.append({
                                "image_filename": image_name,
                                "cell_id": label_id,
                                "frame": t,
                                "channel": "phase",
                                "feature": name,
                                "filter_bank": c,
                                "mean": float(spatial_mean[t, c]),
                                "std": float(spatial_std[t, c])
                            })
                            
                    temp_mean = spatial_mean.mean(axis=0)
                    temp_std = spatial_std.mean(axis=0)
                    for c in range(C_len):
                        key = f"{name}_c{c}"
                        attrs_update[f"{key}_mean"] = float(temp_mean[c])
                        attrs_update[f"{key}_std"] = float(temp_std[c])
                        
            root_group.attrs.update({'phase_stats': attrs_update})

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return True, df_rows


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
    
    if RICH_AVAILABLE:
        # Build Tree Visualization
        tree = Tree(f"[bold blue]{zarr_path}[/bold blue]")
        num_cells = len(cell_ids)
        num_subchannels = 0
        
        for idx, cell_id in enumerate(cell_ids):
            if idx < 5:
                cell_node = tree.add(f"[cyan]{cell_id}[/cyan]")
            
            for k in cells_group[cell_id].keys():
                if isinstance(cells_group[cell_id][k], zarr.Group) and 'phase' in cells_group[cell_id][k]:
                    if idx < 5:
                        cell_node.add(f"[blue]{k}[/blue] [green](Phase found)[/green]")
                    num_subchannels += 1
                elif k == 'phase':
                    if idx < 5:
                        cell_node.add(f"[blue]phase[/blue] [green](Phase found)[/green]")
                        
        if num_cells > 5:
            tree.add(f"... and {num_cells - 5} more cells")
            
        tree.add(f"Found [bold cyan]{num_cells}[/bold cyan] cells, [bold cyan]{num_subchannels}[/bold cyan] subchannels ready for processing.")
        console.print(tree)
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        )
        
        stats = {'processed': 0, 'skipped': 0, 'errors': 0}
        start_time = time.time()
        
        all_df_rows = []
        image_name = os.path.basename(zarr_path).replace("_crops.zarr", "")
        
        with progress:
            main_task = progress.add_task("[bold green]Processing Phase features...", total=len(cell_ids))
            kwargs['rich_progress'] = progress
            kwargs['image_name'] = image_name
            
            for cell_id in cell_ids:
                try:
                    kwargs['rich_cell_name'] = cell_id
                    res = process_cell(cells_group[cell_id], force_recompute=force, **kwargs)
                    if isinstance(res, tuple):
                        processed, rows = res
                    else:
                        processed, rows = res, []
                        
                    if processed:
                        stats['processed'] += 1
                        all_df_rows.extend(rows)
                    else:
                        stats['skipped'] += 1
                except Exception as e:
                    stats['errors'] += 1
                    console.print(f"[red][ ERROR ][/red] {cell_id}: {e}")
                    logger.error(f"Error processing {cell_id}: {e}")
                
                progress.advance(main_task)
                
        # Print summary table
        elapsed = time.time() - start_time
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Total Cells", str(num_cells))
        table.add_row("Processed", str(stats['processed']))
        table.add_row("Skipped", str(stats['skipped']))
        table.add_row("Errors", str(stats['errors']))
        table.add_row("Time Taken", f"{elapsed:.1f}s")
        console.print(table)
        
        if all_df_rows:
            df = pd.DataFrame(all_df_rows)
            out_pq = zarr_path.replace('.zarr', '_phase_features.parquet')
            df.to_parquet(out_pq)
            logger.info(f"Saved aggregated dataframe to {out_pq}")
            
    else:
        all_df_rows = []
        image_name = os.path.basename(zarr_path).replace("_crops.zarr", "")
        kwargs['image_name'] = image_name
        
        for cell_id in tqdm(cell_ids, desc="Processing Phase features", position=0, leave=True):
            try:
                res = process_cell(cells_group[cell_id], force_recompute=force, **kwargs)
                if isinstance(res, tuple):
                    all_df_rows.extend(res[1])
            except Exception as e:
                logger.error(f"Error processing {cell_id}: {e}")
                
        if all_df_rows:
            df = pd.DataFrame(all_df_rows)
            out_pq = zarr_path.replace('.zarr', '_phase_features.parquet')
            df.to_parquet(out_pq)
            logger.info(f"Saved aggregated dataframe to {out_pq}")
            
    return True
