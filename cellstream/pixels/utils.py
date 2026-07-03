import logging
import os
import pickle
import bz2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

logger = logging.getLogger(__name__)

def project_pixels(df, z_col, shape=None, fill_value=float('nan'), filename=None):
    """
    Projects values from a specific column of the pixel profiling DataFrame 
    back into a 2D image spatial layout using the 'x' and 'y' pixel coordinates.
    """
    if filename is not None and not df.empty and 'filename' in df.columns:
        df = df[df['filename'] == filename]
        
    if df.empty:
        h, w = (0, 0) if shape is None else shape
        return torch.full((h, w), fill_value, dtype=torch.float32)

    if shape is None:
        height = int(df['y'].max()) + 1
        width = int(df['x'].max()) + 1
    else:
        height, width = shape

    img = torch.full((height, width), fill_value, dtype=torch.float32)

    y_coords = torch.tensor(df['y'].values, dtype=torch.long)
    x_coords = torch.tensor(df['x'].values, dtype=torch.long)
    z_vals = torch.tensor(df[z_col].values, dtype=torch.float32)

    valid_mask = (y_coords >= 0) & (y_coords < height) & (x_coords >= 0) & (x_coords < width)
    img[y_coords[valid_mask], x_coords[valid_mask]] = z_vals[valid_mask]

    return img

def compute_2d_landscape(
    df,
    x_col='D',
    y_col='E',
    z_col='F_bin',
    bins=100,
    x_range=None,
    y_range=None,
    min_count=10,
    percentiles=(1.0, 99.0)
):
    """
    Computes 2D statistics (mean, std, count) for a target column (z_col) 
    binned by two coordinate columns (x_col, y_col).
    """
    x_vals = df[x_col].values
    y_vals = df[y_col].values
    z_vals = df[z_col].values
    
    if x_range is None:
        if percentiles is not None:
            x_range = (np.nanpercentile(x_vals, percentiles[0]), np.nanpercentile(x_vals, percentiles[1]))
        else:
            x_range = (np.nanmin(x_vals), np.nanmax(x_vals))
            
    if y_range is None:
        if percentiles is not None:
            y_range = (np.nanpercentile(y_vals, percentiles[0]), np.nanpercentile(y_vals, percentiles[1]))
        else:
            y_range = (np.nanmin(y_vals), np.nanmax(y_vals))
            
    stat_args = {
        'x': x_vals,
        'y': y_vals,
        'values': z_vals,
        'bins': bins,
        'range': [list(x_range), list(y_range)]
    }
    
    count_stat, x_edges, y_edges, _ = binned_statistic_2d(**stat_args, statistic='count')
    mean_stat, _, _, _ = binned_statistic_2d(**stat_args, statistic='mean')
    std_stat, _, _, _ = binned_statistic_2d(**stat_args, statistic='std')
    
    mean_stat = mean_stat.astype(float)
    std_stat = std_stat.astype(float)
    count_stat = count_stat.astype(float)
    
    mask = count_stat < min_count
    mean_stat[mask] = np.nan
    std_stat[mask] = np.nan
    count_stat[mask] = np.nan
    
    metadata = {
        'x_col': x_col,
        'y_col': y_col,
        'z_col': z_col,
        'x_range': x_range,
        'y_range': y_range,
        'bins': bins,
        'min_count': min_count
    }
    
    return {
        'mean': mean_stat,
        'std': std_stat,
        'count': count_stat,
        'edges': (x_edges, y_edges),
        'metadata': metadata
    }

def plot_2d_landscape(
    stats_dict,
    stat_name='mean',
    cmap='viridis',
    title=None,
    vmin=None,
    vmax=None,
    x_label=None,
    y_label=None,
    colorbar_label=None,
    nan_color='lightgrey',
    ax=None,
    figsize=(8, 6)
):
    """
    Plots a 2D landscape from the computed binned statistics. NaNs are rendered in grey.
    """
    data = stats_dict[stat_name]
    x_edges, y_edges = stats_dict['edges']
    metadata = stats_dict.get('metadata', {})
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
        
    try:
        current_cmap = plt.get_cmap(cmap).copy()
    except ValueError:
        current_cmap = plt.get_cmap('viridis').copy()
    current_cmap.set_bad(color=nan_color)
    
    mesh = ax.pcolormesh(
        x_edges, 
        y_edges, 
        data.T, 
        cmap=current_cmap, 
        vmin=vmin, 
        vmax=vmax
    )
    
    cb_lbl = colorbar_label if colorbar_label is not None else stat_name.capitalize()
    fig.colorbar(mesh, ax=ax, label=cb_lbl)
    
    x_lbl = x_label if x_label is not None else metadata.get('x_col', 'X')
    y_lbl = y_label if y_label is not None else metadata.get('y_col', 'Y')
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl)
    
    if title is None:
        z_lbl = metadata.get('z_col', 'Value')
        title = f"2D Landscape of {z_lbl} ({stat_name})"
    ax.set_title(title)
    
    return fig, ax

def save_landscape(data, filename="landscape_stats.pbz2"):
    """Compresses and saves statistic dict to a .pbz2 file."""
    logger.info(f"Compressing and saving to {filename}...")
    with bz2.BZ2File(filename, "w") as f:
        pickle.dump(data, f)
    logger.info(f"Done. File size: {os.path.getsize(filename) / 1024**2:.2f} MB")

def load_landscape(filename="landscape_stats.pbz2"):
    """Loads statistic dict from a compressed .pbz2 file."""
    logger.info(f"Loading {filename}...")
    with bz2.BZ2File(filename, "rb") as f:
        return pickle.load(f)
