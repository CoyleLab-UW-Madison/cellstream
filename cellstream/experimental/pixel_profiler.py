# -*- coding: utf-8 -*-
"""
Pixel profiling and frequency/amplitude landscape generation for microscopy timeseries data.
Part of the cellstream experimental suite.
"""

import os
import gc
import pickle
import bz2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.stats import binned_statistic_2d

from ..image import load_image
from ..fft import generate_fft_features
from ..registration import register_and_transform_image_timeseries


def profile_image_pixels(
    img,
    channel_names=None,
    c_val=35.0,
    min_bin=4,
    max_bin=40,
    filter_method='product',
    filter_channel=0,
    peak_constraint='exactly_one',
    device=None,
    max_fft_bin=50,
    fft_batch_size=250,
    filename_label="unknown"
):
    """
    Profiles pixel-level features from a single microscopy image time-series using cellstream.
    
    Parameters
    ----------
    img : torch.Tensor or np.ndarray
        The input image timeseries.
    channel_names : list or dict, optional
        Mapping from channel index to name. If a list is provided, index i maps to channel_names[i].
        If a dict is provided, index i maps to channel_names[i].
        Defaults to {0: 'E', 1: 'D'} if the image has exactly 2 channels, otherwise ['C0', 'C1', ...].
    c_val : float
        Z-score cutoff value for filtering. Default is 35.0.
    min_bin : int
        The lower frequency bin index for filtering and peak detection (inclusive). Default is 4.
    max_bin : int
        The upper frequency bin index for filtering and peak detection (exclusive). Default is 40.
    filter_method : str or callable
        The method used to calculate the filter statistic:
        - 'product': Product of Z-scores across all channels: torch.prod(z_score, dim=1).
        - 'min': Minimum Z-score across all channels (logical AND): torch.min(z_score, dim=1).
        - 'max': Maximum Z-score across all channels (logical OR): torch.max(z_score, dim=1).
        - 'channel': Z-score of a single channel specified by `filter_channel`.
        - custom callable: A function with signature f(z_score_tensor) -> stat_tensor of shape [F, 1, H, W].
    filter_channel : int
        Index of the channel to filter on when filter_method is 'channel'. Default is 0.
    peak_constraint : str
        Constraint on frequency bin threshold crossings to keep a pixel:
        - 'exactly_one': exactly 1 frequency bin in [min_bin:max_bin] exceeds c_val.
        - 'at_least_one': >= 1 frequency bin in [min_bin:max_bin] exceeds c_val.
        - 'none': no constraint is applied.
    device : str, optional
        Device to run the computation on ('cuda' or 'cpu'). Default is 'cuda' if GPU is available, else 'cpu'.
    max_fft_bin : int
        Maximum FFT bin parameter for cellstream.fft.generate_fft_features. Default is 50.
    fft_batch_size : int
        Batch size parameter for cellstream.fft.generate_fft_features. Default is 250.
    filename_label : str
        Name of the file to record in the output DataFrame. Defaults to "unknown".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the profiled pixel data.
    """
    # Ensure image is torch.Tensor
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
        
    # 2. Device detection
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # 3. Generate FFT features
    results = generate_fft_features(
        img,
        device=device,
        max_bin=max_fft_bin,
        batch_size=fft_batch_size
    )
    
    # 5. Parse channels
    # img shape is [T, C, H, W]
    num_channels = img.shape[1]
    if channel_names is None:
        if num_channels == 2:
            channel_map = {0: 'E', 1: 'D'}
        else:
            channel_map = {c: f'C{c}' for c in range(num_channels)}
    elif isinstance(channel_names, list):
        channel_map = {i: name for i, name in enumerate(channel_names)}
    else:
        channel_map = channel_names

    # 6. Apply filter logic
    z_score = results['z_score'] # [F, C, H, W]
    
    # Compute the filtering statistic across the channel dimension (dim=1)
    if callable(filter_method):
        compz = filter_method(z_score)
    elif filter_method == 'product':
        compz = torch.prod(z_score, dim=1, keepdim=True)
    elif filter_method == 'min':
        compz, _ = torch.min(z_score, dim=1, keepdim=True)
    elif filter_method == 'max':
        compz, _ = torch.max(z_score, dim=1, keepdim=True)
    elif filter_method == 'channel':
        compz = z_score[:, filter_channel:filter_channel+1, :, :]
    else:
        raise ValueError(f"Unknown filter_method: {filter_method}")
        
    # Slice the statistic to [min_bin:max_bin]
    compz_slice = compz[min_bin:max_bin] # Shape [N_bins, 1, H, W]
    
    # Create threshold clamps
    clamps = torch.where(compz_slice > c_val, 1, 0)
    
    # Apply peak constraints
    sum_clamps = clamps.sum(dim=0) # [1, H, W]
    if peak_constraint == 'exactly_one':
        testpix = torch.where(sum_clamps == 1, 1, 0)
    elif peak_constraint == 'at_least_one':
        testpix = torch.where(sum_clamps >= 1, 1, 0)
    elif peak_constraint == 'none':
        testpix = torch.ones_like(sum_clamps)
    else:
        raise ValueError(f"Unknown peak_constraint: {peak_constraint}")
        
    # Compute peak bin index (within min_bin:max_bin range)
    # Argmax on the sliced raw statistic is more robust and consistent across constraint methods
    fft_peak_bin = torch.argmax(compz_slice, dim=0) # [1, H, W]
    
    # Get mask indices
    mask_indices = torch.where(testpix == 1)
    
    # If no pixels pass, return empty DataFrame with appropriate schema
    if len(mask_indices[0]) == 0:
        schema = ['y', 'x', 'F_bin', 'filename']
        for name in channel_map.values():
            schema.extend([name, f'{name}_sd', f'{name}_amp', f'{name}_norm_amp', f'{name}_z'])
        return pd.DataFrame(columns=schema)
        
    # Extract coordinates (mask_indices is (Batch_dim, Y, X))
    y_coords = mask_indices[1].cpu().numpy()
    x_coords = mask_indices[2].cpu().numpy()
    
    # Extract frequency bin (add min_bin to find the true frequency bin index)
    f_bin = fft_peak_bin[mask_indices].cpu().numpy() + min_bin
    
    data_dict = {
        'y': y_coords,
        'x': x_coords,
        'F_bin': f_bin,
        'filename': filename_label
    }
    
    # Extract channel features
    # Slices are [min_bin:max_bin] for max calculations
    amp_max, _ = results['full_amplitude'][min_bin:max_bin].max(dim=0, keepdim=True)
    norm_amp_max, _ = results['normalized_amplitude'][min_bin:max_bin].max(dim=0, keepdim=True)
    z_max, _ = results['z_score'][min_bin:max_bin].max(dim=0, keepdim=True)
    
    for c, name in channel_map.items():
        # Intensities over time
        mean_val = img[:, c, :, :].mean(dim=0, keepdim=True)
        std_val = img[:, c, :, :].std(dim=0, keepdim=True)
        
        # Max amplitudes and Z-scores in the frequency band
        amp_val = amp_max[:, c, :, :]
        norm_amp_val = norm_amp_max[:, c, :, :]
        z_val = z_max[:, c, :, :]
        
        # Add to dictionary
        data_dict[name] = mean_val[mask_indices].cpu().numpy()
        data_dict[f'{name}_sd'] = std_val[mask_indices].cpu().numpy()
        data_dict[f'{name}_amp'] = amp_val[mask_indices].cpu().numpy()
        data_dict[f'{name}_norm_amp'] = norm_amp_val[mask_indices].cpu().numpy()
        data_dict[f'{name}_z'] = z_val[mask_indices].cpu().numpy()
        
    return pd.DataFrame(data_dict)


def batch_profile_pixels(
    file_paths,
    channel_names=None,
    c_val=35.0,
    min_bin=4,
    max_bin=40,
    filter_method='product',
    filter_channel=0,
    peak_constraint='exactly_one',
    register_images=True,
    registration_kwargs=None,
    device=None,
    max_fft_bin=50,
    fft_batch_size=250,
    show_progress=True
):
    """
    Profiles pixels for a batch of images and aggregates them into a single DataFrame.
    
    Parameters
    ----------
    file_paths : list of str/Path
        List of paths to images to process.
    register_images : bool
        Whether to register and transform the image timeseries before profiling. Default is True.
    registration_kwargs : dict, optional
        Optional dictionary of keyword arguments passed to register_and_transform_image_timeseries.
    All other parameters match those of profile_image_pixels.
    show_progress : bool
        If True, prints progress updates. Uses `progressbar` package if available.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with all profiled pixels.
    """
    all_data_frames = []
    
    # Attempt to set up a progress bar
    progressbar_available = False
    if show_progress:
        try:
            import progressbar
            bar = progressbar.ProgressBar(max_value=len(file_paths))
            progressbar_available = True
        except ImportError:
            pass

    for i, fp in enumerate(file_paths):
        fp_path = Path(fp)
        if show_progress and not progressbar_available:
            print(f"[{i+1}/{len(file_paths)}] Processing: {fp_path.name} ...")
            
        try:
            loaded_img = load_image(str(fp_path))
            if register_images:
                reg_kwargs = registration_kwargs or {}
                loaded_img = register_and_transform_image_timeseries(loaded_img, **reg_kwargs)
            df = profile_image_pixels(
                img=loaded_img,
                channel_names=channel_names,
                c_val=c_val,
                min_bin=min_bin,
                max_bin=max_bin,
                filter_method=filter_method,
                filter_channel=filter_channel,
                peak_constraint=peak_constraint,
                device=device,
                max_fft_bin=max_fft_bin,
                fft_batch_size=fft_batch_size,
                filename_label=fp_path.name
            )
            if not df.empty:
                all_data_frames.append(df)
        except Exception as e:
            print(f"Error processing {fp_path.name}: {e}")
            
        # Memory Cleanup after each image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if show_progress and progressbar_available:
            bar.update(i + 1)

    if show_progress and progressbar_available:
        bar.finish()

    if all_data_frames:
        if show_progress:
            print("Consolidating all data...")
        return pd.concat(all_data_frames, ignore_index=True)
    else:
        return pd.DataFrame()


def project_pixels(df, z_col, shape=None, fill_value=float('nan'), filename=None):
    """
    Projects values from a specific column of the pixel profiling DataFrame 
    back into a 2D image spatial layout using the 'x' and 'y' pixel coordinates.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the profiled pixel data. Must have 'x' and 'y' columns.
    z_col : str
        The column name of the feature to project (e.g., 'F_bin', 'E', 'D_amp').
    shape : tuple of int, optional
        The desired (height, width) of the output image tensor. If None, it is automatically
        determined from the maximum y and x coordinates in the DataFrame.
    fill_value : float, optional
        The default value to fill coordinates that have no corresponding row in the DataFrame.
        Default is float('nan').
    filename : str, optional
        If provided, filters the DataFrame to only include rows matching this filename.
        
    Returns
    -------
    torch.Tensor
        A 2D PyTorch tensor of shape (height, width) containing the projected values.
    """
    if filename is not None and not df.empty and 'filename' in df.columns:
        df = df[df['filename'] == filename]
        
    if df.empty:
        h, w = (0, 0) if shape is None else shape
        return torch.full((h, w), fill_value, dtype=torch.float32)

    # Determine height and width
    if shape is None:
        height = int(df['y'].max()) + 1
        width = int(df['x'].max()) + 1
    else:
        height, width = shape

    # Create empty tensor filled with the fill_value
    img = torch.full((height, width), fill_value, dtype=torch.float32)

    # Extract coordinates and values
    y_coords = torch.tensor(df['y'].values, dtype=torch.long)
    x_coords = torch.tensor(df['x'].values, dtype=torch.long)
    z_vals = torch.tensor(df[z_col].values, dtype=torch.float32)

    # Filter coordinates that fall within the image boundary
    valid_mask = (y_coords >= 0) & (y_coords < height) & (x_coords >= 0) & (x_coords < width)
    
    # Assign values to coordinates
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
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the pixel profile data.
    x_col : str
        Column name for the X-axis coordinate (default 'D').
    y_col : str
        Column name for the Y-axis coordinate (default 'E').
    z_col : str
        Column name for the value to aggregate (default 'F_bin').
    bins : int or tuple of int
        The number of bins along each axis. Default is 100.
    x_range : tuple of float, optional
        The range [min, max] for the X-axis bins. If None and percentiles is provided,
        computed using the specified percentiles of the data.
    y_range : tuple of float, optional
        The range [min, max] for the Y-axis bins. If None and percentiles is provided,
        computed using the specified percentiles of the data.
    min_count : int
        Minimum number of counts in a bin to keep statistics. Bins with fewer counts
        will have mean and std set to NaN. Default is 10.
    percentiles : tuple of float, optional
        Percentiles to compute range if x_range/y_range are None. Default is (1.0, 99.0).
        Set to None to use full min/max range.

    Returns
    -------
    dict
        A dictionary containing:
        - 'mean': 2D array of binned mean values
        - 'std': 2D array of binned std values
        - 'count': 2D array of binned counts
        - 'edges': Tuple of (x_edges, y_edges) arrays defining bin boundaries
        - 'metadata': Dictionary with columns, ranges, bins info
    """
    # Define coordinates and value to aggregate
    x_vals = df[x_col].values
    y_vals = df[y_col].values
    z_vals = df[z_col].values
    
    # Compute ranges if not provided
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
    
    # Calculate statistics using scipy
    count_stat, x_edges, y_edges, _ = binned_statistic_2d(**stat_args, statistic='count')
    mean_stat, _, _, _ = binned_statistic_2d(**stat_args, statistic='mean')
    std_stat, _, _, _ = binned_statistic_2d(**stat_args, statistic='std')
    
    # Cast to float to support NaN values
    mean_stat = mean_stat.astype(float)
    std_stat = std_stat.astype(float)
    count_stat = count_stat.astype(float)
    
    # Mask bins with low counts
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
    
    Parameters
    ----------
    stats_dict : dict
        The dictionary returned by compute_2d_landscape.
    stat_name : str
        The statistic to plot ('mean', 'std', or 'count'). Default is 'mean'.
    cmap : str
        Colormap name. Default is 'viridis'.
    title : str, optional
        Title of the plot. Defaults to "2D Landscape (mean/std/count of z_col)".
    vmin : float, optional
        Minimum value for the colormap.
    vmax : float, optional
        Maximum value for the colormap.
    x_label : str, optional
        Label for the X-axis. Defaults to x_col from metadata.
    y_label : str, optional
        Label for the Y-axis. Defaults to y_col from metadata.
    colorbar_label : str, optional
        Label for the colorbar. Defaults to stat_name.
    nan_color : str
        Color used to render masked (NaN) bins. Default is 'lightgrey'.
    ax : matplotlib.axes.Axes, optional
        An existing axes object to plot onto. If None, a new figure and axes are created.
    figsize : tuple of float
        Figure dimensions. Default is (8, 6).
        
    Returns
    -------
    fig, ax
        The matplotlib Figure and Axes objects.
    """
    data = stats_dict[stat_name]
    x_edges, y_edges = stats_dict['edges']
    metadata = stats_dict.get('metadata', {})
    
    # 1. Create or retrieve plot canvas
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
        
    # 2. Set up bad (NaN) color handling for colormap
    try:
        current_cmap = plt.get_cmap(cmap).copy()
    except ValueError:
        # Fallback to viridis if colormap is not found
        current_cmap = plt.get_cmap('viridis').copy()
    current_cmap.set_bad(color=nan_color)
    
    # 3. Create the 2D grid plot
    # data is of shape [bins_x, bins_y]. We transpose it so it plots with x on horiz, y on vert.
    mesh = ax.pcolormesh(
        x_edges, 
        y_edges, 
        data.T, 
        cmap=current_cmap, 
        vmin=vmin, 
        vmax=vmax
    )
    
    # 4. Colorbar
    cb_lbl = colorbar_label if colorbar_label is not None else stat_name.capitalize()
    fig.colorbar(mesh, ax=ax, label=cb_lbl)
    
    # 5. Axis Labels
    x_lbl = x_label if x_label is not None else metadata.get('x_col', 'X')
    y_lbl = y_label if y_label is not None else metadata.get('y_col', 'Y')
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl)
    
    # 6. Title
    if title is None:
        z_lbl = metadata.get('z_col', 'Value')
        title = f"2D Landscape of {z_lbl} ({stat_name})"
    ax.set_title(title)
    
    return fig, ax


def save_stats(data, filename="landscape_stats.pbz2"):
    """Compresses and saves statistic dict to a .pbz2 file."""
    print(f"Compressing and saving to {filename}...")
    with bz2.BZ2File(filename, "w") as f:
        pickle.dump(data, f)
    print(f"Done. File size: {os.path.getsize(filename) / 1024**2:.2f} MB")


def load_stats(filename="landscape_stats.pbz2"):
    """Loads statistic dict from a compressed .pbz2 file."""
    print(f"Loading {filename}...")
    with bz2.BZ2File(filename, "rb") as f:
        return pickle.load(f)
