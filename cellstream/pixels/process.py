import logging
import gc
import torch
import pandas as pd
from pathlib import Path

from ..image import load_image
from ..fft import generate_fft_features
from ..registration import register_and_transform_image_timeseries

logger = logging.getLogger(__name__)

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
    """
    #gather metadata
    run_metadata = locals().copy()
    if hasattr(img, 'shape'):
        run_metadata['img'] = f"Tensor/Array(shape={list(img.shape)})"
    else:
        run_metadata['img'] = "Unknown Image Format"

    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
        
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    results = generate_fft_features(
        img,
        device=device,
        max_bin=max_fft_bin,
        batch_size=fft_batch_size
    )
    
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

    z_score = results['z_score']
    
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
        
    compz_slice = compz[min_bin:max_bin]
    clamps = torch.where(compz_slice > c_val, 1, 0)
    
    sum_clamps = clamps.sum(dim=0)
    if peak_constraint == 'exactly_one':
        testpix = torch.where(sum_clamps == 1, 1, 0)
    elif peak_constraint == 'at_least_one':
        testpix = torch.where(sum_clamps >= 1, 1, 0)
    elif peak_constraint == 'none':
        testpix = torch.ones_like(sum_clamps)
    else:
        raise ValueError(f"Unknown peak_constraint: {peak_constraint}")
        
    fft_peak_bin = torch.argmax(compz_slice, dim=0)
    mask_indices = torch.where(testpix == 1)
    
    if len(mask_indices[0]) == 0:
        schema = ['y', 'x', 'F_bin', 'filename']
        for name in channel_map.values():
            schema.extend([name, f'{name}_sd', f'{name}_amp', f'{name}_norm_amp', f'{name}_z'])
        return pd.DataFrame(columns=schema)
        
    y_coords = mask_indices[1].cpu().numpy()
    x_coords = mask_indices[2].cpu().numpy()
    f_bin = fft_peak_bin[mask_indices].cpu().numpy() + min_bin
    
    data_dict = {
        'y': y_coords,
        'x': x_coords,
        'F_bin': f_bin,
        'filename': filename_label
    }
    
    amp_max, _ = results['full_amplitude'][min_bin:max_bin].max(dim=0, keepdim=True)
    norm_amp_max, _ = results['normalized_amplitude'][min_bin:max_bin].max(dim=0, keepdim=True)
    z_max, _ = results['z_score'][min_bin:max_bin].max(dim=0, keepdim=True)
    
    for c, name in channel_map.items():
        mean_val = img[:, c, :, :].mean(dim=0, keepdim=True)
        std_val = img[:, c, :, :].std(dim=0, keepdim=True)
        
        amp_val = amp_max[:, c, :, :]
        norm_amp_val = norm_amp_max[:, c, :, :]
        z_val = z_max[:, c, :, :]
        
        data_dict[name] = mean_val[mask_indices].cpu().numpy()
        data_dict[f'{name}_sd'] = std_val[mask_indices].cpu().numpy()
        data_dict[f'{name}_amp'] = amp_val[mask_indices].cpu().numpy()
        data_dict[f'{name}_norm_amp'] = norm_amp_val[mask_indices].cpu().numpy()
        data_dict[f'{name}_z'] = z_val[mask_indices].cpu().numpy()
    
    result = pd.DataFrame(data_dict)
    result.attrs = run_metadata
    return result

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
    """
    run_metadata = locals().copy()
    all_data_frames = []
    
    tqdm_available = False
    if show_progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=len(file_paths))
            tqdm_available = True
        except ImportError:
            pass
    
    if isinstance(file_paths, str):
        logger.info(f"Converting {file_paths} to list of paths...")
        file_paths = [path for path in Path(file_paths).iterdir() if path.is_file()]
    
    for i, fp in enumerate(file_paths):
        fp_path = Path(fp)
        if show_progress and not tqdm_available:
            logger.info(f"[{i+1}/{len(file_paths)}] Processing: {fp_path.name} ...")
            
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
            logger.error(f"Error processing {fp_path.name}: {e}")
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if show_progress and tqdm_available:
            bar.update(i + 1)

    if show_progress and tqdm_available:
        bar.close()

    if all_data_frames:
        if show_progress:
            logger.info("Consolidating all data...")
        result = pd.concat(all_data_frames, ignore_index=True)
    else:
        result = pd.DataFrame()

    result.attrs = run_metadata
    return result
