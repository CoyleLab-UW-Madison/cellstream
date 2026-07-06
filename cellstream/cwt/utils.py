"""
cellstream.cwt.utils

Low-level utilities for cwt image processing.
@authors: coylelab

Functions:
- query_cwt_block: block processor that generates cwts and queries data lines with a carrier signals.
        When bank_method='max_pool':
            Adaptive max pooling maps the range of selected scales (min_scale, max_scale) to num_filter_banks.
            Within each bank, we store amplitude (maxes) and scale (indices) for the local maximum in the carrier channel.
            Data lines are queried (phase, amp, ...) at these locations and used to generate the associted stream tensor.
        When bank_method='sort'
            Carrier amplitudes are sorted to locate the num_filter_banks peaks.
            Data lines are queried at these locations to generate the associated stream tensor as above.
        For num_banks=1 these methods perform the same and use the dominant carrier signal to query the other lines.

- generate_cwt_image_cellstreams: performs the above processing through blocked processing of the entire image

- extract_cwt_cellstreams: fast extraction of cwt_features timeseries using label image tracks.

"""

import logging
logger = logging.getLogger(__name__)
import os
import warnings
import numpy as np
from tqdm.auto import tqdm
import torch

from ..utils import downsample, normalize_dims, normalize_histogram as norm_hist
from ..analysis import extract_single_cell_data


def query_cwt_block(
    data,
    min_scale=80,
    max_scale=180,
    num_filter_banks=1,
    normalize_amplitudes=False,
    carrier_channel=None,
    channel_outputs=None,
    use_gpu=False,
    bank_method="max_pool",
    sampling=None,
    **ssqueezepy_cwt_kwargs,
):
    """
    Process a block of data using CWT with selective outputs
    Args:
        data: Input tensor of shape (N, T) where N=number of pixels, T=timepoints
        outputs: Tuple of requested outputs ('amp', 'freq', 'phase')
        bank_method: describes how filterbanks will be constructed from the carrier signal
                'max_pool' -> applies adaptive max pooling to collapse the scales down to num_filter_banks
                'sort' -> take the top num_filter_banks amplitude peaks
        **kwargs: Forwarded to ssqueezepy.cwt
    Returns:
        Dictionary of requested outputs
    """

    # Force environment variable BEFORE import
    os.environ["SSQ_GPU"] = "1" if use_gpu else "0"
    from ssqueezepy import cwt

    if channel_outputs is None:
        channel_outputs = {0: ["amp", "freq", "phase"]}

    # Prepare shapes
    BATCH_SIZE, C, T = data.shape

    # prepare channel-specific containers
    split_channels = {}
    split_channels_full_power_sums = {}
    split_channels_full_means = {}
    split_channels_full_std = {}

    for channel in channel_outputs:
        # Compute CWT with forwarded parameters
        Twx, scales = cwt(data[:, channel, :], **ssqueezepy_cwt_kwargs)
        if isinstance(Twx, np.ndarray):
            Twx = torch.tensor(Twx)
        # Always compute full power sums for normalized outputs
        split_channels_full_power_sums[channel] = Twx.abs().sum(
            axis=1, keepdims=True
        )
        if "z_score" in channel_outputs[channel]:
            split_channels_full_means[channel] = Twx.abs().mean(axis=1, keepdims=True)
            split_channels_full_std[channel] = Twx.abs().std(axis=1, keepdims=True)
        Twx_sub = Twx[:, min_scale:max_scale, :].clone()
        split_channels[channel] = Twx_sub

    # Find max_scale channels along scale dimension

    
    # Helper to extract carrier from a given channel
    def _extract_carrier(ch):
        amp = split_channels[ch].abs()
        phase = split_channels[ch].angle()
        if bank_method == "max_pool":
            max_pooler = torch.nn.AdaptiveMaxPool1d(num_filter_banks, return_indices=True)
            amp = amp.permute(0, 2, 1)
            amp, freq = max_pooler(amp)
            amp = amp.permute(0, 2, 1)
            freq = freq.permute(0, 2, 1)
            phase = torch.gather(phase, 1, freq)
        elif bank_method == "sort":
            amp, freq = torch.sort(amp, axis=1, descending=True)
            phase = torch.gather(phase, 1, freq)
        return amp, freq, phase

    global_carrier_amp, global_carrier_freq, global_carrier_phase = None, None, None
    if carrier_channel is not None:
        global_carrier_amp, global_carrier_freq, global_carrier_phase = _extract_carrier(carrier_channel)

    results = {c: {} for c in range(C)}


    freqs_lookup = None
    if sampling is not None:
        wavelet = ssqueezepy_cwt_kwargs.get("wavelet", "gmw")
        from ssqueezepy.experimental import scale_to_freq
        fs = sampling["fs"]
        N = sampling["N"]
        blank_series = torch.ones(T)
        Twx, scales = cwt(blank_series, **ssqueezepy_cwt_kwargs)
        freqs_lookup = scale_to_freq(scales, wavelet=wavelet, N=N, fs=fs)
        freqs_lookup = torch.from_numpy(freqs_lookup.astype("float32")).broadcast_to(BATCH_SIZE, T, -1)
        freqs_lookup = freqs_lookup.permute(0, 2, 1)

    for channel, returns in channel_outputs.items():
        if carrier_channel is None:
            ch_carrier_amp, ch_carrier_freq, ch_carrier_phase = _extract_carrier(channel)
        else:
            ch_carrier_amp, ch_carrier_freq, ch_carrier_phase = global_carrier_amp, global_carrier_freq, global_carrier_phase

        if sampling is not None:
            fl = freqs_lookup.to(ch_carrier_freq.device)
            ch_carrier_freq_converted = torch.gather(fl, 1, ch_carrier_freq + min_scale)

        P = split_channels[channel].abs()
        # Always compute normalized power if requested or if legacy flag is True
        if normalize_amplitudes or "normalized_amp" in returns or "normalized_amplitude" in returns:
            P_norm = P / split_channels_full_power_sums[channel]
            
        if normalize_amplitudes:
            P = P_norm
        
        if ("phase" in returns) or ("phase_difference" in returns):
            PH = split_channels[channel].angle()
            ch_ph = torch.gather(PH, 1, ch_carrier_freq)
            
        if "phase" in returns:
            results[channel]["phase"] = ch_ph[:, :num_filter_banks, :].cpu()
            
        if "phase_difference" in returns:
            if carrier_channel is None:
                # If no carrier is specified, phase difference against itself is 0
                results[channel]["phase_difference"] = torch.zeros_like(ch_ph[:, :num_filter_banks, :]).cpu()
            else:
                results[channel]["phase_difference"] = (((ch_ph - ch_carrier_phase) % (2 * torch.pi)).abs())[:, :num_filter_banks, :].cpu()
                
        if "amp" in returns:
            ch_p = torch.gather(P, 1, ch_carrier_freq)
            results[channel]["amp"] = ch_p[:, :num_filter_banks, :].cpu()
            
        if "normalized_amp" in returns or "normalized_amplitude" in returns:
            ch_p_norm = torch.gather(P_norm, 1, ch_carrier_freq)
            if "normalized_amp" in returns:
                results[channel]["normalized_amp"] = ch_p_norm[:, :num_filter_banks, :].cpu()
            if "normalized_amplitude" in returns:
                results[channel]["normalized_amplitude"] = ch_p_norm[:, :num_filter_banks, :].cpu()
            
        if "z_score" in returns:
            ch_p = torch.gather(P, 1, ch_carrier_freq)
            z = (ch_p - split_channels_full_means[channel]) / split_channels_full_std[channel]
            results[channel]["z_score"] = z[:, :num_filter_banks, :].cpu()
            
        if "freq" in returns:
            if sampling is not None:
                results[channel]["freq"] = ch_carrier_freq_converted[:, :num_filter_banks, :].cpu()
            else:
                results[channel]["freq"] = (ch_carrier_freq[:, :num_filter_banks, :].cpu() + min_scale)

    return results

def _infer_blocks(
    img_shape,
    use_gpu,
    channel_outputs,
    buffer_fraction=0.5,
    **ssqueezepy_cwt_kwargs,
):
    """
    Infers the number of blocks to use for processing based on available memory.
    """
    T, C, X, Y = img_shape
    total_pixels = X * Y

    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                total_mem = torch.cuda.get_device_properties(0).total_memory
                available_mem = total_mem - torch.cuda.memory_allocated(0)
                mem_to_use = available_mem * buffer_fraction
            else:
                warnings.warn(
                    "GPU not available, falling back to default block size of 10."
                )
                return 10
        except ImportError:
            warnings.warn("PyTorch not found, falling back to default block size of 10.")
            return 10
    else:
        try:
            import psutil
            available_mem = psutil.virtual_memory().available
            mem_to_use = available_mem * buffer_fraction
        except ImportError:
            warnings.warn(
                "psutil not found. Cannot infer block size for CPU. "
                "Please install psutil (`pip install psutil`) for this feature, "
                "or specify `blocks` manually. Falling back to default block size of 10."
            )
            return 10


    os.environ["SSQ_GPU"] = "1" if use_gpu else "0"
    try:
        from ssqueezepy import cwt
        import torch

        dummy_input = torch.zeros(T)
        if use_gpu:
            if torch.cuda.is_available():
                dummy_input = dummy_input.to("cuda")
            else:  # handles case where torch is installed but cuda not available
                warnings.warn(
                    "GPU not available, falling back to default block size of 10."
                )
                return 10

        _, scales = cwt(dummy_input, **ssqueezepy_cwt_kwargs)
        num_scales = len(scales)
        
        # ssqueezepy typically pads the sequence length to the next power of 2 internally for FFTs
        padded_T = 2 ** int(np.ceil(np.log2(T)))

        # Memory for one pixel's CWT result inside ssqueezepy (complex64 -> 8 bytes)
        mem_per_pixel_cwt = num_scales * padded_T * 8  
        
        num_channels_to_process = len(channel_outputs)

        # CWT peak memory usage per pixel (approximate):
        # 1. We compute CWT for one channel at a time: `mem_per_pixel_cwt * 2` (to account for PyTorch intermediate FFT arrays)
        # 2. We keep a subset of scales (`Twx_sub`) for ALL channels in `split_channels`: roughly `num_scales * T * 8 * num_channels_to_process`
        total_mem_per_pixel = (mem_per_pixel_cwt * 2) + (num_scales * T * 8 * num_channels_to_process)

        # Max pixels we can process in a single block
        max_pixels_in_block = mem_to_use / (total_mem_per_pixel + 1e-9)

        # How many blocks we need (minimum 1)
        num_blocks = total_pixels / (max_pixels_in_block + 1e-9)
        num_blocks = max(1, num_blocks)

        return int(np.ceil(num_blocks))

    except ImportError:
        warnings.warn("ssqueezepy not found, falling back to default block size of 10.")
        return 10

def generate_cwt_image_cellstreams(
    img,
    min_scale=80,
    max_scale=180,
    num_filter_banks=1,
    normalize_amplitudes=False,
    blocks=10,
    use_gpu=False,
    bank_method="max_pool",
    downsample_by=None,
    normalize_histogram=True,
    mean_center=False,
    carrier_channel=None,
    channel_names=None,
    channel_outputs=None,
    sampling=None,
    return_timeseries=False,
    **ssqueezepy_cwt_kwargs,
):
    """
    Process image in blocks with selective outputs and kwargs forwarding
    Args:
        img: Input array (T, X, Y)
        outputs:
        return_timeseries: If True, adds the preprocessed timeseries to the results dictionary.
        **kwargs: Forwarded to ssqueezepy.cwt
    Returns:
        Dictionary of requested outputs as numpy arrays
    """
    # gpu environment setup for squeezepy
    os.environ["SSQ_GPU"] = "1" if use_gpu else "0"

    if channel_outputs is None:
        channel_outputs = {0: ["amp", "freq", "phase"]}

    img = normalize_dims(img, 1)

    if sampling is not None:
        pass

    # image pre-processing
    if downsample_by is not None:
        logger.info(f"Downsampling image by {downsample_by} ...")
        img = downsample(img, downsample_by)

    if normalize_histogram is not False:
        logger.info("Performing histogram normalization on image ...")
        img = norm_hist(img)

    if mean_center is not False:
        logger.info("Mean centering timeseries ...")
        img = img - img.mean(axis=0)

    if blocks == "auto":
            blocks = _infer_blocks(
                img.shape,
                use_gpu=use_gpu,
                channel_outputs=channel_outputs,
                **ssqueezepy_cwt_kwargs,
            )
            logger.info(f"Automatically determined block size: {blocks}")

    preprocessed_timeseries = img
    # reshape image for blocked processing
    T, C, X, Y = img.shape
    img = img.reshape(T, C, X * Y).permute(2, 1, 0)  # shape is now (x*y,c,t)

    # pre-allocate outputs
    logger.info("Pre-allocating output arrays...")
    final = {c: {} for c in channel_outputs}
    for c in channel_outputs:
        for k in channel_outputs[c]:
            final[c][k] = torch.zeros((X * Y, num_filter_banks, T), dtype=torch.float32)

    # setup blocked processing loop parameters
    total_pixels = X * Y
    blocks = min(blocks, total_pixels)
    block_size = total_pixels // blocks
    remainder = total_pixels % blocks

    logger.info("Generating CWT cellstreams")
    cursor = 0  # position in block to process

    for b in tqdm(range(blocks)):
        this_block_size = block_size + (1 if b < remainder else 0)
        end = cursor + this_block_size
        block = img[cursor:end]  # (this_block_size, C, T)

        block_result = query_cwt_block(
            block,
            min_scale=min_scale,
            max_scale=max_scale,
            num_filter_banks=num_filter_banks,
            normalize_amplitudes=normalize_amplitudes,
            carrier_channel=carrier_channel,
            channel_outputs=channel_outputs,
            use_gpu=use_gpu,
            sampling=sampling,
            **ssqueezepy_cwt_kwargs,
        )
        # Fill in preallocated tensors
        for c in channel_outputs:
            for k in channel_outputs[c]:
                val = block_result[c][k]  # (this_block_size, num_filter_banks, T)
                final[c][k][cursor:end] = val
        cursor = end

    # reshape
    for c in channel_outputs:
        for k in channel_outputs[c]:
            final[c][k] = (
                final[c][k].permute(2, 1, 0).reshape(T, num_filter_banks, X, Y)
            )

    # adjust to match channel names if need be
    if channel_names is not None:
        result = {channel_names[idx]: outdict for idx, outdict in final.items()}
    else:
        result = final

    if return_timeseries:
        result["timeseries"] = preprocessed_timeseries

    attrs = {
        "min_scale": min_scale,
        "max_scale": max_scale,
        "num_filter_banks": num_filter_banks,
        "normalize_amplitudes": normalize_amplitudes,
        "blocks": blocks,
        "use_gpu": use_gpu,
        "bank_method": bank_method,
        "downsample_by": downsample_by,
        "normalize_histogram": normalize_histogram,
        "mean_center": mean_center,
        "carrier_channel": carrier_channel,
        "channel_names": channel_names,
        "channel_outputs": channel_outputs,
        "sampling": sampling,
        "return_timeseries": return_timeseries,
    }
    for k, v in ssqueezepy_cwt_kwargs.items():
        attrs[k] = v

    result["_attrs"] = attrs
    return result


def extract_cwt_cellstreams(features, track_masks):
    """extract single-cell trajectories using label_image tracks"""
    try:
        from torch_scatter import scatter_mean, scatter_std
    except ImportError:
        raise ImportError(
            "torch-scatter is required for single-cell extraction. "
            "Install it following: https://github.com/rusty1s/pytorch_scatter"
        )

    # reshape features
    if features.dim() == 3:
        logger.info("3 channel image detected; unsqueezing C dimension...")
        features = features.unsqueeze(1)
    T, C, X, Y = features.shape
    features = features.reshape(T, C, -1)

    # reshape masks
    if track_masks.dim() == 2:  # static 2D mask
        track_masks = track_masks.broadcast_to(T, C, X, Y)
    elif track_masks.dim() == 3:
        # timeseries mask (T,X,Y)
        track_masks = track_masks.broadcast_to(C, T, X, Y)
        track_masks = track_masks.permute(1, 0, 2, 3)  # (T,C,X)
    track_masks = track_masks.reshape(T, C, -1)

    num_masks = int(track_masks.max().item()) + 1

    cellstreams_mean = scatter_mean(
        features, track_masks, dim=-1, dim_size=num_masks
    )  # T,C,num_masks
    cellstreams_mean = cellstreams_mean.permute(2, 1, 0)

    cellstreams_std = scatter_std(features, track_masks, dim=-1, dim_size=num_masks)
    cellstreams_std = cellstreams_std.permute(2, 1, 0)
    return cellstreams_mean, cellstreams_std
