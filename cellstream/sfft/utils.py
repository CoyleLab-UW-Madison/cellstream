import torch
import numpy as np
import warnings
import os
from tqdm.auto import tqdm
import logging
logger = logging.getLogger(__name__)

from cellstream.utils import downsample, normalize_histogram as norm_hist, normalize_dims

def query_stft_block(
    data,
    min_bin=0,
    max_bin=100,
    num_filter_banks=1,
    normalize_amplitudes=False,
    carrier_channel=0,
    channel_outputs=None,
    use_gpu=False,
    bank_method="max_pool",
    sampling=None,
    n_fft=256,
    win_length=None,
    **torch_stft_kwargs,
):
    if channel_outputs is None:
        channel_outputs = {0: ["amp", "freq", "phase"]}

    BATCH_SIZE, C, T = data.shape

    split_channels = {}
    split_channels_full_power_sums = {}
    split_channels_full_means = {}
    split_channels_full_std = {}

    window = torch_stft_kwargs.pop('window', None)
    if window is None:
        win_len = win_length if win_length is not None else n_fft
        window = torch.hann_window(win_len).to(data.device)

    for channel in channel_outputs:
        input_data = data[:, channel, :]
        
        Twx = torch.stft(
            input_data, 
            n_fft=n_fft, 
            hop_length=1, 
            win_length=win_length, 
            window=window, 
            center=True, 
            pad_mode='reflect', 
            return_complex=True,
            **torch_stft_kwargs
        )
        
        Twx = Twx[..., :T]
        
        if normalize_amplitudes:
            split_channels_full_power_sums[channel] = Twx.abs().sum(
                axis=1, keepdims=True
            )
        if "z_score" in channel_outputs[channel]:
            split_channels_full_means[channel] = Twx.abs().mean(axis=1, keepdims=True)
            split_channels_full_std[channel] = Twx.abs().std(axis=1, keepdims=True)
        Twx_sub = Twx[:, min_bin:max_bin, :].clone()
        split_channels[channel] = Twx_sub

    carrier_amp = split_channels[carrier_channel].abs()
    carrier_phase = split_channels[carrier_channel].angle()

    if bank_method == "max_pool":
        max_pooler = torch.nn.AdaptiveMaxPool1d(num_filter_banks, return_indices=True)
        carrier_amp = carrier_amp.permute(0, 2, 1)
        carrier_amp, carrier_freq = max_pooler(carrier_amp)
        carrier_amp = carrier_amp.permute(0, 2, 1)
        carrier_freq = carrier_freq.permute(0, 2, 1)
        carrier_phase = torch.gather(carrier_phase, 1, carrier_freq)
    elif bank_method == "sort":
        carrier_amp, carrier_freq = torch.sort(
            split_channels[carrier_channel].abs(), axis=1, descending=True
        )
        carrier_phase = split_channels[carrier_channel].angle()
        carrier_phase = torch.gather(carrier_phase, 1, carrier_freq)

    results = {c: {} for c in range(C)}

    if sampling is not None:
        fs = sampling["fs"]
        N = sampling["N"]
        
        num_bins = (n_fft // 2) + 1
        freqs_lookup = np.linspace(0, fs/2, num_bins)
        freqs_lookup = torch.from_numpy(freqs_lookup.astype("float32")).broadcast_to(
            BATCH_SIZE, T, -1
        )
        freqs_lookup = freqs_lookup.permute(0, 2, 1)
        freqs_lookup = freqs_lookup.to(carrier_freq.device)
        carrier_freq_converted = torch.gather(freqs_lookup, 1, carrier_freq + min_bin)

    for channel, returns in channel_outputs.items():
        P = split_channels[channel].abs()
        if normalize_amplitudes:
            P = P / split_channels_full_power_sums[channel]
        if ("phase" in returns) or ("phase_difference" in returns):
            PH = split_channels[channel].angle()
            ch_ph = torch.gather(PH, 1, carrier_freq)
        if "phase" in returns:
            results[channel]["phase"] = ch_ph[:, :num_filter_banks, :].cpu()
        if "phase_difference" in returns:
            results[channel]["phase_difference"] = (
                ((ch_ph - carrier_phase) % (2 * torch.pi)).abs()
            )[:, :num_filter_banks, :].cpu()
        if "amp" in returns:
            ch_p = torch.gather(P, 1, carrier_freq)
            results[channel]["amp"] = ch_p[:, :num_filter_banks, :].cpu()
        if "z_score" in returns:
            if "amp" in returns:
                z = (
                    ch_p - split_channels_full_means[channel]
                ) / split_channels_full_std[channel]
            else:
                ch_p = torch.gather(P, 1, carrier_freq)
                z = (
                    ch_p - split_channels_full_means[channel]
                ) / split_channels_full_std[channel]
            results[channel]["z_score"] = z[:, :num_filter_banks, :].cpu()
        if "freq" in returns:
            if sampling is not None:
                results[channel]["freq"] = carrier_freq_converted[
                    :, :num_filter_banks, :
                ].cpu()
            else:
                results[channel]["freq"] = (
                    carrier_freq[:, :num_filter_banks, :].cpu() + min_bin
                )

    return results

def _infer_blocks(
    img_shape,
    use_gpu,
    channel_outputs,
    buffer_fraction=0.5,
    n_fft=256,
    **torch_stft_kwargs,
):
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
            warnings.warn("psutil not found, falling back to 10")
            return 10

    import torch
    num_bins = (n_fft // 2) + 1
    
    mem_per_pixel_stft = num_bins * T * 8  
    num_channels_to_process = len(channel_outputs)
    total_mem_per_pixel = (mem_per_pixel_stft * 2) + (num_bins * T * 8 * num_channels_to_process)
    max_pixels_in_block = mem_to_use / (total_mem_per_pixel + 1e-9)
    num_blocks = total_pixels / (max_pixels_in_block + 1e-9)
    num_blocks = max(1, num_blocks)

    return int(np.ceil(num_blocks))


def generate_stft_image_cellstreams(
    img,
    n_fft=256,
    min_bin=0,
    max_bin=100,
    num_filter_banks=1,
    normalize_amplitudes=False,
    blocks=10,
    use_gpu=False,
    bank_method="max_pool",
    downsample_by=None,
    normalize_histogram=True,
    mean_center=False,
    carrier_channel=0,
    channel_names=None,
    channel_outputs=None,
    sampling=None,
    return_timeseries=False,
    **torch_stft_kwargs,
):
    """
    Process image in blocks with selective outputs and kwargs forwarding
    Args:
        img: Input array (T, X, Y)
        outputs:
        return_timeseries: If True, adds the preprocessed timeseries to the results dictionary.
        **kwargs: Forwarded to ssqueezepy.stft
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
                **torch_stft_kwargs,
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

    logger.info("Generating STFT cellstreams")
    cursor = 0  # position in block to process

    for b in tqdm(range(blocks)):
        this_block_size = block_size + (1 if b < remainder else 0)
        end = cursor + this_block_size
        block = img[cursor:end]  # (this_block_size, C, T)

        block_result = query_stft_block(
            block,
            min_bin=min_bin,
            max_bin=max_bin,
            num_filter_banks=num_filter_banks,
            normalize_amplitudes=normalize_amplitudes,
            carrier_channel=carrier_channel,
            channel_outputs=channel_outputs,
            use_gpu=use_gpu,
            sampling=sampling,
            **torch_stft_kwargs,
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
        "min_bin": min_bin,
        "max_bin": max_bin,
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
    for k, v in torch_stft_kwargs.items():
        attrs[k] = v

    result["_attrs"] = attrs
    return result


def extract_stft_cellstreams(features, track_masks):
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
