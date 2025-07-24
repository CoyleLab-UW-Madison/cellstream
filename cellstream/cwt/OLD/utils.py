# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 14:52:43 2025

@author: smcoyle
"""

import torch
import progressbar
import os
import numpy as np
import warnings

#from torch_scatter import scatter_mean, scatter_std

from ..image.utils import downsample
from ..image.utils import normalize_histogram as norm_hist

def query_cwt_block(
        data, 
        min_scale=80, 
        max_scale=180, 
        filter_banks=1, 
        normalize_amplitudes=False, 
        carrier_channel=0, 
        channel_outputs={0: ('amp', 'freq', 'phase')},
        use_gpu=False,
        sort_freq=True,
        sampling=None,
        **ssqueezepy_cwt_kwargs
    ):
    
    """
    Process a block of data using CWT with selective outputs
    Args:
        data: Input tensor of shape (N, T) where N=number of pixels, T=timepoints
        outputs: Tuple of requested outputs ('amp', 'freq', 'phase')
        **kwargs: Forwarded to ssqueezepy.cwt
    Returns:
        Dictionary of requested outputs
    """
   
    # Force environment variable BEFORE import
    os.environ['SSQ_GPU'] = '1' if use_gpu else '0'
    from ssqueezepy import cwt 
    
    #Prepare shapes
    BATCH_SIZE, C, T=data.shape
    
    split_channels={}
    if normalize_amplitudes==True:
        split_channels_full_power_sums={}
        
    for channel in channel_outputs:    
    # Compute CWT with forwarded parameters
        Twx, scales = cwt(data[:,channel,:], **ssqueezepy_cwt_kwargs)  # Shape (BATCH*C, N, T)
        if isinstance(Twx, np.ndarray):
            Twx = torch.tensor(Twx)
        if normalize_amplitudes==True:
            split_channels_full_power_sums[channel]=Twx.abs().sum(axis=1,keepdims=True)
        Twx_sub = Twx[:,min_scale:max_scale,:].clone()
        split_channels[channel]=Twx_sub

    # Find max_scale channels along scale dimension
    
    if sort_freq==True:
        carrier_amp,carrier_freq=torch.sort(split_channels[carrier_channel].abs(),axis=1,descending=True)
        carrier_phase=split_channels[carrier_channel].angle()
        carrier_phase=torch.gather(carrier_phase,1,carrier_freq)
    
    # Prepare outputs dictionary
    results = {c: {} for c in range(C)}
    
    #prepare for scale to freq conversion
    wavelet = ssqueezepy_cwt_kwargs.get("wavelet", "gmw")
    if "wavelet" not in ssqueezepy_cwt_kwargs:
        warnings.warn(
            "No wavelet specified in kwargs. Defaulting to 'gmw' for scale-to-frequency conversion. "
            "You should explicitly specify the wavelet for clarity and reproducibility.",
            stacklevel=2
        )
    
    #handles conversion of scales to frequencies
    if sampling is not None:
        from ssqueezepy.experimental import scale_to_freq 
        fs = sampling['fs']
        N = sampling['N']
        blank_series=torch.ones(T)
        Twx, scales = cwt(blank_series, **ssqueezepy_cwt_kwargs)
        freqs_lookup=scale_to_freq(scales,wavelet=wavelet,N=N,fs=fs)
        freqs_lookup=torch.from_numpy(freqs_lookup.astype('float32')).broadcast_to(BATCH_SIZE,T,-1)
        freqs_lookup=freqs_lookup.permute(0,2,1) # (batch, scales, time)
        if use_gpu==True:
            freqs_lookup=freqs_lookup.to('cuda')
        carrier_freq_converted=torch.gather(freqs_lookup,1,carrier_freq+min_scale)
    
    
    for channel,returns in channel_outputs.items():
        P=split_channels[channel].abs()
        if normalize_amplitudes==True:
            P=P/split_channels_full_power_sums[channel]
        if ('phase' in returns) or ('phase_difference' in returns):
            #only compute phase if needed
            PH=split_channels[channel].angle()
            ch_ph=torch.gather(PH,1,carrier_freq)
        if 'phase' in returns:
            results[channel]['phase'] = ch_ph[:,:filter_banks,:].cpu()
        if 'phase_difference' in returns:
            results[channel]['phase_difference']= (((ch_ph-carrier_phase)% (2*torch.pi)).abs())[:,:filter_banks,:].cpu()
        if 'amp' in returns:
            ch_p=torch.gather(P,1,carrier_freq)  # Gather relevant phases
            results[channel]['amp'] = ch_p[:,:filter_banks,:].cpu()
        if 'freq' in returns:
            if sampling is not None:
                results[channel]['freq'] = carrier_freq_converted[:,:filter_banks,:].cpu() #replace with proper conversion soon
            else:
                results[channel]['freq'] = carrier_freq[:,:filter_banks,:].cpu()+min_scale

            
    return results


def generate_cwt_image_cellstreams(
        img, 
        
        min_scale=80, 
        max_scale=180, 
        filter_banks=1, 
        normalize_amplitudes=False, 
        blocks=10, 
        use_gpu=False,
        sort_freq=True,

        
        downsample_by=None,
        normalize_histogram=True,
        mean_center=False,
        
        carrier_channel=0, 
        channel_names=None, 
        channel_outputs={
            0:['amp', 'freq', 'phase']
            },
        sampling=None,
        **ssqueezepy_cwt_kwargs
    ):
    
    """
    Process image in blocks with selective outputs and kwargs forwarding
    Args:
        img: Input array (T, X, Y)
        outputs: 
        **kwargs: Forwarded to ssqueezepy.cwt
    Returns:
        Dictionary of requested outputs as numpy arrays
    """
    #gpu environment setup for squeezepy
    os.environ['SSQ_GPU'] = '1' if use_gpu else '0'
    from ssqueezepy import cwt 
    
    if sampling is not None:
        from ssqueezepy.experimental import scale_to_freq 
    
    #image pre-processing
    if downsample_by is not None:
        print(f"Downsampling image by {downsample_by} ...")
        img=downsample(img,downsample_by)
    
    if normalize_histogram is not False:
        print("Performing histogram normalization on image ...")
        img=norm_hist(img)
    
    if mean_center is not False:
        print("Mean centering timeseries ...")
        img=img-img.mean(axis=0)
        
    #reshape image for blocked processing
    T, C, X, Y = img.shape    
    img=img.reshape(T,C,X*Y).permute(2,1,0) # shape is now (x*y,c,t)
    
    #pre-allocate outputs
    final = {c: {} for c in channel_outputs}
    for c in channel_outputs:
        for k in channel_outputs[c]:
            final[c][k] = torch.zeros((X*Y, filter_banks, T), dtype=torch.float32)
            
    #setup blocked processing loop parameters
    total_pixels = X * Y    
    block_size = total_pixels // blocks
    remainder = total_pixels % blocks 
    
    print("Generating CWT cellstreams")
    cursor = 0 #position in block to process
    
    for b in progressbar.progressbar(range(blocks)):

        this_block_size = block_size + (1 if b < remainder else 0)
        end = cursor + this_block_size
        block = img[cursor:end]  # (this_block_size, C, T)
    
        block_result = query_cwt_block(
            block, 
            min_scale=min_scale, 
            max_scale=max_scale, 
            filter_banks=filter_banks, 
            normalize_amplitudes=normalize_amplitudes,
            carrier_channel=carrier_channel,
            channel_outputs=channel_outputs,
            use_gpu=use_gpu,
            sampling=sampling,
            **ssqueezepy_cwt_kwargs
        )
        # Fill in preallocated tensors
        for c in channel_outputs:
            for k in channel_outputs[c]:
                val = block_result[c][k]  # (this_block_size, filter_banks, T)
                final[c][k][cursor:end] = val
        cursor = end
        
    #reshape
    for c in channel_outputs:
        for k in channel_outputs[c]:
            final[c][k]=final[c][k].permute(2,1,0).reshape(T,filter_banks,X,Y)
    
    #adjust to match channel names if need be 
    if channel_names is not None:
        return {channel_names[idx]: outdict for idx, outdict in final.items()}
    else:
        return final

