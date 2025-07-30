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

from torch_scatter import scatter_mean, scatter_std

from ..image.utils import downsample
from ..image.utils import normalize_histogram as norm_hist

def query_cwt_block(
        data, 
        min_scale=80, 
        max_scale=180, 
        num_filter_banks=1, 
        normalize_amplitudes=False, 
        carrier_channel=0, 
        channel_outputs={0: ['amp', 'freq', 'phase']},
        use_gpu=False,
        bank_method='max_pool',
        sampling=None,
        **ssqueezepy_cwt_kwargs
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
    os.environ['SSQ_GPU'] = '1' if use_gpu else '0'
    from ssqueezepy import cwt 
    
    #Prepare shapes
    BATCH_SIZE, C, T=data.shape
    
    #prepare channel-specific containers
    split_channels={}
    split_channels_full_power_sums={}
    split_channels_full_means={}
    split_channels_full_std={}
        
    for channel in channel_outputs:    
    # Compute CWT with forwarded parameters
        Twx, scales = cwt(data[:,channel,:], **ssqueezepy_cwt_kwargs)
        if isinstance(Twx, np.ndarray):
            Twx = torch.tensor(Twx)
        if normalize_amplitudes==True:
            split_channels_full_power_sums[channel]=Twx.abs().sum(axis=1,keepdims=True)
        if 'z_score' in channel_outputs[channel]:
            split_channels_full_means[channel]=Twx.abs().mean(axis=1,keepdims=True)
            split_channels_full_std[channel]=Twx.abs().std(axis=1,keepdims=True)
        Twx_sub = Twx[:,min_scale:max_scale,:].clone()
        split_channels[channel]=Twx_sub

    # Find max_scale channels along scale dimension
    
  
    carrier_amp=split_channels[carrier_channel].abs()
    carrier_phase=split_channels[carrier_channel].angle()
    
    
    #manage filter-banking methods:
    
    if bank_method=='max_pool':
        #prepare for max pooling
        max_pooler = torch.nn.AdaptiveMaxPool1d(num_filter_banks,return_indices=True)
        carrier_amp=carrier_amp.permute(0,2,1) # max_pooler wants to pool the -1 axis
        carrier_amp,carrier_freq=max_pooler(carrier_amp) #max pool along the scales ()
        carrier_amp=carrier_amp.permute(0,2,1) # shape is (batch,num_filter_banks,time)
        carrier_freq=carrier_freq.permute(0,2,1) # shape is (batch,num_filter_banks,time)
        carrier_phase=torch.gather(carrier_phase,1,carrier_freq)
    elif bank_method=='sort':
        carrier_amp,carrier_freq=torch.sort(split_channels[carrier_channel].abs(),axis=1,descending=True)
        carrier_phase=split_channels[carrier_channel].angle()
        carrier_phase=torch.gather(carrier_phase,1,carrier_freq)
        
    # Prepare outputs dictionary
    results = {c: {} for c in range(C)}
    
    #prepare for scale to freq conversion
    wavelet = ssqueezepy_cwt_kwargs.get("wavelet", "gmw")
    if "wavelet" not in ssqueezepy_cwt_kwargs:
        if sampling is not None:
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
            results[channel]['phase'] = ch_ph[:,:num_filter_banks,:].cpu()
        if 'phase_difference' in returns:
            results[channel]['phase_difference']= (((ch_ph-carrier_phase)% (2*torch.pi)).abs())[:,:num_filter_banks,:].cpu()
        if 'amp' in returns:
            ch_p=torch.gather(P,1,carrier_freq)  # Gather relevant phases
            results[channel]['amp'] = ch_p[:,:num_filter_banks,:].cpu()
        if 'z_score' in returns:
            if 'amp' in returns:
                z=(ch_p-split_channels_full_means[channel])/split_channels_full_std[channel]
            else:
                ch_p=torch.gather(P,1,carrier_freq)
                z=(ch_p-split_channels_full_means[channel])/split_channels_full_std[channel]
            results[channel]['z_score'] = z[:,:num_filter_banks,:].cpu()
        if 'freq' in returns:
            if sampling is not None:
                results[channel]['freq'] = carrier_freq_converted[:,:num_filter_banks,:].cpu() #replace with proper conversion soon
            else:
                results[channel]['freq'] = carrier_freq[:,:num_filter_banks,:].cpu()+min_scale

            
    return results


def generate_cwt_image_cellstreams(
        img, 
        
        min_scale=80, 
        max_scale=180, 
        num_filter_banks=1, 
        normalize_amplitudes=False, 
        blocks=10, 
        use_gpu=False,
        bank_method='max_pool',

        
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
            final[c][k] = torch.zeros((X*Y, num_filter_banks, T), dtype=torch.float32)
            
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
            num_filter_banks=num_filter_banks, 
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
                val = block_result[c][k]  # (this_block_size, num_filter_banks, T)
                final[c][k][cursor:end] = val
        cursor = end
        
    #reshape
    for c in channel_outputs:
        for k in channel_outputs[c]:
            final[c][k]=final[c][k].permute(2,1,0).reshape(T,num_filter_banks,X,Y)
    
    #adjust to match channel names if need be 
    if channel_names is not None:
        return {channel_names[idx]: outdict for idx, outdict in final.items()}
    else:
        return final

def extract_cwt_cellstreams(features,track_masks):
    
    '''extract single-cell trajectories using label_image tracks'''
    
    ##reshape features
    if features.dim()==3:
        print("3 channel image detected; unsqueezing C dimension...")
        features=features.unsqueeze(1)
    T,C,X,Y=features.shape
    features=features.reshape(T,C,-1)
    
    ##reshape masks
    if track_masks.dim()==2: #static 2D mask
        track_masks=track_masks.broadcast_to(T,C,X,Y)
    elif track_masks.dim()==3:
         #timeseries mask (T,X,Y)
        track_masks=track_masks.broadcast_to(C,T,X,Y)
        track_masks=track_masks.permute(1,0,2,3) # (T,C,X)
    track_masks=track_masks.reshape(T,C,-1)
    
    num_masks=int(track_masks.max().item()) + 1
    
    cellstreams_mean=scatter_mean(features,track_masks,dim=-1,dim_size=num_masks) #T,C,num_masks
    cellstreams_mean=cellstreams_mean.permute(2,1,0)
    
    cellstreams_std=scatter_std(features,track_masks,dim=-1,dim_size=num_masks)
    cellstreams_std=cellstreams_std.permute(2,1,0)
    return cellstreams_mean,cellstreams_std
