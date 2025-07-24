# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:04:37 2025

@author: smcoyle
"""

import nd2
import tifffile
import torch

def load_image(image_filename):
    """Load image from file and convert to torch tensor"""
    iname, iext = image_filename.split('.')
    if iext == 'nd2':
        image = nd2.imread(image_filename)
    elif iext == 'tif':
        image = tifffile.imread(image_filename)
    
    image=torch.from_numpy(image.astype('float32'))
    
    if image.dim()==3:
        print("Single-channel image detected; adding channel dimension...")
        image=image.unsqueeze(1)
    
    return image

def load_masks(masks_filename):
    """Load masks from file and convert to torch tensor"""
    mname, mext = masks_filename.split('.')
    if mext == 'nd2':
        masks = nd2.imread(masks_filename)
    elif mext == 'tif':
        masks = tifffile.imread(masks_filename)
    return torch.from_numpy(masks.astype('int64'))

