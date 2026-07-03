"""
cellstream.hilbert

Hilbert transform for computing analytical signals and instantaneous phase/amplitude.
"""

import torch
from tqdm.auto import tqdm
from .utils import normalize_dims, normalize_histogram as norm_hist

def hilbert_transform(
    image,
    normalize_histogram=True,
    batch_size=None,
    device=None,
):
    """
    Compute the Hilbert transform of an image timeseries.
    Expects (T, C, X, Y) tensor.
    """
    image = normalize_dims(image, 1)
    T, C, X, Y = image.shape
    
    if normalize_histogram:
        image = norm_hist(image)

    mean_image = image.mean(axis=0)
    image = image - mean_image

    if device is None:
        device = image.device

    if batch_size is not None:
        image = image.reshape(T, C, X * Y)
        bar = tqdm(total=X * Y)
        ht_image = torch.zeros((T, C, X * Y), dtype=torch.complex64)

        for start in range(0, X * Y, batch_size):
            end = min(start + batch_size, X * Y)
            batch = image[:, :, start:end].to(device)
            ht_chunk = _process_hilbert_batch(batch)
            ht_image[:, :, start:end] = ht_chunk.cpu()
            bar.update(end)
        
        return ht_image.reshape(T, C, X, Y)
    else:
        return _process_hilbert_batch(image.to(device)).reshape(T, C, X, Y)

def _process_hilbert_batch(batch):
    """
    Compute Hilbert transform on a batch (T, ...).
    """
    T = batch.shape[0]
    freqs = torch.fft.rfft(batch, axis=0)  
    transforms = -1j * freqs
    transforms[0] = 0 # zero DC
    imaginary = torch.fft.irfft(transforms, n=T, axis=0)
    real = batch
    
    return torch.complex(real, imaginary)
