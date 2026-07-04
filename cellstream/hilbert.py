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
    batch_size='auto',
    device=None,
    return_type='amp_phase' # 'amp_phase', 'real_imag', or 'complex'
):
    """
    Compute the Hilbert transform of an image timeseries.
    Expects (T, C, X, Y) tensor.
    Returns depending on return_type:
    - 'amp_phase': dict with 'amplitude' and 'phase' tensors
    - 'real_imag': dict with 'real' and 'imag' tensors
    - 'complex': single complex64 tensor
    """
    from .utils import get_auto_batch_size
    image = normalize_dims(image, 1)
    T, C, X, Y = image.shape
    
    if normalize_histogram:
        image = norm_hist(image)

    mean_image = image.mean(axis=0)
    image = image - mean_image

    if device is None:
        device = image.device

    if batch_size == 'auto':
        batch_size = get_auto_batch_size(
            (T * C,), 
            dtype=image.dtype, 
            device=device,
            bytes_per_element_multiplier=12 # Complex operations need more overhead
        )

    if batch_size is not None:
        image = image.reshape(T, C, X * Y)
        bar = tqdm(total=X * Y)
        ht_image = torch.zeros((T, C, X * Y), dtype=torch.complex64)

        for start in range(0, X * Y, batch_size):
            end = min(start + batch_size, X * Y)
            batch = image[:, :, start:end].to(device)
            ht_chunk = _process_hilbert_batch(batch)
            ht_image[:, :, start:end] = ht_chunk.cpu()
            bar.update(end - start)
        
        bar.close()
        ht_image = ht_image.reshape(T, C, X, Y)
    else:
        ht_image = _process_hilbert_batch(image.to(device)).reshape(T, C, X, Y)
        
    if return_type == 'amp_phase':
        return {'amplitude': ht_image.abs(), 'phase': ht_image.angle()}
    elif return_type == 'real_imag':
        return {'real': ht_image.real, 'imag': ht_image.imag}
    else:
        return ht_image

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
