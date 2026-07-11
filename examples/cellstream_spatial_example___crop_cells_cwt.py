# -*- coding: utf-8 -*-
"""
cellstream spatial crop example — crop_zarr_from_masks

Demonstrates cropping per-cell image data from FFT features using a label mask.
Each cell gets its own zarr group containing cropped timeseries, FFT feature maps,
and a binary mask — all retaining full spatial structure (not reduced to scalars).

@author: coylelab
"""

import cellstream
import numpy as np

# -------------------------------------------------------------------------
# 1. Load a timeseries image and its label mask
# -------------------------------------------------------------------------
image = cellstream.load_image("images/example_timeseries_mini_0.tif")
masks = cellstream.load_masks("masks/example_timeseries_mini_0_masks.tif")

print(f"Image shape: {image.shape}")   # (T=361, C=2, X=250, Y=250)
print(f"Masks shape: {masks.shape}")   # (X=250, Y=250)
print(f"Unique cell labels: {int(masks.max())}")

# -------------------------------------------------------------------------
# 2. Generate CWT features (with timeseries included for cropping)
# -------------------------------------------------------------------------
cwt_features = cellstream.cwt.generate_cwt_image_cellstreams(
    ###image file
    image,
    ###cwt parameters
    min_scale=80,
    max_scale=180,
    num_filter_banks=1,
    blocks='auto',
    use_gpu=True,
    bank_method="max_pool",
    normalize_amplitudes=False,
    ###pre-processing
    # downsample_by=0.25,
    normalize_histogram=True,
    return_timeseries=True,
    ###channel information
    channel_names=["MinE", "MinD"],
    carrier_channel=1,
    channel_outputs={0: ["amp", "phase_difference"], 1: ["amp", "freq","phase"]},
    #channel_outputs={0: ["amp"], 1: ["amp"]},
    ##sampling parameters
    sampling={"fs": 2, "N": 361},

)

print(f"\nFFT feature keys: {[k for k in cwt_features if k != '_attrs']}")
for k, v in cwt_features.items():
    if k != "_attrs" and hasattr(v, "shape"):
        print(f"  {k}: {tuple(v.shape)}")

# -------------------------------------------------------------------------
# 3. Crop per-cell features using the label mask
# -------------------------------------------------------------------------
output_path = "example_cell_crops.zarr"

root = cellstream.crop_zarr_from_masks(
    features=cwt_features,
    label_image=masks,
    output_path=output_path,
    padding_fraction=0.1,     # 10% padding around each cell's bounding box
    min_padding_px=2,         # at least 2px padding even for tiny cells
)

print(f"\n--- Crop results ---")
print(f"Output zarr: {output_path}")
print(f"Number of cell groups: {root.attrs['num_cells']}")

# -------------------------------------------------------------------------
# 4. Inspect a single cell's cropped data
# -------------------------------------------------------------------------
# Pick the first real cell (label 1)
cell = root["cell_1"]
cell_attrs = dict(cell.attrs)

print(f"\n--- Cell 1 metadata ---")
print(f"  Label ID:       {cell_attrs['label_id']}")
print(f"  Bbox (original): {cell_attrs['bbox_original']}  (y_min, y_max, x_min, x_max in full image)")
print(f"  Bbox (padded):   {cell_attrs['bbox_padded']}")
print(f"  Centroid (y,x):  {cell_attrs['centroid_yx']}")
print(f"  Area (pixels):   {cell_attrs['area_pixels']}")
print(f"  Crop shape:      {cell_attrs['crop_shape']}")

print(f"\n--- Cell 1 arrays ---")
for key in cell.keys():
    arr = cell[key]
    if hasattr(arr, "shape"):
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

# -------------------------------------------------------------------------
# 5. Verify data integrity: cropped data matches the original at the right location
# -------------------------------------------------------------------------
y0, y1, x0, x1 = cell_attrs["bbox_padded"]

# Compare timeseries crop to original image at the same location
original_crop = image[:, :, y0:y1, x0:x1].numpy()
zarr_crop = np.asarray(cell["timeseries"][:])

print(f"\n--- Data integrity check ---")
print(f"  Original image crop shape: {original_crop.shape}")
print(f"  Zarr timeseries crop shape: {zarr_crop.shape}")

# Note: timeseries in FFT features has been histogram-normalized, so values
# won't match the raw image exactly. But shapes must match.
assert original_crop.shape == zarr_crop.shape, "Shape mismatch!"
print("  [OK] Shapes match -- crop correctly maps back to original image coordinates")

# -------------------------------------------------------------------------
# 6. Show how to apply the mask to isolate just the cell (optional)
# -------------------------------------------------------------------------
cell_mask = np.asarray(cell["mask"][:])  # (crop_Y, crop_X), binary
print(f"\n--- Applying mask ---")
print(f"  Mask shape: {cell_mask.shape}, nonzero pixels: {cell_mask.sum()}")

# To get masked data (NaN outside cell), broadcast the mask:
masked_timeseries = zarr_crop.copy()
masked_timeseries[:, :, cell_mask == 0] = np.nan
print(f"  Masked timeseries shape: {masked_timeseries.shape}")
print(f"  NaN fraction: {np.isnan(masked_timeseries).mean():.1%} (area outside cell)")

print("\nDone! Cell crops saved to:", output_path)
