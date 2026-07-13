"""
Example script for full CWT-based processing pipeline.

This script demonstrates how to use the high-level `process_cwt_image_cellstreams` 
function to process a multi-channel time-series image and corresponding track masks. 
It automatically performs CWT feature extraction, single-cell data aggregation, 
and optionally creates a Zarr store containing spatial crops of each cell along 
with their extracted CWT statistics.
"""

import cellstream

#  Load image and masks
print("Loading image and masks...")
timeseries_image = cellstream.image.load_image("timeseries_for_cwt.tif")
track_masks = cellstream.image.load_masks("timeseries_masks_for_cwt.tif")

# Process image with CWT and enable Zarr cropping
print("Processing image with CWT and generating single-cell crops...")
result = cellstream.cwt.process_cwt_image_cellstreams(
    ### image and masks
    timeseries_image,
    track_masks,
    
    ### cwt parameters
    min_scale=25,
    max_scale=150,
    num_filter_banks=1,
    blocks=50,
    use_gpu=True,
    bank_method="sort",
    normalize_amplitudes=False,
    
    ### pre-processing
    normalize_histogram=True,
    
    ### channel information
    channel_names=["minD", "pkc_activity", "pka_activity"],
    carrier_channel=0,
    channel_outputs={
        0: ["amp", "freq", "phase"],
        1: ["amp", "phase_difference"],
        2: ["amp", "phase_difference"],
    },
    
    ### file metadata
    image_filename="timeseries_for_cwt.tif",
    masks_filename="timeseries_masks_for_cwt.tif",
    
    ### zarr cropping options
    crop_zarrs=True,
    crop_output_path="cwt_example_crops.zarr",
    crop_kwargs={"padding_fraction": 0.2, "show_progress": True}
)

df, crop_root = result # note: crop_zarr=True -> (dataframe,zarr) tuple

print("\n--- Processing Complete ---")
print(f"Dataframe shape: {df.shape}")
print("\nFirst few rows of extracted single-cell CWT trajectories:")
print(df.head())

# Display the zarr output information
print(f"\nGenerated Zarr store at: {crop_root.store.path}")
print(f"Number of cells extracted into Zarr store: {len(list(crop_root.group_keys()))}")

# Inspect the attributes attached to a specific cell
cell_groups = list(crop_root.group_keys())
if cell_groups:
    first_cell = cell_groups[0]
    print(f"\nAttributes attached to {first_cell} (includes extracted summary stats):")
    for key, value in crop_root[first_cell].attrs.items():
        if str(key).startswith("extracted_"):
            print(f"  {key}: {value}")
