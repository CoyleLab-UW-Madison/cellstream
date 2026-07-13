"""
Example script for full CWT-based folder processing pipeline.

This script demonstrates how to process an entire directory of images and masks
using the `process_folder_cwt_cellstreams` function. It aggregates data into a single 
pandas DataFrame and optionally creates Zarr stores containing per-cell spatial crops.
"""

import matplotlib.pyplot as plt
import cellstream

images_dir = "images"
masks_dir = "masks"

print("Processing folder of images with CWT...")
data = cellstream.cwt.process_folder_cwt_cellstreams(
    ### data directories
    images_dir,
    masks_dir,
    
    ### cwt parameters
    min_scale=25,
    max_scale=150,
    num_filter_banks=1,
    blocks='auto', 
    use_gpu=True,
    bank_method="sort",
    normalize_amplitudes=False,
    
    ### pre-processing
    normalize_histogram=True,
    
    ### channel information (matching the images in the example folder)
    channel_names=["minE", "minD"],
    carrier_channel=1, # using minD as carrier
    channel_outputs={
        0: ["amp", "phase_difference"],
        1: ["amp", "freq", "phase"],
    },

    #sampling parameters
    sampling={"fs": 2, "N": 361},
    
    ### zarr cropping options
    crop_zarrs=True,
    crop_kwargs={"padding_fraction": 0.2, "show_progress": True}
)

print("\n--- Folder Processing Complete ---")
print(f"Dataframe shape: {data.shape}")
print("\nFirst few rows of extracted single-cell CWT trajectories:")
print(data.head())

# Filter for the first filter bank
bank0_data = data[data["filter_bank"] == 0]

# Example visualization: Average minE amplitude vs average minD amplitude per cell
if not bank0_data.empty:
    print("\nPlotting average minE amplitude vs minD amplitude...")
    
    # Calculate means over time for each cell
    mean_stats = bank0_data.groupby(['image_filename', 'cell_id', 'channel', 'feature'])['mean'].mean().reset_index()
    
    # Pivot to get features as columns
    pivot_stats = mean_stats.pivot(index=['image_filename', 'cell_id'], columns=['channel', 'feature'], values='mean').reset_index()
    
    # Flatten multi-level columns
    pivot_stats.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in pivot_stats.columns]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(
        pivot_stats['minD_amp'], 
        pivot_stats['minE_amp'],
        alpha=0.7
    )
    plt.xlabel("Mean minD Amplitude (CWT)")
    plt.ylabel("Mean minE Amplitude (CWT)")
    plt.title("minD vs minE Amplitude across folder")
    plt.grid(True, alpha=0.3)
    plt.show()

    # And also for frequency
    plt.figure(figsize=(8, 6))
    plt.scatter(
        pivot_stats['minD_freq'], 
        pivot_stats['minE_phase_difference'],
        alpha=0.7
    )
    plt.xlabel("Mean minD Frequency (CWT)")
    plt.ylabel("Mean minE Phase Shift (CWT)")
    plt.title("minD vs minE Amplitude across folder")
    plt.grid(True, alpha=0.3)
    plt.show()


    print("Plots displayed.")
else:
    print("No data extracted.")
