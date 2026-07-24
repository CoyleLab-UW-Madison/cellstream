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
    #min_mask_size=350,
    return_timeseries=True,
    crop_kwargs={"padding_fraction": 0.2, "show_progress": True, "min_mask_size": 100}
)

print("\n--- Folder Processing Complete ---")
print(f"Dataframe shape: {data.shape}")
print("\nFirst few rows of extracted single-cell CWT trajectories:")
print(data.head())

if not data.empty:
    # Filter for the first filter bank
    bank0_data = data[data.get("filter_bank", 0) == 0] if "filter_bank" in data.columns else data
    
    # Example visualization: Average minE amplitude vs average minD amplitude per cell
    if not bank0_data.empty:
        print("\nPlotting average minE amplitude vs minD amplitude...")
        
        # Calculate means over time for each cell
        mean_stats = bank0_data.groupby(['image_filename', 'cell_id', 'channel', 'feature'])['mean'].mean().reset_index()
        
        # Pivot to get features as columns
        pivot_stats = mean_stats.pivot(index=['image_filename', 'cell_id'], columns=['channel', 'feature'], values='mean').reset_index()
        
        # Flatten multi-level columns
        pivot_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) and col[0] else col[1] if isinstance(col, tuple) else col for col in pivot_stats.columns]
        
        # Plot
        try:
            plt.figure(figsize=(10, 8))
            
            # Use 'minE_amp' and 'minD_amp' if present, adapt if column names differ based on the flattening
            x_col = next((c for c in pivot_stats.columns if 'minD' in c and 'amp' in c), None)
            y_col = next((c for c in pivot_stats.columns if 'minE' in c and 'amp' in c), None)
            
            if x_col and y_col:
                plt.scatter(pivot_stats[x_col], pivot_stats[y_col], alpha=0.6, edgecolors='w', s=50)
                plt.title('Average minE Amp vs minD Amp (Filter Bank 0)')
                plt.xlabel(f'Average {x_col}')
                plt.ylabel(f'Average {y_col}')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('cwt_folder_scatter.png')
                print("Saved example scatter plot to 'cwt_folder_scatter.png'")
            else:
                print(f"Could not find required columns for scatter plot. Available columns: {pivot_stats.columns}")
        except Exception as e:
            print(f"Plotting failed: {e}")
    else:
        print("\nNo data for filter_bank 0.")
else:
    print("No data extracted.")
