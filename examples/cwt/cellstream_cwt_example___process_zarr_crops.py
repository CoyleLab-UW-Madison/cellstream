"""
Zarr Pipeline Step 3: Process CWT
---------------------------------
This script consumes the individual Zarr stores created in Step 1.
It runs the CWT feature generation process cell-by-cell using the 
new `rich` UI and saves the results back into the Zarr store.
"""

import os
from cellstream.cwt.process import process_zarr_store

def main():
    crops_dir = "../zarr_crops"
    
    if not os.path.exists(crops_dir):
        print(f"Please run zarr_pipeline_1_extract_crops.py first to create {crops_dir}/.")
        return
        
    zarr_stores = [f for f in os.listdir(crops_dir) if f.endswith(".zarr")]
    
    for zarr_file in zarr_stores:
        zarr_path = os.path.join(crops_dir, zarr_file)
        print(f"\n--- Processing {zarr_file} ---")
        
        # Use exact settings matching process_folder_cwt_cellstreams
        process_zarr_store(
            zarr_path=zarr_path,
            force=False,
            min_scale=25,
            max_scale=150,
            num_filter_banks=1,
            carrier_channel=1,
            channel_outputs={
                0: ["amp", "phase_difference"],
                1: ["amp", "freq", "phase"],
            },
        )

if __name__ == "__main__":
    main()
