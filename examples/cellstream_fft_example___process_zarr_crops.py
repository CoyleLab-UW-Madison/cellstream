"""
Zarr Pipeline Step 2: Process FFT
---------------------------------
This script consumes the individual Zarr stores created in Step 1.
It runs the FFT feature generation process cell-by-cell using the 
new `rich` UI and saves the results back into the Zarr store.
"""

import os
from cellstream.fft.process import process_zarr_store

def main():
    crops_dir = "zarr_crops"
    
    if not os.path.exists(crops_dir):
        print(f"Please run zarr_pipeline_1_extract_crops.py first to create {crops_dir}/.")
        return
        
    zarr_stores = [f for f in os.listdir(crops_dir) if f.endswith(".zarr")]
    
    for zarr_file in zarr_stores:
        zarr_path = os.path.join(crops_dir, zarr_file)
        print(f"\n--- Processing {zarr_file} ---")
        
        # This will automatically find 'timeseries', run FFT, and save results!
        process_zarr_store(
            zarr_path=zarr_path,
            force=False, # Set to True to overwrite existing FFT data
            fft_features_to_process=["full_amplitude", "phase", "z_score"]
        )

if __name__ == "__main__":
    main()
