"""
Zarr Pipeline Step 4: Process Phase Features
--------------------------------------------
This script consumes the individual Zarr stores and looks for existing 
`phase` arrays (either from the FFT or CWT modules). 
It generates winding numbers, continuous velocities, and FTLE flow fields.
"""

import os
from cellstream.phase.process import process_zarr_store

def main():
    crops_dir = "zarr_crops"
    
    if not os.path.exists(crops_dir):
        print(f"Please run zarr_pipeline_1_extract_crops.py first to create {crops_dir}/.")
        return
        
    zarr_stores = [f for f in os.listdir(crops_dir) if f.endswith(".zarr")]
    
    for zarr_file in zarr_stores:
        zarr_path = os.path.join(crops_dir, zarr_file)
        print(f"\n--- Processing {zarr_file} ---")
        
        # This will automatically crawl through FFT and CWT subgroups
        # finding 'phase' arrays and generating fluid dynamics features.
        process_zarr_store(
            zarr_path=zarr_path,
            force=True, #overwrite existing phase features
            phase_features_to_process=["winding_number", "velocity", "ftle_forward", "ftle_backward"],
            smooth_sigma=1.0,
            defect_window_size=3,
        )

if __name__ == "__main__":
    main()
