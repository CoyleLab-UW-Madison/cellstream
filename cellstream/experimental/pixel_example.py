# -*- coding: utf-8 -*-
"""
Refactored pixel profiling pipeline using the cellstream pixel_profiler module.
"""

from pathlib import Path
import pixel_profiler

# Settings
base_dir = Path("Z:\\Special\\Globus\\Rajasekaran_2024_ImagingDataFromPaper\\3T3") 
files = list(base_dir.glob("*.nd2"))

if not files:
    print(f"No ND2 files found in directory: {base_dir}")
else:
    print(f"Found {len(files)} files to process...")

    # Run batch profiling
    final_df = pixel_profiler.batch_profile_pixels(
        file_paths=files,
        channel_names=['E', 'D'],
        c_val=35.0,
        min_bin=4,
        max_bin=40,
        filter_method='product',
        peak_constraint='exactly_one',
        register_images=True,
        device='cuda',
        max_fft_bin=50,
        fft_batch_size=250,
        show_progress=True
    )

    # Export aggregated data to Parquet format
    if not final_df.empty:
        output_filename = "2026_03_23_3t3_dataset.parquet"
        final_df.to_parquet(output_filename, index=False)
        print(f"Successfully saved to {output_filename}")
    else:
        print("No pixels met the filtering criteria. No output was generated.")
