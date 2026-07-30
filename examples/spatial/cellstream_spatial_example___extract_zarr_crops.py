"""
Zarr Pipeline Step 1: Extract Crops
-----------------------------------
This script demonstrates how to decouple the raw data extraction from the 
feature generation process. It loops over a folder of full-frame TIFFs and 
their corresponding masks, and chops them up into independent per-cell Zarr 
stores on disk.

This is highly recommended for users with limited RAM or for datasets that 
will be processed by multiple modules (FFT, CWT, Phase) to avoid re-cropping.
"""

import os
from cellstream.spatial import process_folder_to_crop_zarrs

def main():
    # Directories containing your raw TIFFs and masks
    images_dir = "../images"
    masks_dir = "../masks"
    
    # Where the Zarr stores will be saved
    output_dir = "../zarr_crops"
    
    # We will name the extracted raw array "timeseries"
    channel_name = "timeseries"
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Please ensure {images_dir}/ and {masks_dir}/ exist with your test images.")
        return

    print("Starting Zarr crop extraction...")
    process_folder_to_crop_zarrs(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        channel_name=channel_name,
        padding_fraction=0.1,  # Add 10% padding around bounding boxes
        show_progress=True, 
        min_mask_size=350      # Skip degenerate tiny masks
    )
    
    print("\nExtraction complete! You can now run the FFT, CWT, or Phase scripts.")

if __name__ == "__main__":
    main()
