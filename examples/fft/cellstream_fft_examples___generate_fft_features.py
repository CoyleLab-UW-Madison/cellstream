import cellstream

img=cellstream.image.load_image("../images/example_timeseries_mini_0.tif")

fft_features=cellstream.fft.generate_fft_features(
    img,
    batch_size=50,
    device='cuda',
    max_bin=50
    )

cellstream.image.write_to_zarr(fft_features,"fft_features.zarr")
