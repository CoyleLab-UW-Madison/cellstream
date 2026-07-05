import cellstream

timeseries_image = cellstream.image.load_image("images/example_timeseries_mini_0.tif")

## perform STFT filter-banking, using 1 bank across frequency bins 0-30

results = cellstream.sfft.generate_stft_image_cellstreams(
    ###image file
    timeseries_image,
    ###stft parameters
    min_bin=5,
    max_bin=35,
    n_fft=60,
    num_filter_banks=8,
    #num_filter_banks=1,
    blocks='auto',
    use_gpu=True,
    bank_method="max_pool",
    normalize_amplitudes=False,
    ###pre-processing
    # downsample_by=0.25,
    normalize_histogram=True,
    ###channel information
    channel_names=["MinE", "MinD"],
    carrier_channel=1,
    #channel_outputs={0: ["amp", "phase_difference"], 1: ["amp", "freq","phase"]},
    channel_outputs={0: ["amp"], 1: ["amp"]},
    ##sampling parameters
    sampling={"fs": 2, "N": 361},
)
