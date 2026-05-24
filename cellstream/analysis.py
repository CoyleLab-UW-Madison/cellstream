"""
cellstream.analysis

Tools for aggregating per-pixel data into single-cell statistics and dataframes.
"""

import pandas as pd
import torch
from torch_scatter import scatter_mean, scatter_std
import re

def extract_single_cell_data(masks_dict, feature_maps, mean_levels_image=None):
    """
    Aggregate per-pixel features into per-cell statistics using segmentation masks.
    """
    # Get spatial dimensions from first feature
    first_feat = feature_maps[list(feature_maps.keys())[0]]
    if first_feat.dim() == 4: # (F, C, X, Y)
        F, C, X, Y = first_feat.shape
    elif first_feat.dim() == 3: # (C, X, Y)
        C, X, Y = first_feat.shape
    else:
        raise ValueError(f"Unexpected feature dimension: {first_feat.dim()}")

    # Flatten X, Y
    reshaped_features = {
        key: val.reshape(-1, X * Y) for key, val in feature_maps.items()
    }

    if mean_levels_image is not None:
        reshaped_mean_levels = mean_levels_image.reshape(-1, X * Y)

    def compute_stats(mask_flat, dim_size):
        stats = {}
        for key, val in reshaped_features.items():
            stats[key] = scatter_mean(val, mask_flat, dim=-1, dim_size=dim_size)
            stats[f"{key}_sd"] = scatter_std(val, mask_flat, dim=-1, dim_size=dim_size)

        if mean_levels_image is not None:
            stats["levels"] = scatter_mean(reshaped_mean_levels, mask_flat, dim=-1, dim_size=dim_size)
        return stats

    results = {}
    num_labels = max(int(mask.max().item()) for mask in masks_dict.values()) + 1
    for name, mask in masks_dict.items():
        mask_flat = mask.reshape(X * Y)
        results[name] = compute_stats(mask_flat, num_labels)

    return results

def create_dataframe(results, channel_names=None, image_filename=None, masks_filename=None):
    """
    Convert aggregation results into a structured pandas DataFrame.
    """
    first_mask_key = next(iter(results))
    first_stat_key = next(iter(results[first_mask_key]))
    num_cells = results[first_mask_key][first_stat_key].shape[1]

    if channel_names is None:
        num_channels = results[first_mask_key][first_stat_key].shape[0]
        channel_names = [f"Channel {i}" for i in range(num_channels)]

    df_data = {
        "cell_id": torch.arange(num_cells).detach().cpu().numpy(),
        "image_filename": image_filename,
        "mask_filename": masks_filename,
    }

    for mask_name, result in results.items():
        suffix = "" if mask_name == "all" else f"___{mask_name}"
        for key, tensor in result.items():
            is_sd = key.endswith("_sd")
            base = key[:-3] if is_sd else key
            stat_suffix = "_sd" if is_sd else "_mean"
            for ch_idx, ch_name in enumerate(channel_names):
                colname = f"{ch_name}_{base}{stat_suffix}{suffix}"
                df_data[colname] = tensor[ch_idx].detach().cpu().numpy()

    return pd.DataFrame(df_data)

def reshape_to_longform(df):
    """
    Reshape wide-form DataFrame to tidy long-form.
    """
    id_vars = ["cell_id", "image_filename", "mask_filename"]
    value_vars = [col for col in df.columns if col not in id_vars]

    long_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="measurement", value_name="value")

    pattern = re.compile(r"(?P<channel>.+?)_(?P<feature>.+?)_(?P<stat>mean|sd)(?:___(?P<mask_type>.*))?")
    extracted = long_df["measurement"].str.extract(pattern)
    long_df = pd.concat([long_df, extracted], axis=1).drop(columns="measurement")
    long_df["mask_type"] = long_df["mask_type"].fillna("all")

    return long_df
