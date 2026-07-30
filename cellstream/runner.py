"""
cellstream.runner

TOML job configuration loader, parser, and sequential pipeline execution engine.
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # Handled with explicit error if called

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Registry of supported job types and their underlying module functions
JOB_DISPATCH = {
    "preprocess": {
        "folder": ("cellstream.preprocess", "process_folder_preprocess"),
        "zarr": ("cellstream.preprocess", "process_zarr_store"),
    },
    "fft": {
        "folder": ("cellstream.fft", "process_folder_cellstreams"),
        "zarr": ("cellstream.fft", "process_zarr_store"),
    },
    "cwt": {
        "folder": ("cellstream.cwt", "process_folder_cwt_cellstreams"),
        "zarr": ("cellstream.cwt", "process_zarr_store"),
    },
    "stft": {
        "folder": ("cellstream.stft", "process_folder_stft_cellstreams"),
        "zarr": None,
    },
    "phase": {
        "folder": None,
        "zarr": ("cellstream.phase", "process_zarr_store"),
    },
    "pixels": {
        "folder": ("cellstream.pixels", "batch_profile_pixels"),
        "zarr": None,
    },
}


def load_job_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load and validate a TOML job configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the .toml file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Job configuration file not found: {config_path}")

    if tomllib is None:
        raise ImportError(
            "TOML parser not available. Please install 'tomli' (for Python < 3.11) "
            "or run on Python >= 3.11."
        )

    with open(config_path, "rb") as f:
        try:
            config = tomllib.load(f)
        except Exception as e:
            raise ValueError(f"Error parsing TOML file '{config_path}': {e}")

    if "jobs" not in config or not isinstance(config["jobs"], list) or len(config["jobs"]) == 0:
        raise ValueError(f"Configuration file '{config_path}' must contain a non-empty [[jobs]] array.")

    base_dir = config_path.resolve().parent
    path_keys = {"images", "masks", "output", "input", "crop_output_path", "dataframe_output_path"}

    def resolve_paths(d: Dict[str, Any]) -> None:
        for k, v in d.items():
            if k in path_keys and isinstance(v, str):
                p = Path(v)
                if not p.is_absolute():
                    d[k] = str((base_dir / p).resolve())

    if "defaults" in config and isinstance(config["defaults"], dict):
        resolve_paths(config["defaults"])

    for idx, job in enumerate(config["jobs"]):
        if not isinstance(job, dict):
            raise ValueError(f"Job entry #{idx + 1} is invalid.")
        if "type" not in job:
            raise ValueError(f"Job entry #{idx + 1} is missing the required 'type' field.")
        job_type = str(job["type"]).lower()
        if job_type not in JOB_DISPATCH:
            valid_types = ", ".join(sorted(JOB_DISPATCH.keys()))
            raise ValueError(f"Unknown job type '{job['type']}' in job #{idx + 1}. Valid types: {valid_types}")
        
        resolve_paths(job)

    return config


def resolve_job_params(
    job_dict: Dict[str, Any],
    defaults: Dict[str, Any],
    previous_output: Optional[str] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Merge global defaults with job-specific parameters and resolve execution mode.

    Parameters
    ----------
    job_dict : dict
        Job parameters from TOML.
    defaults : dict
        Global defaults dictionary from TOML.
    previous_output : str, optional
        Output path from previous job in chain (if available).

    Returns
    -------
    tuple (job_type, mode, resolved_params)
        job_type: str ('cwt', 'fft', 'stft', 'phase', 'pixels')
        mode: str ('folder' or 'zarr')
        resolved_params: dict containing merged parameters
    """
    # Start with global defaults, then override with job-specific settings
    merged = dict(defaults)
    merged.update(job_dict)

    job_type = str(merged.pop("type")).lower()

    # Propagate global carrier_channel to job-specific parameters
    if "carrier_channel" in merged:
        if job_type == "fft" and "carrier_index" not in merged:
            merged["carrier_index"] = merged["carrier_channel"]
        if job_type == "pixels" and "filter_channel" not in merged:
            merged["filter_channel"] = merged["carrier_channel"]

    # Parse channel_outputs to ensure dictionary with integer keys
    if "channel_outputs" in merged:
        co = merged["channel_outputs"]
        if isinstance(co, list):
            carrier = merged.get("carrier_channel", 0)
            merged["channel_outputs"] = {int(carrier): co}
        elif isinstance(co, dict):
            merged["channel_outputs"] = {
                int(k) if str(k).isdigit() else k: v 
                for k, v in co.items()
            }

    # Check explicit job_dict keys first
    job_has_images = "images" in job_dict and job_dict["images"]
    job_has_masks = "masks" in job_dict and job_dict["masks"]
    job_has_input = "input" in job_dict and job_dict["input"]

    if job_has_images and job_has_masks:
        mode = "folder"
    elif job_has_input:
        mode = "zarr"
    elif previous_output is not None and JOB_DISPATCH[job_type]["zarr"] is not None:
        # Chained step: use previous output Zarr store if supported
        mode = "zarr"
        merged["input"] = previous_output
    elif "images" in merged and "masks" in merged:
        mode = "folder"
    elif "input" in merged:
        mode = "zarr"
    else:
        if job_type == "phase":
            raise ValueError(
                f"Job type '{job_type}' operates on Zarr stores. It requires 'input' or must be chained "
                "after a job that produces Zarr crops (crop_zarrs=true)."
            )
        else:
            raise ValueError(
                f"Job type '{job_type}' requires both 'images' and 'masks' parameters (folder mode), "
                "or an 'input' Zarr path (zarr mode), or must be chained after a Zarr-producing job."
            )

    # Check if job type supports the resolved mode
    dispatch_info = JOB_DISPATCH[job_type][mode]
    if dispatch_info is None:
        if mode == "zarr" and job_type == "stft":
            raise ValueError("STFT does not yet support Zarr store processing (zarr mode). Use folder mode instead.")
        elif mode == "folder" and job_type == "phase":
            raise ValueError("Phase processing does not support raw folder mode. Use Zarr crop store input.")
        else:
            raise ValueError(f"Job type '{job_type}' does not support '{mode}' mode.")


    return job_type, mode, merged


def run_job(
    job_type: str,
    mode: str,
    params: Dict[str, Any],
    dry_run: bool = False
) -> Optional[str]:
    """
    Execute or dry-run a single job step.

    Parameters
    ----------
    job_type : str
        The type of job ('cwt', 'fft', etc.).
    mode : str
        Execution mode ('folder' or 'zarr').
    params : dict
        Resolved parameter dictionary.
    dry_run : bool, default False
        If True, validate parameters and print plan without calling compute function.

    Returns
    -------
    str or None
        The output path produced by this job (for pipeline chaining), or None.
    """
    module_path, fn_name = JOB_DISPATCH[job_type][mode]

    # Prepare function arguments copy
    fn_kwargs = dict(params)

    # Determine chaining output path
    output_dir = fn_kwargs.get("output", None)
    crop_zarrs = fn_kwargs.get("crop_zarrs", False)

    chain_output_path: Optional[str] = None
    if mode == "zarr":
        chain_output_path = str(fn_kwargs.get("input", ""))
    elif mode == "folder" and (crop_zarrs or job_type == "preprocess"):
        if "crop_output_path" in fn_kwargs and fn_kwargs["crop_output_path"]:
            chain_output_path = str(fn_kwargs["crop_output_path"])
        elif output_dir:
            chain_output_path = str(output_dir)
        elif "images" in fn_kwargs:
            chain_output_path = str(Path(fn_kwargs["images"]).parent)

    if dry_run:
        console.print(f"[bold cyan]DRY-RUN:[/] Job type='[bold yellow]{job_type}[/]' mode='[bold yellow]{mode}[/]'")
        console.print(f"  Target function: [dim]{module_path}.{fn_name}[/]")
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Parameter")
        table.add_column("Value")
        for k, v in fn_kwargs.items():
            table.add_row(str(k), str(v))
        console.print(table)
        if chain_output_path:
            console.print(f"  [dim]Expected chain output -> {chain_output_path}[/]")
        console.print("")
        return chain_output_path

    # Lazy-import target module and function
    import importlib
    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name)

    # Dispatch invocation based on mode and function signature requirements
    
    # Always pop routing/configuration keys that shouldn't be passed to inner compute functions
    images_dir = fn_kwargs.pop("images", None)
    masks_dir = fn_kwargs.pop("masks", None)
    out_dir = fn_kwargs.pop("output", None)
    
    if mode == "folder":

        # Validate path existence
        if images_dir and not Path(images_dir).exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if masks_dir and not Path(masks_dir).exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)

        if job_type == "pixels":
            # batch_profile_pixels requires file_paths list
            search_pattern_tif = os.path.join(images_dir, "*.tif")
            search_pattern_nd2 = os.path.join(images_dir, "*.nd2")
            file_paths = glob.glob(search_pattern_tif) + glob.glob(search_pattern_nd2)
            if not file_paths:
                raise FileNotFoundError(f"No .tif or .nd2 files found in images directory: {images_dir}")
            fn_kwargs["file_paths"] = file_paths
            if out_dir and "dataframe_output_path" not in fn_kwargs:
                fn_kwargs["dataframe_output_path"] = str(Path(out_dir) / "pixel_profiles.parquet")
            result = fn(**fn_kwargs)
        else:
            if out_dir:
                if "dataframe_output_path" not in fn_kwargs:
                    fn_kwargs["dataframe_output_path"] = str(Path(out_dir) / f"{job_type}_summary.parquet")
                if crop_zarrs and "crop_output_path" not in fn_kwargs:
                    fn_kwargs["crop_output_dir"] = out_dir
                # Pass output explicitly back to fn_kwargs so the folder handler can use it
                fn_kwargs["output"] = out_dir

            result = fn(images_directory=images_dir, masks_directory=masks_dir, **fn_kwargs)

    elif mode == "zarr":
        input_zarr = fn_kwargs.pop("input", None)
        if not input_zarr or not Path(input_zarr).exists():
            raise FileNotFoundError(f"Zarr store input not found: {input_zarr}")
        
        input_path = Path(input_zarr)
        if input_path.is_dir() and not str(input_path).endswith('.zarr'):
            zarr_stores = list(input_path.glob("*.zarr"))
            if not zarr_stores:
                raise FileNotFoundError(f"No .zarr stores found in directory: {input_zarr}")
            
            console.print(f"\n[bold cyan]Found {len(zarr_stores)} Zarr stores to process.[/]")
            for idx, zs in enumerate(zarr_stores, 1):
                console.print(f"[bold blue]──▶ Processing {idx}/{len(zarr_stores)}: {zs.name}[/]")
                fn(zarr_path=str(zs), **fn_kwargs)
        else:
            result = fn(zarr_path=input_zarr, **fn_kwargs)

    return chain_output_path


def run_pipeline(config_path: str | Path, dry_run: bool = False) -> None:
    """
    Parse and execute a multi-step job pipeline from a TOML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the TOML job configuration file.
    dry_run : bool, default False
        If True, prints execution plan without executing jobs.
    """
    config = load_job_config(config_path)
    defaults = config.get("defaults", {})
    jobs = config.get("jobs", [])

    header = f"cellstream Pipeline {'[DRY-RUN]' if dry_run else ''}"
    console.print(Panel(f"[bold green]{header}[/]\nConfig: [dim]{config_path}[/]", expand=False))

    previous_output: Optional[str] = None
    total_jobs = len(jobs)

    for idx, job_entry in enumerate(jobs, start=1):
        console.print(f"[bold blue]Step [{idx}/{total_jobs}]:[/] Processing job type='[yellow]{job_entry.get('type')}[/]'...")
        
        job_type, mode, resolved_params = resolve_job_params(job_entry, defaults, previous_output)
        
        previous_output = run_job(job_type, mode, resolved_params, dry_run=dry_run)
        
        if not dry_run:
            console.print(f"[bold green][OK] Step [{idx}/{total_jobs}] completed successfully.[/]\n")

    console.print(f"[bold green][OK] All {total_jobs} job(s) finished.[/]")

