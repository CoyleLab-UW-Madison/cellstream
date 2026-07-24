# Unpushed Changes Summary
This document summarizes the changes incorporated since the last official commit on `origin/main` (online GitHub repository) for both the core `cellstream` codebase and the `napari-cellstream` plugin.

## 🧠 Core `cellstream` Codebase Updates
*(~32 unpushed commits on local branch compared to online remote)*

### Architecture & Restructuring
* **Phase & Flow Merge:** Completely overhauled the architecture by merging the `flow/` module into `phase/` and creating a unified `cellstream.features` module. This centralizes all phase-field and flow analysis, making the API significantly cleaner and easier to maintain.
* **Module Cleanup:** Moved `generate_phase_features` to utility files, fixed device defaults, and added comprehensive shape validation for Zarr schema handling.

### Advanced Flow Dynamics (FTLE & Streamlines)
* **High-Performance Tracing:** Replaced the older PIV approach with a GPU-accelerated particle streamline tracer.
* **FTLE Computation:** Added capabilities to compute both **Forward and Backward FTLE** (Finite-Time Lyapunov Exponent) fields.
* **Advanced Streamline Models:** Introduced sophisticated streamline features like phase-colored particles, static streamlines, and a birth-rate particle injection model. Includes `stream_mask` support to restrict and respawn particles inside cells.

### Spatial Cropping & Zarr Integration
* **Spatial Cropping:** Added spatial tools like `crop_zarr_from_masks` along with thumbnail generation.
* **Raw Datastreams:** Integrated raw datastreams and mean expression levels directly into Zarr crop outputs.
* **Handling Enhancements:** Corrected boolean masks, resolved phase array searching logic for 4D CWT shapes, and supported Zarr stores where cells are at the root level.

### Performance & Optimization
* **CWT Decoupling:** Drastically accelerated CWT metadata processing by pre-aggregating temporal statistics and fully decoupling the stats engine from Pandas using tensor means.
* **Lookups:** Pre-computed `scale_to_freq` lookups once and used advanced indexing in `query_cwt_block`.

---

## 🎨 `napari-cellstream` Plugin Updates
*(~19 unpushed commits on local branch compared to online remote)*

### Phase Velocity & Image Tools UI
* **New Extraction Tool:** Added a brand new Phase Velocity extraction tool to the Image Tools widget stack, exposing the new GPU streamline tracer directly in Napari.
* **Visualizations:** Added options for visualizing **Transport Highways**, **Static Streamlines**, and generating dense angle/magnitude images.
* **FTLE & Vectors:** Added built-in toggles for Forward and Backward FTLE visualizations. Aligned phase velocity vectors to the 4D Napari grid scales and color maps.

### Phase Features Widget Consolidation
* **Widget Merging:** Merged the older `phase_defects_widget` into a unified `phase_features_widget`.
* **API Delegation:** The unified widget now delegates completely to the refactored `cellstream` core API for its logic, drastically simplifying the UI code layer.

### Results Routing & Zarr Saving
* **Unified Results:** Routed all phase velocity outputs through the central Results Tree.
* **Seamless Saving:** Added a new `save_to_zarr` toggle, allowing you to easily save pipeline outputs right back to your source Zarr stores straight from the UI.

---

*(Note: There are also a few unstaged working changes and new example scripts like `process_zarr_crops.py` currently sitting in the local directories.)*
