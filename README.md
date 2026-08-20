# cellstream

Fast image analysis tools for digital signal processing of single-cell data streams 

`cellstream` is a PyTorch-accelerated Python image processing package that provides a suite of tools for single-cell analysis of frequency-domain and time-frequency domain features in fluorescence microscopy data. Initially designed for use with [programmable reaction diffusion systems](https://www.cell.com/cell/fulltext/S0092-8674(23)01339-9?uuid=uuid%3Ab0fd3eba-81d4-4a72-86ea-1f65d8adf288) and [genetically-encoded oscillator circuits](https://www.biorxiv.org/content/10.1101/2025.02.28.640587v1) (GEOs), the tools can also be applied to a wide range of dynamic cellular systems. Continuous wavelet transforms (CWT) make use of the excellent [ssqueezepy](https://github.com/OverLordGoldDragon/ssqueezepy) package. GPU functionality is available but not required.

---

## Key Features

- **Pixel-Level Signal Processing**: Fast FFT, CWT, Hilbert, and filter-bank transforms of large image stacks.
- **Image Sequence Utilities**: Optimized methods for temporal re-coloring, downsampling, and along-axis convolution.
- **Single-Cell Extraction**: Efficient per-cell extraction of multidimensional time-series and summary statistics using segmentation masks.
- **Zarr-Based Data Management**: Chunked reading and writing for out-of-core processing of large microscopy datasets.

---

## Installation

### Interactive Exploration
For interactive data exploration and visualization, we highly recommend using our **[napari-cellstream](https://github.com/CoyleLab-UW-Madison/napari-cellstream)** plugin. Conda environment files are provided for one-step setup:

- **GPU / CUDA systems:**
  ```bash
  conda env create -f environment_cuda.yml
  conda activate cellstream-env
  ```

- **CPU-only / macOS systems:**
  ```bash
  conda env create -f environment_cpu.yml
  conda activate cellstream-cpu-env
  ```

### Headless Batch Processing
For high-throughput or remote server environments, a `Dockerfile` is provided for running `cellstream` headlessly with full GPU acceleration. 

### Manual Installation
This package requires PyTorch and `torch-scatter` for full functionality. These must be installed to match your system and GPU configuration. See the [PyTorch](https://pytorch.org/) and [torch-scatter](https://github.com/rusty1s/pytorch_scatter) installation instructions for guidance.

Once dependencies are configured, install the package with:

```bash
pip install git+https://github.com/CoyleLab-UW-Madison/cellstream
```

and for the **[napari-cellstream](https://github.com/CoyleLab-UW-Madison/napari-cellstream)** plugin:

```bash
pip install git+https://github.com/CoyleLab-UW-Madison/napari-cellstream
```

Note this package is still in early stages of development.

## Example Usage

We include raw input data and specific examples of generating FFT/CWT features, single-cell datastream extraction, and data visualization in the "examples" folder of this repository.

A Google Colab notebook demoing the core `cellstream` functionalities with example data is also availble [here](https://colab.research.google.com/drive/1IKTQLDbRJS1Yl-Au3Fsxp6Y7WZNqAvjs?usp=sharing)
