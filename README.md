# cellstream

Fast image analysis tools for digital signal processing of single-cell data streams 

`cellstream` is a PyTorch-accelerated Python image processing package that provides a suite of tools for single-cell analysis of frequency-domain and time-frequency domain features in fluorescence microscopy data. Initially designed for use with [programmable reaction diffusion systems](https://www.cell.com/cell/fulltext/S0092-8674(23)01339-9?uuid=uuid%3Ab0fd3eba-81d4-4a72-86ea-1f65d8adf288) and [genetically-encoded oscillator circuits](https://www.biorxiv.org/content/10.1101/2025.02.28.640587v1) (GEOs), the tools can also be applied to a wide range of dynamic cellular systems. Continuous wavelet transforms (CWT) make use of the excellent [ssqueezepy](https://github.com/OverLordGoldDragon/ssqueezepy) package. GPU functionality is available but not required.

---

## Key Features

- Fast FFT / CWT / filter-bank transforms of large image stacks
- Optimized methods for downsampling, along-axis convolution, and memory-efficient filtering
- Fast multi-channel query tools for dynamic reporters that are coupled to carrier signals
- Efficient per-cell extraction and summarization of dynamic datastreams

---

## Installation

This package requires PyTorch and `torch-scatter` for full functionality. These must be installed manually to match your system and GPU configuration. See the [PyTorch](https://pytorch.org/) and [torch-scatter](https://github.com/rusty1s/pytorch_scatter) installation instructions for guidance.

Once dependencies are configured, install the package with:

```bash
pip install git+https://github.com/CoyleLab-UW-Madison/cellstream
```

Note this package is still in early stages of development.

## Example Usage

We include raw input data and specific examples of generating FFT/CWT features, single-cell datastream extraction, and data visualization in the "examples" folder of this repository.

A Google Colab notebook demoing the core `cellstream` functionalities with example data is also availble [here](https://colab.research.google.com/drive/1IKTQLDbRJS1Yl-Au3Fsxp6Y7WZNqAvjs?usp=sharing)