# Use the official PyTorch image with CUDA support as the base
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Set environment variables to prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (git for cloning, build-essential just in case)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# This URL MUST match the PyTorch and CUDA versions of the base image (torch 2.2.0, cu118)
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# Install the rest of the dependencies
RUN pip install \
    "numpy<2" \
    pandas \
    scipy \
    cupy-cuda11x \
    matplotlib \
    tifffile \
    nd2 \
    zarr \
    tqdm \
    ssqueezepy \
    pystackreg

# Clone and install the cellstream repository
RUN git clone https://github.com/CoyleLab-UW-Madison/cellstream.git && \
    cd cellstream && \
    pip install -e .

# Example usage: docker run --gpus all -v /local/data:/app/data cellstream-headless python /app/cellstream/examples/cellstream_cwt_example___generate_cwts.py
CMD ["/bin/bash"]
