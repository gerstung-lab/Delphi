#####################################################################################################################
# Description: Delphi-docker
# Basic Dockerfile for Delphi container
# Including the Delphi main git branch and all dependencies
# Login is restricted to a non-root user:delphi_dev
# GPU / Cuda support is available
#####################################################################################################################
# Example Docker commands:
# Build: docker build -t delphi-env . 
# Run [No GPU]: docker run -it delphi-env bash
# Run [with GPU]: docker run --gpus all -it delphi-env bash
#####################################################################################################################
# Small training test command
# cd /Delphi
# python train.py config/train_delphi_demo.py --out_dir=/workspace/Delphi-demo-2M --max_iters=100 --eval_interval=25
#####################################################################################################################

# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.5.1-runtime-ubuntu22.04
#FROM nvidia/cuda:12.3.1-base-ubuntu22.04
#FROM nvidia/cuda:11.2.2-runtime-ubi8

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages and Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    wget \
    nano \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Get Delphi main
RUN git clone https://github.com/gerstung-lab/Delphi.git

# Make sure all dependancies are available
RUN pip install -r Delphi/requirements.txt

# Create python alias
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Create a non-root user
RUN useradd -m -s /bin/bash delphi_dev \
    && chown -R delphi_dev:delphi_dev /workspace

USER delphi_dev

# Set default command to bash
CMD ["/bin/bash"]
