#!/bin/bash

source ~/.bashrc

# source env variables
set -a
source submit/dkfz/var.env
set +a

# conda env
micromamba activate delphi-cf-torch2.3

# cuda
CUDA=11.7
export CUDA_HOME=/usr/local/cuda-${CUDA}
export CUDA_CACHE_DISABLE=1

python apps/train.py config=config/train_delphi_demo.yaml device=cuda
