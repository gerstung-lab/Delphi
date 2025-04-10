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

# --- Check for script argument ---
if [ $# -lt 1 ]; then
  echo "Usage: ./run_script.sh <script.py> [key=value args...]"
  exit 1
fi

SCRIPT=$1
shift  # Shift arguments so "$@" only contains key=value pairs

# --- Run Python script with remaining args ---
echo "running command:"
echo python "$SCRIPT" "$@"
python "$SCRIPT" "$@"
