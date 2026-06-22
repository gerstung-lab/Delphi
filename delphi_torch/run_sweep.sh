#!/bin/bash
# Hyperparameter sweep script
# Note: MPS doesn't support true parallel GPU training, so we run sequentially
# Or you can run each on a different terminal

set -e

cd "$(dirname "$0")"

echo "========================================"
echo "Running Delphi Hyperparameter Sweep"
echo "========================================"

# Run sweep 1: Higher LR, more capacity
echo ""
echo "[1/3] Running Sweep 1: Higher LR, larger d_model..."
python -m delphi_torch.run --config config.sweep1.toml

# Run sweep 2: Lower LR, longer warmup
echo ""
echo "[2/3] Running Sweep 2: Smaller batch, slower LR decay..."
python -m delphi_torch.run --config config.sweep2.toml

# Run sweep 3: Baseline (no hierarchical)
echo ""
echo "[3/3] Running Sweep 3: Baseline without hierarchical embeddings..."
python -m delphi_torch.run --config config.sweep3.toml

echo ""
echo "========================================"
echo "Sweep complete! Check wandb for results."
echo "========================================"
