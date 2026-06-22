# Delphi Torch (refactor)

This folder contains a clean, PyTorch‑aligned training pipeline with
Pydantic configs, Dataset/DataLoader, explicit losses, and a small
orchestrator. The original code in the repo is untouched.

## Setup

```bash
cd delphi_torch
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[wandb]"
```

## Run (demo config)

```bash
python -m delphi_torch.run --config config.demo.toml
```

The demo config points at `data/ukb_simulated_data` and mirrors the original
`config/train_delphi_demo.py` defaults.

## Config

Use `.toml` or `.json`. See `delphi_torch/config.demo.toml` for a working example.

Key sections:
- `[data]`: dataset + DataLoader settings
- `[model]`: model architecture + masking settings
- `[train]`: optimization, device, and schedule

Note: keep `[data].block_size` and `[model].block_size` in sync.

## W&B run naming

You can use `{timestamp}` in `wandb_run_name` for unique runs, e.g.:

```toml
wandb_run_name = "run-{timestamp}"
```

## ONNX Export

Export a trained model to ONNX format for deployment with ONNX Runtime or other inference engines.

### Export

```bash
# Install onnx (optional, for validation)
pip install onnx onnxruntime

# Export model
python export_onnx.py --checkpoint Delphi-2M/ckpt.pt --output delphi.onnx
```

Options:
- `--checkpoint, -c`: Path to checkpoint file (required)
- `--output, -o`: Output ONNX file path (default: `delphi.onnx`)
- `--opset`: ONNX opset version (default: 17)
- `--no-dynamic`: Disable dynamic axes (fixed batch/seq dimensions)

### Inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("delphi.onnx")

# Prepare inputs
idx = np.array([[1, 5, 10, 15]], dtype=np.int64)  # token IDs
age = np.array([[0, 365, 730, 1095]], dtype=np.float32)  # ages in days

# Run inference
logits = session.run(["logits"], {"idx": idx, "age": age})[0]
print(logits.shape)  # (1, 4, vocab_size)

# Get predictions
next_token_logits = logits[0, -1, :]  # logits for next token
predicted_token = np.argmax(next_token_logits)
```

### Model Inputs/Outputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| **idx** (input) | int64 | (batch, seq_len) | Token IDs (medical event codes) |
| **age** (input) | float32 | (batch, seq_len) | Age in days at each position |
| **logits** (output) | float32 | (batch, seq_len, vocab_size) | Raw logits for next token prediction |

### Notes

- The exported model is for **inference only** (no dropout, no gradient computation)
- Dynamic axes are enabled by default, allowing variable batch size and sequence length
- For sampling trajectories, you'll need to implement the age-aware sampling loop externally (see `model.py:generate()` in the original code for reference)

### Inference Notebook

See `inference_onnx.ipynb` for a complete example with:
- Creating patient health timelines from human-readable events
- Converting data to model input format
- Running inference and interpreting predictions
- Calculating disease risk probabilities
