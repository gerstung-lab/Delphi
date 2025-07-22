<p align="center">
  <img src=".github/delphi-logo-white-bg.svg" width="400" alt="Delphi Logo"/>
</p>

## Learning the natural history of human disease with generative transformers

[![Paper](https://img.shields.io/badge/Paper-medRxiv-blue)](https://www.medrxiv.org/content/10.1101/2024.06.07.24308553v1)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit)

**Authors:** Artem Shmatko*, Alexander Wolfgang Jung*, Kumar Gaurav*, Søren Brunak, Laust Mortensen, Ewan Birney, Tom Fitzgerald, Moritz Gerstung (*Equal Contribution)

## Overview

This repository contains the code for **Delphi**, a modified GPT-2 model designed to learn the natural history of human disease using generative transformers. The implementation is based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and includes training code and analysis notebooks.

## Installation

### Option 1: Conda Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/gerstung-lab/Delphi.git
   cd Delphi
   ```

2. **Create and activate the conda environment:**

   ```bash
   conda create -n delphi python=3.11
   conda activate delphi
   pip install -r requirements.txt
   ```

   > **Note:** Installing requirements typically takes a few minutes.

### Option 2: Docker

We provide a Dockerfile for containerized training and downstream analyses. See `containers/Dockerfile` for implementation details.

## Data

### UK Biobank Access

Delphi-2M is trained on 500K patient health trajectories from the UK Biobank dataset. Access to this data requires a research application through the [UK Biobank](https://www.ukbiobank.ac.uk/).

### Data Preparation

For detailed instructions on preparing training data, please refer to [`data/README.md`](data/README.md).

## Configuration and Training

### Prerequisites

Set the following environment variables:

- `DELPHI_DATA_DIR`: Directory containing training and validation data
- `DELPHI_CKPT_DIR`: Directory for storing model checkpoints

> **Tip:** We recommend using a `.env` file with [direnv](https://direnv.net) for environment management.

### Training Configuration

Delphi uses OmegaConf for experiment configuration management. Here's an example configuration:

```yaml
# example.yaml
ckpt_dir: example # Saves to $DELPHI_CKPT_DIR/example
eval_interval: 25
eval_iters: 25
eval_only: false
init_from: scratch
seed: 42
gradient_accumulation_steps: 1
batch_size: 128
device: cuda # Options: cuda, cpu, mps

# Training data configuration
infer_train_biomarkers: true
train_data:
  data_dir: ukb_real_data # Loads from $DELPHI_DATA_DIR/ukb_real_data
  subject_list: ukb_real_data/participants/train_fold.bin
  seed: 42
  biomarker_dir: ukb_real_data/biomarkers
  expansion_pack_dir: ukb_real_data/expansion_packs
  expansion_packs:
    - prescriptions
    - summary_ops
  transforms:
    - name: no-event
      args:
        interval_in_years: 5
        mode: random

# Validation data configuration
infer_val_biomarkers: true
infer_val_expansion_packs: true
val_data:
  data_dir: ukb_real_data
  subject_list: ukb_real_data/participants/val_fold.bin
  seed: 42
  biomarker_dir: ukb_real_data/biomarkers
  expansion_pack_dir: ukb_real_data/expansion_packs
  transforms:
    - name: no-event
      args:
        interval_in_years: 5
        mode: random

ignore_expansion_tokens: true

# Model architecture
model:
  n_layer: 12
  n_head: 12
  n_embd: 120
  dropout: 0.1
  token_dropout: 0.0
  t_min: 0.1
  bias: true
  mask_ties: true
  ignore_tokens:
    - padding
    - male
    - female
    - config/disease_list/lifestyle.yaml
  biomarkers:
    prs:
      projector: linear
      input_size: 36
    wbc:
      projector: linear
      input_size: 31
  modality_emb: true
  loss:
    ce_beta: 1.0
    dt_beta: 1.0

# Logging configuration
log:
  wandb_log: true
  wandb_project: ${ckpt_dir}
  run_name: example
  log_interval: 25
  always_ckpt_after_eval: true
  ckpt_interval: null

# Optimization settings
optim:
  learning_rate: 6e-4
  max_iters: 100000
  weight_decay: 2e-1
  lr_decay_iters: 100000
  min_lr: 6e-5
  beta2: 0.99
  warmup_iters: 1000
```

### Training

Execute the following command to start training:

```bash
python train.py config=example.yaml
# Override specific parameters: python train.py config=example.yaml device=cuda
```

## Development

### Code Quality

This project uses [`pre-commit`](https://pre-commit.com) hooks to maintain code quality standards.

#### Setup Pre-commit Hooks

1. **Install pre-commit** (if not already available):

   ```bash
   # Via conda (recommended for base environment)
   conda install pre-commit
   # Or via Homebrew
   brew install pre-commit
   ```

2. **Install hooks in your local repository:**

   ```bash
   # From project root directory
   pre-commit install --install-hooks
   ```

3. **Manual execution** (optional):

   ```bash
   pre-commit run --all-files
   ```

## Citation

If you use this work, please cite our paper:

```bibtex
@article{Shmatko2024.06.07.24308553,
    title = {Learning the natural history of human disease with generative transformers},
    author = {Shmatko, Artem and Jung, Alexander Wolfgang and Gaurav, Kumar and Brunak, S{\o}ren and Mortensen, Laust and Birney, Ewan and Fitzgerald, Tom and Gerstung, Moritz},
    doi = {10.1101/2024.06.07.24308553},
    journal = {medRxiv},
    publisher = {Cold Spring Harbor Laboratory Press},
    year = {2024}
}
```

## License

This project is licensed under the MIT License - see the badge above for details.
