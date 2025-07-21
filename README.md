<p align="center">
  <img src=".github/delphi-logo-white-bg.svg" width="400"/>
</p>


## Learning the natural history of human disease with generative transformers

[[`Paper`](https://www.medrxiv.org/content/10.1101/2024.06.07.24308553v1)] [[`BibTeX`](#Citation)]

Artem Shmatko*, Alexander Wolfgang Jung*, Kumar Gaurav*, Søren Brunak, Laust Mortensen, Ewan Birney, Tom Fitzgerald, Moritz Gerstung (*Equal Contribution)

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit)


## Repository Overview

This repository contains the code for Delphi, the modified GPT-2 model used in the paper "Learning the natural history of human disease with generative transformers", along with the training code and analysis notebooks.

The implementation is based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## Installation

### Conda environment

1. Download the repository:

```bash
git clone https://github.com/gerstung-lab/Delphi.git
cd Delphi
```

2. Create a virtual conda environment and install the requirements:
```bash
conda create -n delphi python=3.11
conda activate delphi
pip install -r requirements.txt
```

Installing the requirements normally takes a few minutes.

### Docker

We provide a Dockerfile to run the training and downstream analyses. Please refer to the `containers/Dockerfile` for the details.

## Data

### UK Biobank availability

Delphi-2M is trained on 500K patient health trajectories from the UK Biobank data, which is available to researchers upon [application](https://www.ukbiobank.ac.uk/).

### Data preparation

Please refer to the `data/README.md` file for the details on how to prepare the data for training.

## Experiment

To train the model, run:

```bash
python train.py config/train_delphi_demo.py --device=cuda --out_dir=Delphi-2M
```

**If** you want to train the model on a CPU, remove the `--device=cuda` argument.
For more information on the training configuration, check the `config/train_delphi_demo.py` file.

Training a demo model takes around 10 minutes on a single GPU.

Training the original model took 1 GPU-hour (NVIDIA V100, CentOS 7). Training on M1 Macbook Pro's MPS takes around 10 hours.

## Contributing

We use [`pre-commit`](https://pre-commit.com) hooks to ensure high-quality code.
Make sure it's installed on the system where you're developing
(it is in the dependencies of the project, but you may be editing the code from outside the development environment.
If you have conda you can install it in your base environment, otherwise, you can install it with `brew`).
Install the pre-commit hooks with

```bash
# When in the PROJECT_ROOT.
pre-commit install --install-hooks
```

Then every time you commit, the pre-commit hooks will be triggered.
You can also trigger them manually with:

```bash
pre-commit run --all-files
```

## Citation

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
