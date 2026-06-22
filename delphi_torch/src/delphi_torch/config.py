from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    dataset: str = "ukb_simulated_data"
    batch_size: int = 128
    block_size: int = 48
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = True
    data_fraction: float = 1.0
    padding: Literal["regular", "random", "none"] = "regular"
    select: Literal["left", "right", "random"] = "left"
    lifestyle_augmentations: bool = True
    no_event_token_rate: int = 5


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_size: int = 48
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 120
    dropout: float = 0.1
    token_dropout: float = 0.0
    bias: bool = False
    vocab_size: int = 1270
    t_min: float = 0.1
    mask_ties: bool = True
    ignore_tokens: list[int] = Field(
        default_factory=lambda: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    
    # Note: Hierarchical, TimeDelta, and Enhanced embeddings were removed.
    # The original Delphi architecture (simple token + sinusoidal age encoding)
    # is sufficient and avoids unnecessary complexity.


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    out_dir: Path = Path("out")
    seed: int = 42
    max_steps: int = 5000
    eval_interval: int = 250
    eval_iters: int = 25
    log_interval: int = 25

    learning_rate: float = 2e-3
    weight_decay: float = 2e-1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0

    warmup_steps: int = 500
    lr_decay_steps: int = 5000
    min_lr: float = 2e-4

    device: str = "cpu"
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    compile: bool = False
    wandb_log: bool = False
    wandb_project: str = "delphi"
    wandb_run_name: str = "run"
    early_stop: bool = False
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0
    early_stop_min_steps: int = 0


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()


def load_config(path: Path | None) -> AppConfig:
    if path is None:
        return AppConfig()

    if path.suffix == ".toml":
        import tomllib

        payload = tomllib.loads(path.read_text())
    elif path.suffix == ".json":
        import json

        payload = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    return AppConfig.model_validate(payload)
