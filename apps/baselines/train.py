import os
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from delphi import distributed
from delphi.baselines import ethos, motor
from delphi.data.mimic import MIMICDataset
from delphi.data.ukb import UKBDataset
from delphi.env import DELPHI_CKPT_DIR, DELPHI_DATA_DIR
from delphi.experiment.config import TrainBaseConfig
from delphi.experiment.train import BaseTrainer
from delphi.log import TrainLogConfig
from delphi.model import delphi
from delphi.model.config import GPT2Config, parse_token_list
from delphi.optim import OptimConfig
from delphi.tokenizer import Tokenizer, update_tokenizer


@dataclass
class TrainConfig(TrainBaseConfig):

    model_type: str = "ethos"
    data: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)

    optim: OptimConfig = field(default_factory=OptimConfig)
    log: TrainLogConfig = field(default_factory=TrainLogConfig)


def experiment(cfg: TrainConfig):

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(
        f"Environment vars: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}"
    )
    backend = distributed.make_backend_from_args(cfg)

    if cfg.data["data_dir"] == "ukb_real_data":
        common_args = {
            "data_dir": cfg.data["data_dir"],
            "seed": cfg.data["seed"],
            "no_event_interval": cfg.data["no_event_interval"],
            "block_size": cfg.data["block_size"],
        }
        if cfg.model_type == "ethos":
            train_ds = ethos.UKBDataset(
                **common_args,
                subject_list=cfg.data["train_subject_list"],
                time_bins=cfg.data["time_bins"],
            )
            val_ds = ethos.UKBDataset(
                **common_args,
                subject_list=cfg.data["train_subject_list"],
                time_bins=cfg.data["time_bins"],
            )
        else:
            train_ds = UKBDataset(
                **common_args,
                subject_list=cfg.data["train_subject_list"],
            )
            val_ds = UKBDataset(
                **common_args,
                subject_list=cfg.data["val_subject_list"],
            )
    elif cfg.data["data_dir"] == "mimic":
        sep_time_tokens = cfg.model_type == "delphi"
        train_ds = MIMICDataset(
            input_dir=Path(DELPHI_DATA_DIR) / "mimic" / "train",
            n_positions=cfg.data["block_size"],
            sep_time_tokens=sep_time_tokens,
        )
        val_ds = MIMICDataset(
            input_dir=Path(DELPHI_DATA_DIR) / "mimic" / "test",
            n_positions=cfg.data["block_size"],
            sep_time_tokens=sep_time_tokens,
        )
    else:
        raise ValueError

    if cfg.model_type == "ethos":
        model_cls = ethos.Model
        model_cfg_cls = GPT2Config
    elif cfg.model_type == "motor":
        if cfg.model["motor_task_tokens"] != "all":
            motor_task_tokens = parse_token_list(cfg.model["motor_task_tokens"])
            cfg.model["motor_task_tokens"] = train_ds.tokenizer.encode(motor_task_tokens)  # type: ignore
        else:
            cfg.model["motor_task_tokens"] = list(range(1, train_ds.vocab_size))

        if cfg.model["motor_pieces"] is None:
            rng = np.random.default_rng(cfg.seed)
            sample_idx = rng.permutation(np.arange(len(train_ds)))[:10000]
            X, T = train_ds.get_batch(sample_idx)

            pieces = motor.estimate_pieces(
                X=X,
                T=T,
                task_tokens=cfg.model["motor_task_tokens"],  # type: ignore
                n_pieces=cfg.model["motor_n_pieces"],
                vocab_size=train_ds.vocab_size,
            )
            cfg.model["motor_pieces"] = pieces.tolist()
        print(f"motor time pieces:")
        for i in range(len(pieces) - 1):
            print(f"\t- {pieces[i]} – {pieces[i+1]}")

        model_cls = motor.Model
        model_cfg_cls = motor.ModelConfig
    elif cfg.model_type == "delphi":
        model_cls = delphi.Model
        model_cfg_cls = delphi.ModelConfig
    else:
        raise ValueError

    model_cfg = model_cfg_cls(**cfg.model)
    if cfg.init_from == "scratch":
        model = model_cls(model_cfg)  # type: ignore
    else:
        raise NotImplementedError

    trainer = BaseTrainer(
        cfg=cfg,
        backend=backend,
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
    )

    trainer.train()

    backend.finalize()


def main():

    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainConfig())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    # convert structured config back to underlying dataclass
    cfg = OmegaConf.to_object(cfg)

    experiment(cfg)  # type: ignore


if __name__ == "__main__":
    main()
