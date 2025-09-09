import os
from dataclasses import asdict, dataclass, field

import torch
import yaml
from omegaconf import OmegaConf

from delphi import distributed, legacy
from delphi.baselines import ethos, motor
from delphi.data.ukb import UKBDataset
from delphi.log import TrainLogConfig
from delphi.model import delphi
from delphi.model.config import GPT2Config
from delphi.optim import OptimConfig
from delphi.train import BaseTrainer, TrainBaseConfig


@dataclass
class TrainConfig(TrainBaseConfig):

    data_dir: str = "ukb_real_data"
    train_subject_list: str = "participants/train_fold.bin"
    val_subject_list: str = "participants/val_fold.bin"
    seed: int = 42
    no_event_interval: float = 5.0 * 365.25
    augment_lifestyle: bool = True
    time_bins: str | list = "config/time_interval/ukb-ethos-10token-preset.yaml"

    model_type: str = "ethos"
    model: dict = field(default_factory=dict)

    optim: OptimConfig = field(default_factory=OptimConfig)
    log: TrainLogConfig = field(default_factory=TrainLogConfig)


def experiment(cfg: TrainConfig):

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(
        f"Environment vars: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}"
    )
    backend = distributed.make_backend_from_args(cfg)

    data_args = {
        "data_dir": cfg.data_dir,
        "seed": cfg.seed,
        "no_event_interval": cfg.no_event_interval,
        "block_size": cfg.model["block_size"],
    }
    if cfg.model_type == "ethos":
        if isinstance(cfg.time_bins, str):
            with open(cfg.time_bins, "r") as f:
                data_args["time_bins"] = yaml.safe_load(f)
        else:
            data_args["time_bins"] = cfg.time_bins
        train_ds = ethos.UKBDataset(
            **data_args,
            subject_list=cfg.train_subject_list,
        )
        val_ds = ethos.UKBDataset(
            **data_args,
            subject_list=cfg.val_subject_list,
        )
    else:
        train_ds = UKBDataset(
            **data_args,
            subject_list=cfg.train_subject_list,
        )
        val_ds = UKBDataset(
            **data_args,
            subject_list=cfg.val_subject_list,
        )

    if cfg.model_type == "ethos":
        model_cls = ethos.Model
        model_cfg_cls = GPT2Config
    elif cfg.model_type == "motor":
        model_cls = motor.Model
        model_cfg_cls = motor.ModelConfig
    elif cfg.model_type == "delphi":
        model_cls = delphi.Model
        model_cfg_cls = delphi.ModelConfig
    elif cfg.model_type == "delphi-2m":
        model_cls = legacy.model.Delphi
        model_cfg_cls = legacy.model.DelphiConfig
    else:
        raise ValueError

    model_cfg = model_cfg_cls(**cfg.model)
    if cfg.init_from == "scratch":
        model = model_cls(model_cfg)  # type: ignore
    else:
        raise NotImplementedError
    cfg.model = asdict(model_cfg)

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
