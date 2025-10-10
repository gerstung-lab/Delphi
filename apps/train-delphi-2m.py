import os
from dataclasses import dataclass, field

import torch
from omegaconf import OmegaConf

from delphi import distributed
from delphi.data.ukb import UKBDataset
from delphi.model import Delphi2M, Delphi2MConfig
from delphi.train import BaseTrainer, TrainBaseConfig


@dataclass
class TrainConfig(TrainBaseConfig):

    data_dir: str = "ukb_real_data"
    train_subject_list: str = "participants/train_fold.bin"
    val_subject_list: str = "participants/val_fold.bin"
    seed: int = 42
    no_event_interval: float = 5.0 * 365.25
    no_event_mode: str = "legacy-random"
    exclude_lifestyle: bool = False
    augment_lifestyle: bool = True
    crop_mode: str = "right"
    fix_no_event_rate: bool = False
    model_type: str = "delphi-2m"
    model: Delphi2MConfig = field(default_factory=Delphi2MConfig)


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
        "no_event_mode": cfg.no_event_mode,
        "block_size": cfg.model.block_size,
        "crop_mode": cfg.crop_mode,
        "exclude": cfg.exclude_lifestyle,
    }
    train_ds = UKBDataset(
        **data_args,
        perturb=cfg.augment_lifestyle,
        subject_list=cfg.train_subject_list,
    )
    val_ds = UKBDataset(
        **data_args,
        perturb=False,
        subject_list=cfg.val_subject_list,
    )

    if cfg.fix_no_event_rate:
        cfg.model.no_event_rate = 1 / cfg.no_event_interval

    if cfg.init_from == "scratch":
        print(f"initializing {cfg.model_type} from scratch")
        model = Delphi2M(cfg.model)
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

    default_cfg = OmegaConf.structured(TrainConfig())
    cli_args = OmegaConf.from_cli()
    if hasattr(cli_args, "config"):
        file_cfg = OmegaConf.load(cli_args.config)
        del cli_args.config
    else:
        file_cfg = default_cfg
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    # convert structured config back to underlying dataclass
    cfg = OmegaConf.to_object(cfg)

    experiment(cfg)  # type: ignore


if __name__ == "__main__":
    main()
