import os
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from delphi.baselines import ethos, motor
from delphi.data import core
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import BaseTrainer, TrainBaseConfig
from delphi.log import TrainLogConfig
from delphi.model import delphi
from delphi.model.config import GPT2Config, parse_token_list
from delphi.optim import OptimConfig


@dataclass
class TrainConfig(TrainBaseConfig):

    model_type: str = "ethos"
    train_data: dict = field(default_factory=dict)
    val_data: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)

    optim: OptimConfig = field(default_factory=OptimConfig)
    log: TrainLogConfig = field(default_factory=TrainLogConfig)


def experiment(cfg: TrainConfig):

    run_dir = Path(DELPHI_CKPT_DIR) / cfg.log.run_name
    os.makedirs(run_dir, exist_ok=True)

    if cfg.model_type == "ethos":
        train_ds, val_ds = ethos.build_datasets(cfg.train_data, cfg.val_data)
        model_cls = ethos.Model
        model_cfg_cls = GPT2Config
        trainer_cls = ethos.Trainer
        loader = ethos.load_sequences
    elif cfg.model_type == "motor":
        train_ds, val_ds = core.build_datasets(cfg.train_data, cfg.val_data)
        motor_task_tokens = parse_token_list(cfg.model["motor_task_tokens"])
        cfg.model["motor_task_tokens"] = train_ds.tokenizer.encode(motor_task_tokens)  # type: ignore
        model_cls = motor.Model
        model_cfg_cls = motor.ModelConfig
        trainer_cls = BaseTrainer
        loader = core.load_sequences
    elif cfg.model_type == "delphi":
        train_ds, val_ds = core.build_datasets(cfg.train_data, cfg.val_data)
        model_cls = delphi.Model
        model_cfg_cls = delphi.ModelConfig
        trainer_cls = BaseTrainer
        loader = core.load_sequences
    else:
        raise ValueError

    tokenizer = train_ds.tokenizer
    model_cfg = model_cfg_cls(**cfg.model)
    if model_cfg.vocab_size is None:
        model_cfg.vocab_size = tokenizer.vocab_size
        print(
            f"\nvocab_size not set, using vocab size of training dataset's tokenizer: {model_cfg.vocab_size}"
        )
    else:
        print(f"vocab_size: {model_cfg.vocab_size}")
        assert (
            model_cfg.vocab_size == tokenizer.vocab_size
        ), f"inconsistent vocab size between tokenizer ({tokenizer.vocab_size}) and model"

    if cfg.init_from == "scratch":
        print("Initializing a new model from scratch")
        model = model_cls(model_cfg)  # type: ignore
    else:
        raise NotImplementedError

    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        loader=loader,
    )

    trainer.train()


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
