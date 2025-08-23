import os
from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf

from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import clock, eval_task, mimic, ukb

task_type_to_args_type = {
    "ukb-auc": ukb.InstantAUCArgs,
    "ukb-forecast": ukb.ForecastArgs,
    "mimic-forecast": mimic.ForecastArgs,
    "mimic-drg": mimic.DRGClassificationArgs,
    "mimic-sofa": mimic.SofaArgs,
}


@dataclass
class TaskConfig:
    task_name: str
    task_type: str
    task_args: Any


@clock
def eval(cfg: TaskConfig, ckpt: str):

    ckpt = os.path.join(DELPHI_CKPT_DIR, ckpt)
    assert os.path.exists(ckpt), f"checkpoint {ckpt} does not exist."

    args_type = task_type_to_args_type[cfg.task_type]
    default_args = OmegaConf.structured(args_type())
    task_args = OmegaConf.merge(default_args, cfg.task_args)
    task_args = OmegaConf.to_object(task_args)

    eval_task(
        task_args,
        task_name=cfg.task_name,
        ckpt=ckpt,
    )


def main():

    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config
    ckpt = cli_args.ckpt
    del cli_args.ckpt

    default_cfg = OmegaConf.structured(TaskConfig)
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    eval(cfg=cfg, ckpt=ckpt)  # type: ignore

    pass


if __name__ == "__main__":
    main()
