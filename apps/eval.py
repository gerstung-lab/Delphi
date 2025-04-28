import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from omegaconf import OmegaConf

from delphi.eval import clock, eval_task
from delphi.eval.auc import AUCArgs
from delphi.eval.burden import BurdenArgs
from delphi.model.transformer import load_model
from delphi.tokenizer import load_tokenizer_from_ckpt
from delphi.visualize.incidence import IncidencePlotConfig


class TaskType(Enum):
    AUC = "auc"
    CALIBRATION = "calibration"
    INCIDENCE = "incidence"
    BURDEN = "burden"


task_type_to_args_type = {
    TaskType.AUC: AUCArgs,
    TaskType.INCIDENCE: IncidencePlotConfig,
    TaskType.BURDEN: BurdenArgs,
}


@dataclass
class TaskConfig:
    task_name: str
    task_type: str
    task_input: str
    task_args: Any


@dataclass
class EvalConfig:
    tasks: List[TaskConfig] = field(default_factory=list)


@clock
def eval(cfg: EvalConfig, ckpt: str):

    model, _ = load_model(ckpt)
    tokenizer = load_tokenizer_from_ckpt(ckpt)

    for task in cfg.tasks:

        task_type = TaskType(task.task_type)
        args_type = task_type_to_args_type[task_type]
        default_args = OmegaConf.structured(args_type())
        task_args = OmegaConf.merge(default_args, task.task_args)
        task_args = OmegaConf.to_object(task_args)

        eval_task(
            task_args,
            task_name=task.task_name,
            task_input=task.task_input,
            ckpt=ckpt,
            model=model,
            tokenizer=tokenizer,
        )


def main():

    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config
    ckpt = cli_args.ckpt
    del cli_args.ckpt

    default_cfg = OmegaConf.structured(EvalConfig)
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    eval(cfg=cfg, ckpt=ckpt)  # type: ignore

    pass


if __name__ == "__main__":
    main()
