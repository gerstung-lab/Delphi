import os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from generate import GeneratorConfig, load_model, load_tokenizer
from omegaconf import OmegaConf
from tqdm import tqdm

from delphi.eval.auc import AUCConfig, run_auc_eval
from delphi.utils import get_batch, get_p2i


class TaskType(Enum):
    AUC = "auc"
    CALIBRATION = "calibration"


@dataclass
class TaskConfig:
    task_name: str
    task_type: str
    task_config: AUCConfig


@dataclass
class EvalConfig:
    name: str = "debug"
    ckpt_path: str = "checkpoints/Delphi-demo"
    tasks: Optional[List[TaskConfig]] = None


def eval(cfg: EvalConfig):

    model, _ = load_model(cfg.ckpt_path)
    tokenizer = load_tokenizer(cfg.ckpt_path)

    eval_dump_dir = os.path.join(cfg.ckpt_path, cfg.name)

    for task in cfg.tasks:

        task_type = TaskType(task.task_type)

        if task_type == TaskType.AUC:
            task_dump_dir = os.path.join(eval_dump_dir, task.task_name)
            os.makedirs(task_dump_dir, exist_ok=True)
            run_auc_eval(
                model=model,
                tokenizer=tokenizer,
                task_cfg=task.task_config,
                dump_dir=os.path.join(task_dump_dir),
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")


def main():

    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(EvalConfig)
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    eval(cfg)

    pass


if __name__ == "__main__":
    main()
