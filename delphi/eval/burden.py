import json
import os
from dataclasses import dataclass

import numpy as np
import yaml
from dacite import from_dict

from apps.generate import GenConfig
from delphi.data.dataset import Dataset
from delphi.eval import eval_task


@dataclass
class BurdenArgs:
    gen_name: str = "burden"
    box_plot: bool = True


def describe_stats(
    burden: np.ndarray,
) -> dict:

    stats_dict = {
        "mean": float(np.mean(burden)),
        "std": float(np.std(burden)),
        "max": int(np.max(burden)),
        "min": int(np.min(burden)),
    }
    return stats_dict


@eval_task.register
def run_burden_eval(
    task_args: BurdenArgs,
    task_name: str,
    task_input: str,
    ckpt: str,
    **kwargs,
) -> None:

    with open(os.path.join(ckpt, task_input, "config.yaml"), "r") as file:
        gen_cfg = yaml.safe_load(file)
    gen_cfg = from_dict(GenConfig, gen_cfg)

    gen_token_path = os.path.join(ckpt, gen_cfg.name, "gen.bin")
    assert os.path.exists(gen_token_path)
    "gen.bin not found in the checkpoint directory"

    gen_np = np.fromfile(gen_token_path, dtype=np.uint32).reshape(-1, 3)
    _, gen_burden = np.unique(gen_np[:, 0], return_counts=True)

    gen_np = gen_np[gen_np[:, 1] != 1, ...]
    _, gen_burden_wo_nil = np.unique(gen_np[:, 0], return_counts=True)

    ds = Dataset(cfg=gen_cfg.data)
    real_burden = ds.seq_len

    stats_dict = {
        "gen_burden": describe_stats(gen_burden),
        "gen_burden_without_no_events": describe_stats(gen_burden_wo_nil),
        "real_burden": describe_stats(real_burden),
    }

    dump_dir = os.path.join(ckpt, task_input, task_name)
    os.makedirs(dump_dir, exist_ok=True)
    with open(os.path.join(dump_dir, "disease_burden.json"), "w") as f:
        json.dump(stats_dict, f, indent=4)

    if task_args.box_plot:
        pass
    # TODO: complete box plot

    pass
