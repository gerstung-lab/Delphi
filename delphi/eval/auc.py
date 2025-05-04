import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dacite import from_dict
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from apps.generate import GenConfig
from delphi import DAYS_PER_YEAR
from delphi.data.dataset import tricolumnar_to_2d
from delphi.data.trajectory import DiseaseRateTrajectory
from delphi.eval import eval_task
from delphi.tokenizer import Gender, Tokenizer


@dataclass
class AgeGroups:
    groups: Optional[list[int]] = None
    start: int = 40
    end: int = 85
    step: int = 5


@dataclass
class CalibrateAUCArgs:
    disease_lst: str = ""
    time_offset: AgeGroups = field(default_factory=AgeGroups)
    age_groups: AgeGroups = field(default_factory=AgeGroups)
    min_time_gap: float = 0.1
    box_plot: bool = True


def parse_diseases(diseases: str):

    assert os.path.exists(diseases)
    with open(diseases, "r") as file:
        diseases = yaml.safe_load(file)

    return diseases


def parse_age_groups(age_groups: AgeGroups) -> np.ndarray:
    if age_groups.groups is None:
        return np.arange(
            age_groups.start,
            age_groups.end,
            age_groups.step,
        )
    else:
        return np.array(age_groups.groups)


def mann_whitney_auc(x1: np.ndarray, x2: np.ndarray) -> float:

    n1 = len(x1)
    n2 = len(x2)
    R1 = np.concatenate([x1, x2]).argsort().argsort()[:n1].sum() + n1
    U1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    if n1 == 0 or n2 == 0:
        return np.nan
    return U1 / n1 / n2


def box_plot_disease_rates(
    age_buckets: Sequence[tuple[int, int]],
    disease_rates: Sequence[tuple[np.ndarray, np.ndarray]],
    auc_vals: Sequence[float],
) -> Tuple[Figure, Axes]:

    fig, box_ax = plt.subplots(
        nrows=1,
        ncols=len(disease_rates),
        figsize=(20 / 8 * len(disease_rates), 1.5),
        sharex=True,
        sharey=False,
        # height_ratios=[1, 0.5]
    )
    box_ax = box_ax[np.newaxis, :]
    axf = box_ax.ravel()
    for i, ax in enumerate(axf):

        ax.text(
            0.5,
            0.8,
            s=f"AUC={auc_vals[i]:.2}",
            transform=ax.transAxes,
            va="center",
            ha="center",
        )

        ctl_disease_rates, dis_disease_rates = disease_rates[i]
        ax.boxplot(
            [ctl_disease_rates, dis_disease_rates],
            vert=False,
            sym=".",
            widths=0.5,
            whis=(5, 95),
            flierprops=dict(marker=".", markeredgecolor="white", markerfacecolor="k"),
        )
        ax.set_xscale("log")
        ax.set_xlim((1e-5, 1))
        ax.set_xlabel(f"Predicted rate [1/yr]")

        ylabels = ["healthy", "diseased"] if i == 0 else ["", ""]
        ax.set_yticks((1, 2), ylabels)
        ax.set_ylim((0.5, 3.5))

        age_start, age_end = age_buckets[i]
        ax.text(
            0.5,
            1,
            s=f"{age_start}-{age_end}yr",
            transform=axf[i].transAxes,
            va="bottom",
            ha="center",
            weight="bold",
        )

    fig.tight_layout()
    fig.set_dpi(300)

    return fig, ax


def auc_by_age_group(
    disease_token: int,
    val_trajectories: DiseaseRateTrajectory,
    offset: float,
    task_args: CalibrateAUCArgs,
) -> tuple:

    age_groups = parse_age_groups(task_args.age_groups)
    age_buckets = [(i, j) for i, j in zip(age_groups[:-1], age_groups[1:])]
    l = len(age_buckets)

    ctl_counts = np.zeros(l)
    dis_counts = np.zeros(l)
    auc_vals = np.zeros(l)
    token_rates = []

    for i, (age_start, age_end) in enumerate(age_buckets):

        tw = (age_start * DAYS_PER_YEAR, age_end * DAYS_PER_YEAR)
        offset_tw = (age_start + offset) * DAYS_PER_YEAR, (
            age_end + offset
        ) * DAYS_PER_YEAR
        has_valid_pred_in_tw = val_trajectories.has_any_valid_predictions(
            min_time_gap=task_args.min_time_gap,
            t0_range=tw,
        )
        has_dis_in_offset_tw = val_trajectories.has_token(
            disease_token,
            token_type="target",
            t0_range=offset_tw,
        )
        disease_free = ~val_trajectories.has_token(
            disease_token,
            token_type="target",
        )

        ctl_paths = val_trajectories[disease_free & has_valid_pred_in_tw]
        dis_paths = val_trajectories[has_dis_in_offset_tw & has_valid_pred_in_tw]
        ctl_counts[i] = ctl_paths.n_participants
        dis_counts[i] = dis_paths.n_participants

        _, ctl_token_rates = ctl_paths.disease_rate(t0_range=tw, keep="average")
        _, dis_token_rates = dis_paths.disease_rate(t0_range=tw, keep="average")
        auc_vals[i] = mann_whitney_auc(ctl_token_rates.ravel(), dis_token_rates.ravel())
        token_rates.append((ctl_token_rates, dis_token_rates))

    return age_buckets, auc_vals, ctl_counts, dis_counts


@eval_task.register
def calibrate_auc(
    task_args: CalibrateAUCArgs,
    task_name: str,
    task_input: str,
    ckpt: str,
    tokenizer: Tokenizer,
    **kwargs,
) -> None:

    with open(os.path.join(ckpt, task_input, "config.yaml"), "r") as file:
        gen_cfg = yaml.safe_load(file)
    gen_cfg = from_dict(GenConfig, gen_cfg)

    task_dump_dir = os.path.join(ckpt, task_input, task_name)
    os.makedirs(task_dump_dir, exist_ok=True)

    gen_logits_path = os.path.join(ckpt, task_input, "logits.bin")
    assert os.path.exists(gen_logits_path)
    "logits.bin not found in the checkpoint directory"
    gen_path = os.path.join(ckpt, task_input, "gen.bin")
    assert os.path.exists(gen_path)
    "gen.bin not found in the checkpoint directory"

    logits = np.fromfile(gen_logits_path, dtype=np.float16).reshape(
        -1, tokenizer.vocab_size + 1
    )
    XT = np.fromfile(gen_path, dtype=np.uint32).reshape(-1, 3)

    X, T = tricolumnar_to_2d(XT)
    X_t0, X_t1 = X[:, :-1], X[:, 1:]
    T_t0, T_t1 = T[:, :-1], T[:, 1:]

    diseases = parse_diseases(task_args.disease_lst)
    time_offsets = parse_age_groups(task_args.time_offset)

    for disease in diseases:

        dis_token = tokenizer[disease]

        Y = np.zeros_like(X, dtype=np.float16)
        sub_idx, pos_idx = np.nonzero(X)
        Y[sub_idx, pos_idx] = logits[:, dis_token]
        Y = np.exp(Y) * DAYS_PER_YEAR
        Y = 1 - np.exp(-Y)

        Y_t1 = Y[:, :-1]
        traj = DiseaseRateTrajectory(
            X_t0=X_t0,
            T_t0=T_t0,
            X_t1=X_t1,
            T_t1=T_t1,
            Y_t1=Y_t1,
        )

        is_female = traj.has_token(tokenizer[Gender.FEMALE.value], token_type="input")
        is_male = traj.has_token(tokenizer[Gender.MALE.value], token_type="input")
        is_gender_dict = {
            "female": is_female,
            "male": is_male,
            "either": is_female | is_male,
        }

        for time_offset in time_offsets:

            for gender, is_gender in is_gender_dict.items():

                age_buckets, auc_vals, ctl_counts, dis_counts = auc_by_age_group(
                    disease_token=dis_token,
                    val_trajectories=traj[is_gender],
                    offset=time_offset,
                    task_args=task_args,
                )

                column_types = {
                    "age_group": "string",
                    "auc": "float32",
                    "ctl_counts": "uint32",
                    "dis_counts": "uint32",
                }
                df = pd.DataFrame(
                    {
                        "age_group": [f"{i}-{j}" for i, j in age_buckets],
                        "auc": auc_vals,
                        "ctl_counts": ctl_counts,
                        "dis_counts": dis_counts,
                    }
                ).astype(column_types)
                df.to_csv(
                    os.path.join(
                        task_dump_dir, f"{disease}_{gender}_{time_offset}.csv"
                    ),
                    index=False,
                    float_format="%.3f",
                )
                # fig, ax = box_plot_disease_rates(
                #     age_buckets=age_buckets,
                #     disease_rates=token_rates,
                #     auc_vals=list(auc_vals)
                # )

                # fig.savefig(os.path.join(task_dump_dir, f"{disease}_{gender}_{time_offset}.png"))
