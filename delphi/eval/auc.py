import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dacite import from_dict
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from apps.forward import FeedForwardConfig
from delphi import DAYS_PER_YEAR
from delphi.data.dataset import tricolumnar_to_2d
from delphi.data.trajectory import DiseaseRateTrajectory, corrective_indices
from delphi.eval import eval_task
from delphi.tokenizer import Gender, Tokenizer


@dataclass
class AgeGroups:
    custom_groups: Optional[list[int]] = None
    start: int = 40
    end: int = 85
    step: int = 5


@dataclass
class CalibrateAUCArgs:
    disease_lst: str = ""
    age_groups: AgeGroups = field(default_factory=AgeGroups)
    min_time_gap: float = 0.1
    box_plot: bool = True
    seed: int = 42


def parse_diseases(diseases: str):

    assert os.path.exists(diseases)
    with open(diseases, "r") as file:
        diseases = yaml.safe_load(file)

    return diseases


def parse_age_groups(age_groups: AgeGroups) -> np.ndarray:
    if age_groups.custom_groups is None:
        return np.arange(
            age_groups.start,
            age_groups.end,
            age_groups.step,
        )
    else:
        return np.array(age_groups.custom_groups)


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
    trajectory: DiseaseRateTrajectory,
    task_args: CalibrateAUCArgs,
) -> tuple:

    rng = np.random.default_rng(task_args.seed)

    age_groups = parse_age_groups(task_args.age_groups)
    age_buckets = [(i, j) for i, j in zip(age_groups[:-1], age_groups[1:])]
    l = len(age_buckets)

    ctl_counts = np.zeros(l)
    dis_counts = np.zeros(l)
    auc_vals = np.zeros(l)
    token_rates = []

    disease_free = ~trajectory.has_token(disease_token, token_type=None)

    for i, (age_start, age_end) in enumerate(age_buckets):

        tw = (age_start * DAYS_PER_YEAR, age_end * DAYS_PER_YEAR)

        any_in_tw = trajectory.has_any_token(t0_range=tw)
        ctl_paths = trajectory[disease_free & any_in_tw]

        disease_in_tw = trajectory.has_token(
            disease_token, token_type="target", t0_range=tw
        )
        dis_paths = trajectory[disease_in_tw]

        ctl_counts[i] = ctl_paths.n_participants
        dis_counts[i] = dis_paths.n_participants

        _, ctl_token_rates = ctl_paths.disease_rate(t0_range=tw, keep="random", rng=rng)
        _, dis_token_rates = dis_paths.penultimate_disease_rate(disease_token)
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

    with open(os.path.join(ckpt, task_input, "config.yaml"), "r") as f:
        forward_cfg = yaml.safe_load(f)
    forward_cfg = from_dict(FeedForwardConfig, forward_cfg)

    task_dump_dir = os.path.join(ckpt, task_input, task_name)
    os.makedirs(task_dump_dir, exist_ok=True)
    with open(os.path.join(task_dump_dir, "config.yaml"), "w") as f:
        yaml.dump(asdict(task_args), f, default_flow_style=False, sort_keys=False)

    logits_path = os.path.join(ckpt, task_input, "logits.bin")
    assert os.path.exists(logits_path)
    "logits.bin not found in the checkpoint directory"
    xt_path = os.path.join(ckpt, task_input, "gen.bin")
    assert os.path.exists(xt_path)
    "gen.bin not found in the checkpoint directory"

    logits = np.fromfile(logits_path, dtype=np.float16).reshape(
        -1, tokenizer.vocab_size + 1
    )
    XT = np.fromfile(xt_path, dtype=np.uint32).reshape(-1, 3)

    X, T = tricolumnar_to_2d(XT)
    X_t0, X_t1 = X[:, :-1], X[:, 1:]
    T_t0, T_t1 = T[:, :-1], T[:, 1:]

    diseases = parse_diseases(task_args.disease_lst)

    for disease in diseases:

        dis_token = tokenizer[disease]

        Y = np.zeros_like(X, dtype=np.float16)
        sub_idx, pos_idx = np.nonzero(X)
        Y[sub_idx, pos_idx] = logits[:, dis_token]
        # Y = np.exp(Y) * DAYS_PER_YEAR
        # Y = 1 - np.exp(-Y)
        C = corrective_indices(
            T0=T_t0,
            T1=T_t1,
            offset=task_args.min_time_gap,
        )

        Y_t1 = Y[:, :-1]
        traj = DiseaseRateTrajectory(
            X_t0=X_t0,
            T_t0=np.take_along_axis(T_t0, C, axis=1),
            X_t1=X_t1,
            T_t1=T_t1,
            Y_t1=Y_t1,
        )

        is_female = traj.has_token(tokenizer[Gender.FEMALE.value], token_type=None)
        is_male = traj.has_token(tokenizer[Gender.MALE.value], token_type=None)
        is_gender_dict = {
            "female": is_female,
            "male": is_male,
            "either": is_female | is_male,
        }

        dis_dump_dir = os.path.join(task_dump_dir, disease)
        os.makedirs(dis_dump_dir, exist_ok=True)

        for gender, is_gender in is_gender_dict.items():

            age_buckets, auc_vals, ctl_counts, dis_counts = auc_by_age_group(
                disease_token=dis_token,
                trajectory=traj[is_gender],
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
            total_row = {
                "age_group": "total",
                "auc": np.nanmean(auc_vals),
                "ctl_counts": int(np.sum(ctl_counts)),
                "dis_counts": int(np.sum(dis_counts)),
            }
            df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
            df.to_csv(
                os.path.join(dis_dump_dir, f"{gender}_{task_args.min_time_gap}.csv"),
                index=False,
                float_format="%.3f",
            )
            # fig, ax = box_plot_disease_rates(
            #     age_buckets=age_buckets,
            #     disease_rates=token_rates,
            #     auc_vals=list(auc_vals)
            # )

            # fig.savefig(os.path.join(task_dump_dir, f"{disease}_{gender}_{time_offset}.png"))
