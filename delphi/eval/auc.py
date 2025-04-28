import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm

from delphi import DAYS_PER_YEAR
from delphi.data.cohort import cohort_from_ukb_data
from delphi.data.dataset import get_p2i
from delphi.data.trajectory import DelphiTrajectories
from delphi.eval import eval_task
from delphi.model.transformer import Delphi
from delphi.tokenizer import Gender, Tokenizer
from delphi.utils import get_batch


@dataclass
class AgeGroups:
    groups: Optional[list[int]] = None
    start: int = 40
    end: int = 85
    step: int = 5


@dataclass
class AUCArgs:
    diseases: list[str] = field(default_factory=list)
    data_memmap: str = "data/ukb_simulated_data/val.bin"
    sample_size: int = 1000
    device: str = "cpu"
    age_groups: AgeGroups = field(default_factory=AgeGroups)
    min_time_gap: float = 0.1
    box_plot: bool = True


def auc(x1: np.ndarray, x2: np.ndarray) -> float:

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
    val_trajectories: DelphiTrajectories,
    task_args: AUCArgs,
) -> tuple:

    if task_args.age_groups.groups is None:
        age_groups = np.arange(
            task_args.age_groups.start,
            task_args.age_groups.end,
            task_args.age_groups.step,
        )
    else:
        age_groups = task_args.age_groups.groups

    age_buckets = [(i, j) for i, j in zip(age_groups[:-1], age_groups[1:])]
    l = len(age_buckets)

    ctl_counts = np.zeros(l)
    dis_counts = np.zeros(l)
    auc_vals = np.zeros(l)
    token_rates = []

    for i, (age_start, age_end) in enumerate(age_buckets):

        tw = (age_start * DAYS_PER_YEAR, age_end * DAYS_PER_YEAR)
        has_valid_pred_in_tw = val_trajectories.has_any_valid_predictions(
            min_time_gap=task_args.min_time_gap,
            t0_range=tw,
        )
        has_dis_in_tw = val_trajectories.has_token(
            disease_token,
            token_type="target",
            t0_range=tw,
        )
        disease_free = ~val_trajectories.has_token(
            disease_token,
            token_type="target",
        )

        ctl_paths = val_trajectories[disease_free & has_valid_pred_in_tw]
        dis_paths = val_trajectories[has_dis_in_tw]
        ctl_counts[i] = ctl_paths.n_participants
        dis_counts[i] = dis_paths.n_participants

        _, ctl_token_rates = ctl_paths.token_rates(
            disease_token, t0_range=tw, keep="average"
        )
        _, dis_token_rates = dis_paths.token_rates(
            disease_token, t0_range=tw, keep="average"
        )
        auc_vals[i] = auc(ctl_token_rates.ravel(), dis_token_rates.ravel())
        token_rates.append((ctl_token_rates, dis_token_rates))

    fig, ax = box_plot_disease_rates(
        age_buckets=age_buckets, disease_rates=token_rates, auc_vals=list(auc_vals)
    )

    return auc_vals, ctl_counts, dis_counts, fig, ax


@eval_task.register
def run_auc_eval(
    task_args: AUCArgs, model: Delphi, tokenizer: Tokenizer, dump_dir: str
) -> None:

    val = np.fromfile(task_args.data_memmap, dtype=np.uint32).reshape(-1, 3)
    val_p2i = get_p2i(val)

    d100k = get_batch(
        range(task_args.sample_size),
        val,
        val_p2i,
        select="left",
        block_size=64,
        device=task_args.device,
        padding="random",
    )

    p100k = []
    model.to(task_args.device)
    batch_size = 512
    with torch.no_grad():
        for dd in tqdm(
            zip(*map(lambda x: torch.split(x, batch_size), d100k)),
            total=d100k[0].shape[0] // batch_size + 1,
        ):
            p100k.append(
                model(*[x.to(task_args.device) for x in dd])[0].cpu().detach().numpy()
            )
    p100k = np.vstack(p100k)

    y = np.exp(p100k) * DAYS_PER_YEAR
    y = 1 - np.exp(-y)  # y/(1+y)

    val_trajectories = DelphiTrajectories(
        X_t0=d100k[0].detach().numpy(),
        T_t0=d100k[1].detach().numpy(),
        X_t1=d100k[2].detach().numpy(),
        T_t1=d100k[3].detach().numpy(),
        Y_t1=y,
    )

    val_cohort = cohort_from_ukb_data(data=val)
    val_cohort = val_cohort[range(task_args.sample_size)]

    is_female = val_cohort.has_token(tokenizer[Gender.FEMALE.value])
    is_male = val_cohort.has_token(tokenizer[Gender.MALE.value])
    is_gender_dict = {
        "female": is_female,
        "male": is_male,
        "either": is_female | is_male,
    }

    for gender, is_gender in is_gender_dict.items():

        for disease in task_args.diseases:

            disease_token = tokenizer[disease]

            auc_vals, ctl_counts, dis_counts, fig, ax = auc_by_age_group(
                disease_token=disease_token,
                val_trajectories=val_trajectories[is_gender],
                task_args=task_args,
            )

            pd.DataFrame(
                {"auc": auc_vals, "ctl_counts": ctl_counts, "dis_counts": dis_counts}
            ).to_csv(os.path.join(dump_dir, f"{disease}_{gender}.csv"), index=False)

            fig.savefig(os.path.join(dump_dir, f"{disease}_{gender}.png"))
