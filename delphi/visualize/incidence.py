from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from delphi import DAYS_PER_YEAR
from delphi.data.cohort import Cohort, cohort_from_ukb_data
from delphi.data.trajectory import DelphiTrajectories
from delphi.eval import eval_task
from delphi.model.transformer import Delphi
from delphi.tokenizer import Gender, Tokenizer
from delphi.utils import get_batch, get_p2i

OptionalPath = Optional[str]


@dataclass
class IncidencePlotConfig:
    diseases: list[str] = field(default_factory=list)
    train_data_memmap: str = "data/ukb_simulated_data/train.bin"
    val_data_memmap: str = "data/ukb_simulated_data/val.bin"
    sample_size: int = 1000
    device: str = "cpu"


def plot_age_gender_incidence(
    train_cohort: Cohort,
    val_cohort: Cohort,
    val_trajectories: DelphiTrajectories,
    disease: str,
    tokenizer: Tokenizer,
    savefig: OptionalPath,
) -> None:

    female = tokenizer[Gender.FEMALE.value]
    male = tokenizer[Gender.MALE.value]

    is_female = val_cohort.has_token(female)
    is_male = val_cohort.has_token(male)
    has_gender = is_male | is_female
    has_dis = val_cohort.has_token(tokenizer[disease])

    ctl_paths = val_trajectories[has_gender & ~has_dis]
    dis_paths = val_trajectories[has_gender & has_dis]

    ctl_cohort = val_cohort[has_gender & ~has_dis]
    dis_cohort = val_cohort[has_gender & has_dis]

    _, ax = plt.subplots()

    t = ctl_paths.T0
    y = ctl_paths.Y_t1[..., tokenizer[disease]]
    ax.scatter(
        t.ravel() / DAYS_PER_YEAR,
        y.ravel(),
        marker=".",
        c=np.repeat(
            np.array(["#DEC7FF", "#BAEBE3"])[ctl_cohort.has_token(male).astype(int)],
            ctl_paths.max_seq_len,
        ).ravel(),
        edgecolors="white",
        s=50,
        label="all other tokens",
    )

    t, y = dis_paths.penultimate_token_rates(tokenizer[disease])
    t /= DAYS_PER_YEAR
    ax.scatter(
        t.ravel(),
        y.ravel(),
        marker=".",
        c=np.array(["#7A00BF", "#00574A"])[dis_cohort.has_token(male).astype(int)],
        edgecolors="white",
        s=50,
        label="last token prior to disease",
    )

    individual_path = dis_paths[[0]]
    t = individual_path.T0
    y = individual_path.Y_t1[..., tokenizer[disease]]
    # t, y = individual_path.token_rates(tokenizer[disease])
    t /= DAYS_PER_YEAR
    ax.plot(
        t.ravel(),
        y.ravel(),
        ds="steps-post",
        c="k",
        ls="-",
        marker=".",
        markersize=8,
        markeredgecolor="white",
        markerfacecolor="k",
        label="selected case",
    )
    ax.scatter(
        t.ravel()[-1],
        y.ravel()[-1],
        marker=".",
        s=200,
        edgecolors="white",
        c="k",
        zorder=3,
    )

    female_train_cohort = train_cohort[train_cohort.has_token(female)]
    male_train_cohort = train_cohort[train_cohort.has_token(male)]

    time_bins = np.arange(100) * DAYS_PER_YEAR
    survival_hist, time_bin_edges = female_train_cohort.token_incidence(
        token=tokenizer[disease], time_bins=time_bins
    )
    ax.stairs(
        survival_hist,
        time_bin_edges / DAYS_PER_YEAR,
        color="#8520F1",
        lw=2,
        label="XX incidence",
    )
    survival_hist, time_bin_edges = male_train_cohort.token_incidence(
        token=tokenizer[disease], time_bins=time_bins
    )
    ax.stairs(
        survival_hist,
        time_bin_edges / DAYS_PER_YEAR,
        color="#0FB8A1",
        lw=2,
        label="XY incidence",
    )
    ax.set_ylim((1e-5, 1))
    ax.set_xlim(left=0)
    ax.set_yscale("log")

    ax.set_yscale("log")
    ax.set_ylabel("Rate per year")
    ax.legend(loc="upper left")
    ax.set_xlabel("Age")

    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    plt.close()


@eval_task.register
def run_incidence_plot(
    task_args: IncidencePlotConfig, model: Delphi, tokenizer: Tokenizer, dump_dir: str
) -> None:

    val = np.fromfile(task_args.val_data_memmap, dtype=np.uint32).reshape(-1, 3)
    val_p2i = get_p2i(val)
    train = np.fromfile(task_args.train_data_memmap, dtype=np.uint32).reshape(-1, 3)
    train_cohort = cohort_from_ukb_data(data=train)
    val_cohort = cohort_from_ukb_data(data=val)
    sample_cohort = val_cohort[range(task_args.sample_size)]

    d = get_batch(
        range(task_args.sample_size),
        val,
        val_p2i,
        select="left",
        block_size=64,
        device=task_args.device,
        padding="random",
    )

    p = []
    model.to(task_args.device)
    batch_size = 512
    with torch.no_grad():
        for d_batch in tqdm(
            zip(*map(lambda x: torch.split(x.to(task_args.device), batch_size), d)),
            total=d[0].shape[0] // batch_size + 1,
        ):
            p.append(model(*d_batch)[0].cpu().detach())
    p = torch.vstack(p)

    d = [d_.cpu() for d_ in d]

    y = np.exp(p.detach().numpy()) * DAYS_PER_YEAR
    y = 1 - np.exp(-y)  # y/(1+y)
    sample_paths = DelphiTrajectories(
        X_t0=d[0].detach().numpy(),
        T_t0=d[1].detach().numpy(),
        X_t1=d[2].detach().numpy(),
        T_t1=d[3].detach().numpy(),
        Y_t1=y,
    )

    for disease in task_args.diseases:
        plot_age_gender_incidence(
            train_cohort=train_cohort,
            val_cohort=sample_cohort,
            val_trajectories=sample_paths,
            disease=disease,
            tokenizer=tokenizer,
            savefig=f"{dump_dir}/incidence_plot_{disease}.png",
        )
