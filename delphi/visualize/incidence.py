import os
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml
from dacite import from_dict

from apps.generate import GenConfig
from delphi import DAYS_PER_YEAR
from delphi.data.cohort import Cohort, build_ukb_cohort
from delphi.data.dataset import UKBDataConfig, tricolumnar_to_2d
from delphi.data.trajectory import DiseaseRateTrajectory
from delphi.eval import eval_task
from delphi.model.transformer import Delphi
from delphi.tokenizer import Gender, Tokenizer

OptionalPath = Optional[str]


@dataclass
class IncidencePlotConfig:
    diseases: list[str] = field(default_factory=list)
    train_data: UKBDataConfig = field(default_factory=UKBDataConfig)


def plot_age_gender_incidence(
    train_cohort: Cohort,
    val_cohort: Cohort,
    val_trajectories: DiseaseRateTrajectory,
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

    t, y = dis_paths.penultimate_disease_rate(tokenizer[disease])
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
    task_args: IncidencePlotConfig,
    task_name: str,
    task_input: str,
    ckpt: str,
    model: Delphi,
    tokenizer: Tokenizer,
) -> None:

    with open(os.path.join(ckpt, task_input, "config.yaml"), "r") as file:
        gen_cfg = yaml.safe_load(file)
    gen_cfg = from_dict(GenConfig, gen_cfg)

    task_dump_dir = os.path.join(ckpt, gen_cfg.name, task_name)
    os.makedirs(task_dump_dir, exist_ok=True)

    train_cohort = build_ukb_cohort(cfg=task_args.train_data)
    val_cohort = build_ukb_cohort(cfg=gen_cfg.data)
    val_cohort = val_cohort[np.arange(0, gen_cfg.subsample)]

    gen_logits_path = os.path.join(ckpt, gen_cfg.name, "logits.bin")
    assert os.path.exists(gen_logits_path)
    "logits.bin not found in the checkpoint directory"
    gen_path = os.path.join(ckpt, gen_cfg.name, "gen.bin")
    assert os.path.exists(gen_path)
    "gen.bin not found in the checkpoint directory"

    logits = np.fromfile(gen_logits_path, dtype=np.float16).reshape(
        -1, tokenizer.vocab_size + 1
    )
    XT = np.fromfile(gen_path, dtype=np.uint32).reshape(-1, 3)

    X, T = tricolumnar_to_2d(XT)
    X_t0, X_t1 = X[:, :-1], X[:, 1:]
    T_t0, T_t1 = T[:, :-1], T[:, 1:]

    for disease in task_args.diseases:

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

        plot_age_gender_incidence(
            train_cohort=train_cohort,
            val_cohort=val_cohort,
            val_trajectories=traj,
            disease=disease,
            tokenizer=tokenizer,
            savefig=f"{task_dump_dir}/incidence_plot_{disease}.png",
        )
