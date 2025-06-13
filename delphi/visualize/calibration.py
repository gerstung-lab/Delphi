import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import yaml
from dacite import from_dict

from apps.generate import GenConfig
from delphi import DAYS_PER_YEAR
from delphi.data.dataset import tricolumnar_to_2d
from delphi.data.trajectory import DiseaseRateTrajectory, corrective_indices
from delphi.eval import eval_task
from delphi.eval.auc import TimeBins
from delphi.tokenizer import Gender, load_tokenizer_from_ckpt


@dataclass
class CalibrationArgs:
    diseases: list[str] = field(default_factory=list)
    age_groups: TimeBins = field(default_factory=TimeBins)
    min_time_gap: float = 0.1


def calibrate_by_age_group(
    disease_token: int,
    val_paths: DiseaseRateTrajectory,
    task_args: CalibrationArgs,
):

    if task_args.age_groups.custom_groups is None:
        age_groups = np.arange(
            task_args.age_groups.start,
            task_args.age_groups.end,
            task_args.age_groups.step,
        )
    else:
        age_groups = task_args.age_groups.custom_groups

    age_buckets = [(i, j) for i, j in zip(age_groups[:-1], age_groups[1:])]
    l = len(age_buckets)

    rate_bins = 10 ** np.arange(-6.0, 1.5, 0.5)

    observed_rates = []
    predicted_rates = []
    confidence_intervals = []

    for i, (age_start, age_end) in enumerate(age_buckets):

        tw = (age_start * DAYS_PER_YEAR, age_end * DAYS_PER_YEAR)

        any_in_tw = val_paths.has_any_token(t0_range=tw)
        tw_paths = val_paths[any_in_tw]
        _, rates = tw_paths.disease_rate(t0_range=tw, keep="random")

        rates = np.exp(np.array(rates)) * 365.25
        rates = 1 - np.exp(-rates * (age_end - age_start))

        binned_rates = np.zeros(len(rate_bins) - 1)
        binned_incidence = np.zeros(len(rate_bins) - 1)
        ci = np.zeros((len(rate_bins) - 1, 2))
        for j in range(0, len(rate_bins) - 1):
            bin_mask = (rates > rate_bins[j]) & (rates <= rate_bins[j + 1])
            if bin_mask.sum() == 0:
                binned_rates[j] = np.nan
                binned_incidence[j] = np.nan
                ci[j] = np.nan
                continue
            binned_rates[j] = rates[bin_mask].mean()

            binned_paths = tw_paths[bin_mask]
            n_dis = binned_paths.has_token(disease_token, t1_range=tw).sum()
            n_ctl = binned_paths.n_participants - n_dis

            binned_incidence[j] = n_dis / binned_paths.n_participants

            ci[j] = scipy.stats.beta(0.1 + n_dis, 0.1 + n_ctl).ppf([0.025, 0.975])

        observed_rates.append(binned_incidence)
        predicted_rates.append(binned_rates)
        confidence_intervals.append(ci)

    return age_buckets, observed_rates, predicted_rates, confidence_intervals


def scatter_plot_calibration(
    age_buckets: list[tuple],
    obs_rates: list[np.ndarray],
    pred_rates: list[np.ndarray],
    cis: list[np.ndarray],
):

    l = len(age_buckets)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=l,
        figsize=(20 / 8 * l, 1.5),
        sharex=True,
        sharey=False,
    )

    for i, (age_start, age_end) in enumerate(age_buckets):

        ax[i].errorbar(
            pred_rates[i],
            obs_rates[i] + 1e-5,
            yerr=[abs(obs_rates[i] - cis[i][:, 0]), abs(cis[i][:, 1] - obs_rates[i])],
            fmt=".",
            color="black",
            alpha=0.7,
            elinewidth=0.5,
        )

        ax[i].set_box_aspect(1)
        ax[i].scatter(
            pred_rates[i][~np.isnan(pred_rates[i])].mean(),
            obs_rates[i][~np.isnan(obs_rates[i])].mean(),
            c="r",
            ec="w",
        )
        ax[i].set_yscale("log")
        ax[i].set_xscale("log")
        ax[i].set_ylim((1e-5, 1))
        ax[i].set_xlim((1e-5, 1))
        if i == 0:
            ax[i].set_ylabel(f"observed rate [1/yr]")
        if i > 0:
            ax[i].yaxis.set_ticklabels([])
        ax[i].set_xlabel(f"predicted rate [1/yr]", fontsize=8)
        ax[i].plot([0, 1], [0, 1], transform=ax[i].transAxes, lw=0.5, c="k", ls="--")
        ax[i].grid()
        ax[i].set_title(f"{age_start}-{age_end}yr")

    return fig, ax


def line_plot_calibration(
    age_buckets: list[tuple],
    obs_rates: list[np.ndarray],
    pred_rates: list[np.ndarray],
    cis: list[np.ndarray],
):

    fig, ax = plt.subplots()
    for i in range(len(age_buckets)):

        age_start, age_end = age_buckets[i]
        x = pred_rates[i]
        y = obs_rates[i]
        x = x[y > 0]
        y = y[y > 0]
        ax.plot(
            x,
            y,
            label=f"{age_start}-{age_end}yrs",
        )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim((1e-5, 1))
        ax.set_xlim((1e-5, 1))
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, lw=0.5, c="k", ls="--")
        ax.set_xlabel("Model rate [1/yr]")
        ax.set_ylabel("Observed rate [1/yr]")

    return fig, ax


@eval_task.register
def calibrate(
    task_args: CalibrationArgs,
    task_name: str,
    task_input: str,
    ckpt: str,
    **kwargs,
):

    tokenizer = load_tokenizer_from_ckpt(ckpt)

    with open(os.path.join(ckpt, task_input, "config.yaml"), "r") as file:
        gen_cfg = yaml.safe_load(file)
    gen_cfg = from_dict(GenConfig, gen_cfg)

    task_dump_dir = os.path.join(ckpt, gen_cfg.name, task_name)
    os.makedirs(task_dump_dir, exist_ok=True)

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

        Y = np.zeros((X.shape[0], X.shape[1], 1))
        sub_idx, pos_idx = np.nonzero(X)
        Y[sub_idx, pos_idx, :] = logits[:, [dis_token]]
        Y_t1 = Y[:, :-1, :]

        C = corrective_indices(
            T0=T_t0,
            T1=T_t1,
            offset=task_args.min_time_gap,
        )

        trajectories = DiseaseRateTrajectory(
            X_t0=X_t0,
            T_t0=np.take_along_axis(T_t0, C, axis=1),
            X_t1=X_t1,
            T_t1=T_t1,
            Y_t1=Y_t1,
        )

        gender_masks = {
            Gender.MALE.value: trajectories.has_token(
                token=tokenizer[Gender.MALE.value],
                token_type=None,
            ),
            Gender.FEMALE.value: trajectories.has_token(
                token=tokenizer[Gender.FEMALE.value],
                token_type=None,
            ),
            "all": np.ones(trajectories.n_participants, dtype=bool),
        }

        for gender, gender_mask in gender_masks.items():

            age_buckets, obs_rates, pred_rates, cis = calibrate_by_age_group(
                disease_token=dis_token,
                val_paths=trajectories[gender_mask],
                task_args=task_args,
            )

            fig, ax = scatter_plot_calibration(age_buckets, obs_rates, pred_rates, cis)
            fig.savefig(
                os.path.join(task_dump_dir, f"scatter-{disease}-{gender}.png"),
                dpi=300,
                bbox_inches="tight",
            )

            fig, ax = line_plot_calibration(age_buckets, obs_rates, pred_rates, cis)
            fig.savefig(
                os.path.join(task_dump_dir, f"line-{disease}-{gender}.png"),
                dpi=300,
                bbox_inches="tight",
            )
