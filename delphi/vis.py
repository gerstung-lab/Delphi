from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy

from delphi import DAYS_PER_YEAR
from delphi.data import Cohort, DelphiTrajectories
from delphi.tokenizer import CoreEvents, Tokenizer

OptionalPath = Optional[str]


def plot_age_gender_incidence(
    train_cohort: Cohort,
    val_cohort: Cohort,
    val_trajectories: DelphiTrajectories,
    disease: str,
    tokenizer: Tokenizer,
    savefig: OptionalPath,
) -> None:

    female = tokenizer[CoreEvents.FEMALE.value]
    male = tokenizer[CoreEvents.MALE.value]

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


def plot_calibration(
    disease,
    d,
    p,
    tokenizer: Tokenizer,
    offset=365.25,
    age_groups=np.arange(45, 85, 5),
    calibration="bins",
    # lifestyle_ix=range(10, 12),
    binning="power",
    bins=10 ** np.arange(-6.0, 1.5, 0.5),
    savefig: Optional[str] = None,
):

    death = tokenizer[CoreEvents.DEATH.value]

    l = len(age_groups)
    age_step = age_groups[1] - age_groups[0]

    _, ax = plt.subplots(
        2, l, figsize=(20 / 8 * l, 3), sharex=True, sharey=False, height_ratios=[1, 0.5]
    )
    ax = ax[np.newaxis, :]
    axf = ax.ravel()

    ## Indexes of cases
    wk = np.where(d[2].detach().numpy() == tokenizer[disease])

    if len(wk[0]) < 2:
        return np.repeat(np.nan, l)

    wc = np.where(
        (d[2] != tokenizer[disease])
        * (~(d[2] == tokenizer[disease]).any(-1))[..., None]
    )
    # wc = np.where(d[2] != tokenizer[disease])
    wall = (
        np.concatenate([wk[0], wc[0]]),
        np.concatenate([wk[1], wc[1]]),
    )

    pred_idx = (d[1][wall[0]] <= d[3][wall].reshape(-1, 1) - offset).sum(1) - 1
    z = d[1].detach().numpy()[(wall[0], pred_idx)]
    z = z[pred_idx != -1]

    zk = d[3].detach().numpy()[wall]  # Target times, cases and controls
    zk = zk[pred_idx != -1]

    x = np.exp(p[..., tokenizer[disease]][(wall[0], pred_idx)]) * 365.25
    x = x[pred_idx != -1]
    x = 1 - np.exp(
        -x * age_step
    )  # x * 1/age_step/ (1/age_step+x) # Rate can't exceed 1/age bin

    wk = (wk[0][pred_idx[: len(wk[0])] != -1], wk[1][pred_idx[: len(wk[0])] != -1])
    p_idx = wall[0][pred_idx != -1]

    out = []

    for i, aa in enumerate(age_groups):
        a = np.logical_and(z / 365.25 >= aa, z / 365.25 < aa + age_step)
        # a *= zk - z < 365.25  # * age_step

        # uniq_p_idx = np.unique(p_idx[a], return_index=True)[0]
        # p_idx = np.isin(p_idx, uniq_p_idx) & (np.cumsum(np.isin(p_idx, uniq_p_idx)) == 1)
        # a *= p_idx
        P = p_idx.copy()
        P[~a] = -1

        a *= np.isin(
            np.arange(a.shape[0]), np.unique(P, return_index=True)[1]
        )  # Mask duplicated people in age bracket
        axf[i + l].boxplot(
            (x[len(wk[0]) :][a[len(wk[0]) :]], x[: len(wk[0])][a[: len(wk[0])]]),
            vert=False,
            sym=".",
            widths=0.5,
            whis=(5, 95),
            flierprops=dict(marker=".", markeredgecolor="white", markerfacecolor="k"),
        )
        axf[i + l].set_xscale("log")
        axf[i + l].set_xlim((1e-5, 1))
        axf[i + l].set_yticks((1, 2), ["", ""])
        if i == 0:
            axf[i].set_title(
                f"{tokenizer.name_for_plot(disease)}\n",
                fontsize=10,
                weight="bold",
                loc="left",
            )
            axf[i].set_ylabel(f"Observed rate [1/yr]")
            axf[i + l].set_yticks(
                (1, 2),
                (
                    f'{["Healthy","Alive"][tokenizer[disease]==death]}',
                    f'{["Diseased","Deceased"][tokenizer[disease]==death]}',
                ),
            )
        y = auc(x[len(wk[0]) :][a[len(wk[0]) :]], x[: len(wk[0])][a[: len(wk[0])]])

        foo = ["dis'd", "dec'd"]
        axf[i].text(
            0,
            0.9,
            s=f'{len(x[len(wk[0]):][a[len(wk[0]):]])} {["healthy","alive"][tokenizer[disease]==death]}\n{len(x[:len(wk[0])][a[:len(wk[0])]])} {foo[tokenizer[disease]==death]}',
            transform=axf[i].transAxes,
            va="top",
        )
        axf[i + l].text(
            0.5,
            0.8,
            s=f"AUC={y:.2}",
            transform=axf[i + l].transAxes,
            va="center",
            ha="center",
        )
        axf[i + l].set_xlabel(f"Predicted rate [1/yr]")
        axf[i + l].set_ylim((0.5, 3.5))
        axf[i].text(
            0.5,
            1,
            s=f"{aa}-{aa+age_step}yr",
            transform=axf[i].transAxes,
            va="bottom",
            ha="center",
            weight="bold",
        )

        xa = x[a]
        ya = np.concatenate([np.ones(len(wk[0])), np.zeros(x.shape[0] - len(wk[0]))])[
            a
        ]  # * (zk - z)[a]

        if len(xa) == 0:
            continue

        if calibration == "bins":
            if binning == "deciles":
                bins = np.quantile(xa, np.arange(0, 1.05, 0.05))
            else:
                bins = bins
            pred = np.array(
                [
                    xa[np.logical_and(xa > bins[b - 1], xa <= bins[b])].mean()
                    for b in range(1, len(bins))
                ]
            )
            obs = np.array(
                [
                    ya[np.logical_and(xa > bins[b - 1], xa <= bins[b])].mean()
                    for b in range(1, len(bins))
                ]
            )
            ci = np.array(
                [
                    scipy.stats.beta(
                        0.1 + ya[np.logical_and(xa > bins[b - 1], xa <= bins[b])].sum(),
                        0.1
                        + (
                            1 - ya[np.logical_and(xa > bins[b - 1], xa <= bins[b])]
                        ).sum(),
                    ).ppf([0.025, 0.975])
                    for b in range(1, len(bins))
                ]
            )
            axf[i].scatter(pred, obs + 1e-5, marker=".", c="k")
            for j, pr in enumerate(pred):
                if not np.isnan(obs[j]):
                    axf[i].plot(np.repeat(pr, 2), ci[j], c="k", lw=0.5, ls=":")
            wgt = np.array(
                [
                    [
                        ya[np.logical_and(xa > bins[b - 1], xa <= bins[b])].sum(),
                        np.logical_and(xa > bins[b - 1], xa <= bins[b]).sum(),
                    ]
                    for b in range(1, len(bins))
                ]
            )
            out.append([pred, obs, ci, wgt])
        else:
            o = np.argsort(xa)
            axf[i].plot(
                xa[o], ya[o] / (ya.sum() - np.cumsum(ya[o])) / age_step, ds="steps"
            )
            out.append(np.nan)

        axf[i].set_box_aspect(1)
        axf[i].scatter(xa.mean(), ya.mean(), c="r", ec="w")
        axf[i].set_yscale("log")
        axf[i].set_xscale("log")
        axf[i].set_ylim((1e-5, 1))
        axf[i].set_xlim((1e-5, 1))
        axf[i].plot([0, 1], [0, 1], transform=axf[i].transAxes, lw=0.5, c="k", ls="--")

    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    # plt.close()

    return out


def plot_calibration_test(
    disease: str,
    val_trajectories: DelphiTrajectories,
    tokenizer: Tokenizer,
    min_time_gap: float = 0.1,
    age_groups: Sequence[float] = list(np.arange(45.0, 85.0, 5.0)),
    calibration="bins",
    binning="power",
    bins=10 ** np.arange(-6.0, 1.5, 0.5),
    savefig: Optional[str] = None,
):
    death = tokenizer[CoreEvents.DEATH.value]

    age_buckets = [(i, j) for i, j in zip(age_groups[:-1], age_groups[1:])]
    l = len(age_buckets)

    _, ax = plt.subplots(
        2, l, figsize=(20 / 8 * l, 3), sharex=True, sharey=False, height_ratios=[1, 0.5]
    )
    ax = ax[np.newaxis, :]
    axf = ax.ravel()

    for i, (age_start, age_end) in enumerate(age_buckets):

        calib_ax = axf[i]
        box_ax = axf[i + l]

        tw = (age_start * DAYS_PER_YEAR, age_end * DAYS_PER_YEAR)
        has_valid_pred_in_tw = val_trajectories.has_any_valid_predictions(
            min_time_gap=min_time_gap,
            t0_range=tw,
        )
        has_dis_in_tw = val_trajectories.has_token(
            tokenizer[disease],
            token_type="target",
            t0_range=tw,
        )
        disease_free = ~val_trajectories.has_token(
            tokenizer[disease],
            token_type="target",
        )

        ctl_paths = val_trajectories[disease_free & has_valid_pred_in_tw]
        dis_paths = val_trajectories[has_dis_in_tw]
        n_ctl = ctl_paths.n_participants
        n_dis = dis_paths.n_participants

        _, ctl_token_rates = ctl_paths.token_rates(
            tokenizer[disease], t0_range=tw, keep="average"
        )
        _, dis_token_rates = dis_paths.token_rates(
            tokenizer[disease], t0_range=tw, keep="average"
        )
        auc_val = auc(ctl_token_rates.ravel(), dis_token_rates.ravel())

        box_ax.text(
            0.5,
            0.8,
            s=f"AUC={auc_val:.2}",
            transform=box_ax.transAxes,
            va="center",
            ha="center",
        )
        box_ax.boxplot(
            (ctl_token_rates, dis_token_rates),
            vert=False,
            sym=".",
            widths=0.5,
            whis=(5, 95),
            flierprops=dict(marker=".", markeredgecolor="white", markerfacecolor="k"),
        )
        box_ax.set_xscale("log")
        box_ax.set_xlim((1e-5, 1))
        box_ax.set_yticks((1, 2), ["", ""])
        box_ax.set_xlabel(f"Predicted rate [1/yr]")
        box_ax.set_ylim((0.5, 3.5))

        calib_ax.text(
            0.5,
            1,
            s=f"{age_start}-{age_end}yr",
            transform=calib_ax.transAxes,
            va="bottom",
            ha="center",
            weight="bold",
        )
        ctl_text = "healthy" if tokenizer[disease] != death else "alive"
        dis_text = "dis'd" if tokenizer[disease] != death else "dec'd"
        calib_ax.text(
            0,
            0.9,
            s=f"{n_ctl} {ctl_text}\n{n_dis} {dis_text}",
            transform=calib_ax.transAxes,
            va="top",
        )

    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")

    pass
