import os
from dataclasses import asdict, dataclass, field

import numpy as np
import yaml

from delphi import DAYS_PER_YEAR
from delphi.data.cohort import Cohort
from delphi.data.dataset import get_p2i, tricolumnar_to_2d
from delphi.data.trajectory import DiseaseRateTrajectory
from delphi.eval import eval_task
from delphi.eval.auc import (
    AgeGroups,
    mann_whitney_auc,
    parse_age_groups,
    parse_diseases,
)
from delphi.eval.utils import write_auc_results
from delphi.tokenizer import CoreEvents, Gender, Tokenizer


def estimate_cohort_baseline_rate(
    cohort: Cohort,
    disease_token: int,
    time_bins: np.ndarray,
) -> np.ndarray:

    token_incidence, _ = cohort.token_incidence(
        token=disease_token, time_bins=time_bins
    )
    delta_t = time_bins[1:] - time_bins[:-1]
    bl_rate = token_incidence / delta_t

    return bl_rate


def estimate_model_baseline_rate(
    trajectory: DiseaseRateTrajectory,
    time_bins: np.ndarray,
) -> np.ndarray:

    bl_rate = np.zeros(time_bins.shape[0] - 1, dtype=np.float32)
    for i in range(time_bins.shape[0] - 1):
        t0_range = (time_bins[i], time_bins[i + 1])
        has_token_in_tw = trajectory.has_any_token(t0_range=t0_range)
        if has_token_in_tw.sum() > 0:
            _, rates_in_tw = trajectory[has_token_in_tw].disease_rate(
                t0_range=t0_range, keep="average"
            )
            bl_rate[i] = rates_in_tw.mean()
        else:
            bl_rate[i] = bl_rate[i - 1] if i > 0 else 0.0

    return bl_rate


def plot_baseline_rate(
    baseline_rates: np.ndarray, time_bins: np.ndarray, save_path: str
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 5))
    plt.plot(time_bins[:-1] / DAYS_PER_YEAR, baseline_rates, marker="o")
    plt.xlabel("time (year)")
    plt.ylabel("baseline Rate")
    plt.yscale("log", base=10)
    plt.grid()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def cumulative_baseline_risks(
    baseline_rates: np.ndarray, time_bins: np.ndarray, until: np.ndarray
):
    assert (
        baseline_rates.ndim == time_bins.ndim == 1
    ), "baseline_rates and time_bins must be 1D arrays"
    l = baseline_rates.shape[0]
    assert (
        np.unique(time_bins).shape[0] == l + 1
    ), "time_bins must be unique and have one more element than baseline_rates"

    n = until.shape[0]
    delta_t = time_bins[1:] - time_bins[:-1]
    delta_t = np.broadcast_to(delta_t, (n, l))
    time_bin_t0 = np.broadcast_to(time_bins[:-1], (n, l))

    until = until[:, np.newaxis]
    until = np.broadcast_to(until, (n, l))
    delta_t = np.clip((until - time_bin_t0), 0, delta_t)

    cumul_risk = np.sum(baseline_rates * delta_t, axis=1)

    return cumul_risk


@dataclass
class CumulRiskAUCArgs:
    disease_lst: str = ""
    age_groups: AgeGroups = field(default_factory=AgeGroups)
    baseline_estimate: str = "cohort"


@eval_task.register
def cumul_risk_auc(
    task_args: CumulRiskAUCArgs,
    task_name: str,
    task_input: str,
    ckpt: str,
    tokenizer: Tokenizer,
    **kwargs,
) -> None:

    task_dump_dir = os.path.join(ckpt, task_input, task_name)
    os.makedirs(task_dump_dir, exist_ok=True)
    with open(os.path.join(task_dump_dir, "config.yaml"), "w") as f:
        yaml.dump(asdict(task_args), f, default_flow_style=False, sort_keys=False)

    logits_path = os.path.join(ckpt, task_input, "logits.bin")
    assert os.path.exists(
        logits_path
    ), "logits.bin not found in the checkpoint directory"
    xt_path = os.path.join(ckpt, task_input, "gen.bin")
    assert os.path.exists(xt_path), "gen.bin not found in the checkpoint directory"

    logits = np.fromfile(logits_path, dtype=np.float16).reshape(
        -1, tokenizer.vocab_size + 1
    )
    XT = np.fromfile(xt_path, dtype=np.uint32).reshape(-1, 3)
    X, T = tricolumnar_to_2d(XT)
    X_t0, X_t1 = X[:, :-1], X[:, 1:]
    T_t0, T_t1 = T[:, :-1], T[:, 1:]

    p2i = get_p2i(XT)
    cohort = Cohort(
        participants=XT[:, 0][p2i[:, 0]],
        tokens=X,
        time_steps=T,
    )
    time_bins = parse_age_groups(task_args.age_groups)
    time_bins = time_bins * DAYS_PER_YEAR
    diseases = parse_diseases(task_args.disease_lst)

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

            csv_path = os.path.join(dis_dump_dir, f"{gender}.csv")

            tr = traj[is_gender]
            ctl = ~tr.has_token(dis_token)
            has_passed = tr.has_token(tokenizer[CoreEvents.DEATH.value])
            dis = tr.has_token(dis_token)
            ctl_tr, dis_tr = tr[ctl & has_passed], tr[dis]
            n_ctl, n_dis = ctl_tr.n_participants, dis_tr.n_participants
            if n_ctl == 0 or n_dis == 0:
                write_auc_results(
                    auc_val=np.nan, n_ctl=n_ctl, n_dis=n_dis, csv_path=csv_path
                )
                continue

            ctl_predict_risk = ctl_tr.cumulative_disease_risk()
            dis_t0, _ = dis_tr.token_timestamps(token=dis_token)
            dis_predict_risk = dis_tr.cumulative_disease_risk(until=dis_t0)

            if task_args.baseline_estimate == "cohort":
                bl_rate = estimate_cohort_baseline_rate(
                    cohort=cohort[is_gender],
                    disease_token=dis_token,
                    time_bins=time_bins,
                )
            elif task_args.baseline_estimate == "model":
                bl_rate = estimate_model_baseline_rate(
                    trajectory=tr,
                    time_bins=time_bins,
                )
            else:
                raise ValueError(
                    f"unknown baseline estimate method: {task_args.baseline_estimate}"
                )
            plot_baseline_rate(
                baseline_rates=bl_rate,
                time_bins=time_bins,
                save_path=os.path.join(dis_dump_dir, f"baseline_rate_{gender}.png"),
            )

            ctl_max_t0 = ctl_tr.T0.max(axis=1)
            ctl_bl_risk = cumulative_baseline_risks(
                baseline_rates=bl_rate, time_bins=time_bins, until=ctl_max_t0
            )

            dis_t0, _ = dis_tr.token_timestamps(dis_token)
            dis_bl_risk = cumulative_baseline_risks(
                baseline_rates=bl_rate, time_bins=time_bins, until=dis_t0
            )

            ctl_relative_risk = (
                ctl_predict_risk[ctl_bl_risk > 0] / ctl_bl_risk[ctl_bl_risk > 0]
            )
            dis_relative_risk = (
                dis_predict_risk[dis_bl_risk > 0] / dis_bl_risk[dis_bl_risk > 0]
            )
            auc_val = mann_whitney_auc(ctl_relative_risk, dis_relative_risk)

            write_auc_results(
                auc_val=auc_val, n_ctl=n_ctl, n_dis=n_dis, csv_path=csv_path
            )

    pass
