import os
from dataclasses import asdict, dataclass, field

import numpy as np
import yaml

from delphi import DAYS_PER_YEAR
from delphi.data.cohort import Cohort
from delphi.data.dataset import get_p2i, tricolumnar_to_2d
from delphi.data.trajectory import DiseaseRateTrajectory, corrective_indices
from delphi.eval import eval_task
from delphi.eval.auc import (
    TimeBins,
    mann_whitney_auc,
    parse_time_bins,
)
from delphi.eval.cumul_risk_auc import (
    estimate_cohort_baseline_rate,
    estimate_model_baseline_rate,
    plot_baseline_rate,
)
from delphi.eval.utils import write_auc_results
from delphi.tokenizer import Gender, Tokenizer


def normal_pdf(x, mu, sigma, eps=1e-6):

    sigma = np.maximum(sigma, eps)
    coef = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coef * np.exp(exponent)


def nearest_baseline_risks(
    timestamps: np.ndarray, baseline_rates: np.ndarray, time_bins: np.ndarray
):

    assert (
        time_bins.shape == baseline_rates.shape
    ), "time_bins and baseline_rates must have the same shape"
    delta_t = np.abs(timestamps[:, np.newaxis] - time_bins[np.newaxis, :])

    return baseline_rates[np.argmin(delta_t, axis=1)]


@dataclass
class NormRiskAUCArgs:
    disease_lst: str = ""
    age_groups: TimeBins = field(default_factory=TimeBins)
    min_time_gap: float = 0.1
    baseline_estimate: str = "cohort"
    seed: int = 42
    sample_control: str = "random"


@eval_task.register
def norm_risk_auc(
    task_args: NormRiskAUCArgs,
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

    time_bins = parse_time_bins(task_args.age_groups)
    time_bins = time_bins * DAYS_PER_YEAR
    with open(task_args.disease_lst, "r") as f:
        diseases = yaml.safe_load(f)

    for disease in diseases:

        dis_token = tokenizer[disease]

        Y = np.zeros_like(X, dtype=np.float16)
        sub_idx, pos_idx = np.nonzero(X)
        Y[sub_idx, pos_idx] = logits[:, dis_token]
        Y = np.exp(Y) * DAYS_PER_YEAR
        Y = 1 - np.exp(-Y)
        Y_t1 = Y[:, :-1]
        C = corrective_indices(
            T0=T_t0,
            T1=T_t1,
            offset=task_args.min_time_gap,
        )
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

            csv_path = os.path.join(dis_dump_dir, f"{gender}.csv")

            tr = traj[is_gender]
            ctl = ~tr.has_token(dis_token, token_type=None)
            is_valid = tr.has_any_token(t0_range=(0, None))
            dis = tr.has_token(dis_token, token_type=None)
            ctl_tr, dis_tr = tr[ctl & is_valid], tr[dis]
            n_ctl, n_dis = ctl_tr.n_participants, dis_tr.n_participants
            if n_ctl == 0 or n_dis < 2:
                write_auc_results(
                    auc_val=np.nan, n_ctl=n_ctl, n_dis=n_dis, csv_path=csv_path
                )
                continue

            if task_args.baseline_estimate == "cohort":
                co = cohort[is_gender]
                bl_rate = estimate_cohort_baseline_rate(
                    cohort=co,
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

            dis_t0, dis_predict_risk = dis_tr.penultimate_disease_rate(token=dis_token)

            rng = np.random.default_rng(task_args.seed)
            n_ctl = ctl_tr.T0.shape[0]
            if task_args.sample_control == "normal":
                mu, sigma = np.mean(dis_t0), np.std(dis_t0)
                sample_mask = ctl_tr.T0 != -1e4
                keep_idx = np.zeros((n_ctl,), dtype=int)
                for i in range(n_ctl):
                    candidate_idx = np.nonzero(sample_mask[i])[0]
                    p = normal_pdf(
                        ctl_tr.T0[i, candidate_idx],
                        mu=mu,
                        sigma=sigma,
                    )
                    p = p / np.sum(p) if np.sum(p) > 0 else np.ones_like(p) / len(p)
                    keep_idx[i] = rng.choice(candidate_idx, p=p)
                sample_mask = np.zeros_like(sample_mask)
                sample_mask[np.arange(n_ctl), keep_idx] = 1
                ctl_t0, ctl_predict_risk = (
                    ctl_tr.T0[sample_mask],
                    ctl_tr.Y_t1[sample_mask],
                )
            elif task_args.sample_control == "random":
                ctl_t0, ctl_predict_risk = ctl_tr.disease_rate(
                    t0_range=(0, None), keep=task_args.sample_control, rng=rng
                )
            else:
                raise ValueError(f"unknown sampling scheme: {task_args.sample_control}")

            dis_bl_risk = nearest_baseline_risks(
                timestamps=dis_t0,
                baseline_rates=bl_rate,
                time_bins=time_bins[:-1],
            )
            ctl_bl_risk = nearest_baseline_risks(
                timestamps=ctl_t0,
                baseline_rates=bl_rate,
                time_bins=time_bins[:-1],
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
