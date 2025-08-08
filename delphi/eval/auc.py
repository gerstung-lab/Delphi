import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import yaml
from scipy.stats import rankdata
from tqdm import tqdm

from delphi import DAYS_PER_YEAR
from delphi.data import core
from delphi.data.transform import sort_by_time
from delphi.eval import eval_task
from delphi.experiment import load_ckpt
from delphi.tokenizer import Gender


@dataclass
class TimeBins:
    # in years
    custom_bin_edges: Optional[list[int]] = None
    bin_start: int = 40
    bin_end: int = 85
    bin_width: int = 5


@dataclass
class CalibrateAUCArgs:
    data: dict = field(default_factory=dict)
    disease_lst: str = ""
    age_groups: TimeBins = field(default_factory=TimeBins)
    min_time_gap: float = 0.1
    event_input_only: bool = True
    subsample: Optional[int] = None
    device: str = "cpu"
    batch_size: int = 128


def parse_time_bins(time_bins: TimeBins) -> list[tuple[int, int]]:
    if time_bins.custom_bin_edges is None:
        age_group_edges = np.arange(
            time_bins.bin_start,
            time_bins.bin_end + time_bins.bin_width,
            time_bins.bin_width,
        )
    else:
        age_group_edges = np.array(time_bins.custom_bin_edges)
    age_grps = [(i, j) for i, j in zip(age_group_edges[:-1], age_group_edges[1:])]
    return age_grps


def mann_whitney_auc(x1: np.ndarray, x2: np.ndarray) -> float:

    n1 = len(x1)
    n2 = len(x2)
    x12 = np.concatenate([x1, x2])
    ranks = rankdata(x12, method="average")

    R1 = ranks[:n1].sum()
    U1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    if n1 == 0 or n2 == 0:
        return np.nan
    return U1 / n1 / n2


def rates_by_age_bin(
    input_time: np.ndarray,
    predicted_rates: np.ndarray,
    disease_free: np.ndarray,
    targets: np.ndarray,
    sub_idx: np.ndarray,
    dis_token: int,
    age_groups: list[tuple[int, int]],
):

    ctl_subjects = []
    dis_subjects = []
    ctl_rates = []
    dis_rates = []

    have_disease = targets == dis_token

    for age_start, age_end in age_groups:

        in_time_range = input_time >= age_start * DAYS_PER_YEAR
        in_time_range &= input_time < age_end * DAYS_PER_YEAR

        is_ctl = disease_free & in_time_range
        is_dis = have_disease & in_time_range

        ctl_subjects.append(sub_idx[is_ctl])
        dis_subjects.append(sub_idx[is_dis])

        ctl_rates.append(predicted_rates[is_ctl])
        dis_rates.append(predicted_rates[is_dis])

    return ctl_subjects, dis_subjects, ctl_rates, dis_rates


def sample_one_per_participant(subjects, rates):

    assert len(subjects) == len(rates), "subjects and rates must have the same length"

    perm = np.random.permutation(len(subjects))
    subjects = subjects[perm]
    rates = rates[perm]
    _, uniq_idx = np.unique(subjects, return_index=True)

    return rates[uniq_idx]


def corrective_indices(T0: np.ndarray, T1: np.ndarray, offset: float):

    m, _ = T0.shape
    _, p = T1.shape

    C = np.zeros((m, p), dtype=int)

    for i in range(m):
        t0_row = T0[i]
        t1_row = T1[i]

        c_idx = (
            np.broadcast_to(t0_row, (t1_row.size, t0_row.size))
            <= (t1_row - offset).reshape(-1, 1)
        ).sum(axis=1) - 1

        C[i] = c_idx

    return C


def move_batch_to_device(args, device: str):

    return tuple([arg.to(device) for arg in args])


@eval_task.register
def calibrate_auc(
    task_args: CalibrateAUCArgs,
    task_name: str,
    ckpt: str,
) -> None:

    device = task_args.device
    model, _, tokenizer = load_ckpt(ckpt)
    model.to(device)
    model.eval()

    ds = core.build_dataset(task_args.data)
    n_participants = len(ds) if task_args.subsample is None else task_args.subsample
    it = core.eval_iter(total_size=n_participants, batch_size=128)
    data_loader = core.load_sequences(it=it, dataset=ds)
    data_loader = tqdm(
        data_loader, total=math.ceil(n_participants / task_args.batch_size), leave=True
    )

    idx_lst = list()
    age_lst = list()
    logits_lst = list()
    with torch.no_grad():
        for batch_input in data_loader:
            batch_input = move_batch_to_device(batch_input, device=device)

            batch_logits, batch_X, batch_T = model.eval_step(
                *batch_input, horizon=task_args.age_groups.bin_width * DAYS_PER_YEAR
            )

            batch_X = batch_X.detach().cpu().numpy()
            batch_T = batch_T.detach().cpu().numpy()
            idx_lst.extend([x[t != -1e4] for x, t in zip(batch_X, batch_T)])
            age_lst.extend([t[t != -1e4] for t, t in zip(batch_X, batch_T)])
            batch_sub_idx, batch_pos_idx = np.nonzero(batch_T != -1e4)
            logits_lst.append(
                batch_logits[batch_sub_idx, batch_pos_idx].detach().cpu().numpy()
            )

    X = core.collate_batch_data(idx_lst)
    T = core.collate_batch_time(age_lst)
    T, X = sort_by_time(T, X)
    logits = np.concatenate(logits_lst)

    sub_idx, pos_idx = np.nonzero(T != -1e4)

    max_len = T.shape[1] - 1
    X_t0, X_t1 = X[:, :-1], X[:, 1:]
    T_t0, T_t1 = T[:, :-1], T[:, 1:]
    C = corrective_indices(
        T0=T_t0,
        T1=T_t1,
        offset=task_args.min_time_gap * 365.25,
    )

    remove_last = pos_idx < max_len
    sub_idx, pos_idx = sub_idx[remove_last], pos_idx[remove_last]
    offset_pos_idx = C[sub_idx, pos_idx]
    logits_idx = np.arange(logits.shape[0])[remove_last]

    has_input = offset_pos_idx >= 0
    sub_idx, pos_idx = sub_idx[has_input], pos_idx[has_input]
    offset_pos_idx = offset_pos_idx[has_input]
    logits_idx = logits_idx[has_input]

    logits_idx = logits_idx[np.arange(logits_idx.shape[0]) + offset_pos_idx - pos_idx]
    t_t0 = T_t0[sub_idx, offset_pos_idx]
    targets = X_t1[sub_idx, pos_idx]

    is_female = (X_t0 == tokenizer[Gender.FEMALE.value]).any(axis=1)[sub_idx]
    is_male = (X_t0 == tokenizer[Gender.MALE.value]).any(axis=1)[sub_idx]
    is_gender_dict = {
        "female": is_female,
        "male": is_male,
        "either": is_female | is_male,
    }

    age_groups = parse_time_bins(task_args.age_groups)
    age_group_keys = [f"{start}-{end}" for start, end in age_groups]

    with open(task_args.disease_lst, "r") as f:
        disease_lst = yaml.safe_load(f)

    logbook = {}
    for disease in tqdm(disease_lst):
        logbook[disease] = {}
        dis_token = tokenizer[disease]

        disease_free = (~(X_t1 == dis_token).any(axis=1))[sub_idx]
        y_t1 = logits[logits_idx, dis_token]

        for gender, is_gender in is_gender_dict.items():

            ctl_subjects, dis_subjects, ctl_rates, dis_rates = rates_by_age_bin(
                input_time=t_t0[is_gender],
                sub_idx=sub_idx[is_gender],
                predicted_rates=y_t1[is_gender],
                disease_free=disease_free[is_gender],
                targets=targets[is_gender],
                dis_token=dis_token,
                age_groups=age_groups,
            )

            ctl_rates = [
                sample_one_per_participant(subj, rate)
                for subj, rate in zip(ctl_subjects, ctl_rates)
            ]
            auc = [
                mann_whitney_auc(ctl_rate, dis_rate)
                for ctl_rate, dis_rate in zip(ctl_rates, dis_rates)
            ]

            n_ctl = [len(np.unique(s)) for s in ctl_subjects]
            n_dis = [len(s) for s in dis_subjects]
            logbook[disease][gender] = {
                age_group_keys[i]: {
                    "auc": round(float(auc[i]), 2) if not np.isnan(auc[i]) else None,
                    "ctl_count": int(n_ctl[i]),
                    "dis_count": int(n_dis[i]),
                }
                for i in range(len(age_groups))
            }
            mean_auc = float(np.nanmean(auc))
            all_ctl = np.concatenate(ctl_subjects)
            all_dis = np.concatenate(dis_subjects)
            logbook[disease][gender]["total"] = {
                "auc": round(mean_auc, 2) if not np.isnan(mean_auc) else None,
                "ctl_count": len(np.unique(all_ctl)),
                "dis_count": len(all_dis),
            }

    logbook_path = os.path.join(ckpt, f"{task_name}.json")
    with open(logbook_path, "w") as f:
        json.dump(logbook, f, indent=4)
