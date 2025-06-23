import os
import time

import numpy as np
import pytest

from archive.evaluate_auc import get_calibration_auc
from delphi.data.dataset import tricolumnar_to_2d
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval.auc import corrective_indices, rates_by_age_bin
from delphi.tokenizer import load_tokenizer_from_ckpt


def load_XT(ckpt: str, task_input: str = "forward"):

    xt_path = os.path.join(DELPHI_CKPT_DIR, ckpt, task_input, "gen.bin")
    XT = np.fromfile(xt_path, dtype=np.uint32).reshape(-1, 3)

    X, T = tricolumnar_to_2d(XT)

    return X, T


def load_Y(
    ckpt: str,
    disease: str,
    timesteps: np.ndarray,
    task_input: str = "forward",
):

    tokenizer = load_tokenizer_from_ckpt(os.path.join(DELPHI_CKPT_DIR, ckpt))
    dis_token = tokenizer[disease]
    logits_path = os.path.join(DELPHI_CKPT_DIR, ckpt, task_input, "logits.bin")
    logits = np.fromfile(logits_path, dtype=np.float16).reshape(
        -1, tokenizer.vocab_size
    )

    Y = np.zeros_like(timesteps, dtype=np.float16)
    sub_idx, pos_idx = np.nonzero(timesteps != -1e4)
    Y[sub_idx, pos_idx] = logits[:, dis_token]

    return logits, Y, dis_token


def new_pipeline(X, T, logits, dis_token, offset):

    now = time.time()

    sub_idx, pos_idx = np.nonzero(T != -1e4)
    max_len = T.shape[1] - 1
    X_t0, X_t1 = X[:, :-1], X[:, 1:]
    T_t0, T_t1 = T[:, :-1], T[:, 1:]
    C = corrective_indices(
        T0=T_t0,
        T1=T_t1,
        offset=offset,
    )

    remove_last = pos_idx < max_len
    sub_idx, pos_idx = sub_idx[remove_last], pos_idx[remove_last]
    logits_idx = np.arange(logits.shape[0])[remove_last]

    offset_pos_idx = C[sub_idx, pos_idx]

    has_input = offset_pos_idx >= 0
    sub_idx, pos_idx = sub_idx[has_input], pos_idx[has_input]
    offset_pos_idx = offset_pos_idx[has_input]
    logits_idx = logits_idx[has_input]

    logits_idx = logits_idx[np.arange(logits_idx.shape[0]) + offset_pos_idx - pos_idx]
    t_t0 = T_t0[sub_idx, offset_pos_idx]
    targets = X_t1[sub_idx, pos_idx]

    disease_free = (~(X_t1 == dis_token).any(axis=1))[sub_idx]
    y_t1 = logits[logits_idx, dis_token]

    age_group_edges = np.arange(45, 85, 5)
    age_groups = [(i, j) for i, j in zip(age_group_edges[:-1], age_group_edges[1:])]
    ctl_subjects, dis_subjects, ctl_rates, dis_rates = rates_by_age_bin(
        input_time=t_t0,
        sub_idx=sub_idx,
        predicted_rates=y_t1,
        disease_free=disease_free,
        targets=targets,
        dis_token=dis_token,
        age_groups=age_groups,
    )
    ctl_counts = np.array([len(np.unique(subj)) for subj in ctl_subjects])
    dis_counts = np.array([len(np.unique(subj)) for subj in dis_subjects])

    time_elapsed = time.time() - now

    return (ctl_counts, dis_counts, ctl_rates, dis_rates), time_elapsed


def baseline(X, T, Y, dis_token, offset):

    now = time.time()

    X_t0, X_t1 = X[:, :-1], X[:, 1:]
    T_t0, T_t1 = T[:, :-1], T[:, 1:]
    Y_t1 = Y[:, :-1]

    age_groups, ctl_counts, dis_counts, ctl_rates, dis_rates = get_calibration_auc(
        j=0, k=dis_token, d=(X_t0, T_t0, X_t1, T_t1), p=Y_t1[..., None], offset=offset
    )  # type: ignore

    time_elapsed = time.time() - now

    return (
        np.array(ctl_counts),
        np.array(dis_counts),
        ctl_rates,
        dis_rates,
    ), time_elapsed


def same_counts(
    ctl_counts_new: np.ndarray,
    dis_counts_new: np.ndarray,
    baseline_ctl_n: np.ndarray,
    baseline_dis_n: np.ndarray,
) -> bool:

    return np.array_equal(ctl_counts_new, baseline_ctl_n) and np.array_equal(
        dis_counts_new, baseline_dis_n
    )


def same_ctl_rates_for_every_age_bin(
    ctl_rates: list[np.ndarray],
    bl_ctl_rates: list[np.ndarray],
) -> bool:

    for ctl_rate, baseline_ctl_rate in zip(ctl_rates, bl_ctl_rates):
        ctl_rate = np.sort(ctl_rate)
        baseline_ctl_rate = np.sort(baseline_ctl_rate)
        if not np.array_equal(ctl_rate, baseline_ctl_rate):
            return False

    return True


def same_dis_rates_for_every_age_bin(
    dis_rates: list[np.ndarray],
    bl_dis_rates: list[np.ndarray],
) -> bool:

    for dis_rate, baseline_dis_rate in zip(dis_rates, bl_dis_rates):
        if not np.array_equal(dis_rate, baseline_dis_rate):
            return False

    return True


def not_too_slow(new_t: float, bl_t: float) -> bool:

    if new_t < bl_t:
        print("new pipeline faster by {:.2f} seconds".format(bl_t - new_t))
        return True
    else:
        print("new pipeline slower by {:.2f} seconds".format(new_t - bl_t))
        return new_t < bl_t * 1.1


ckpt = ["debug"]
disease = ["a41_(other_septicaemia)", "g30_(alzheimer's_disease)", "death"]
# offset = [0.1, 0.5, 1.0, 5.0, 10.0]
offset = [10.0]


@pytest.mark.parametrize(
    "ckpt, disease, offset", [(c, d, o) for c in ckpt for d in disease for o in offset]
)
def test(ckpt: str, disease: str, offset: float):

    X, T = load_XT(ckpt)
    logits, Y, dis_token = load_Y(ckpt, disease, T)
    offset = offset * 365.25  # convert years to days
    (ctl_counts_new, dis_counts_new, ctl_rates, dis_rates), new_t = new_pipeline(
        X, T, logits, dis_token, offset
    )
    (bl_ctl_n, bl_dis_n, bl_ctl_rates, bl_dis_rates), bl_t = baseline(
        X, T, Y, dis_token, offset
    )

    assert same_counts(
        ctl_counts_new=ctl_counts_new,
        dis_counts_new=dis_counts_new,
        baseline_ctl_n=bl_ctl_n,
        baseline_dis_n=bl_dis_n,
    )

    assert same_dis_rates_for_every_age_bin(
        dis_rates=dis_rates,
        bl_dis_rates=bl_dis_rates,
    )
    assert same_ctl_rates_for_every_age_bin(
        ctl_rates=ctl_rates,
        bl_ctl_rates=bl_ctl_rates,
    )

    assert not_too_slow(new_t, bl_t)
