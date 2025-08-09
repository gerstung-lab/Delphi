from typing import Any, Optional

import numpy as np

from delphi.multimodal import Modality


def add_no_event(
    X: np.ndarray,
    T: np.ndarray,
    rng: np.random.Generator,
    interval: float,
    mode: str = "random",
    token: int = 1,
) -> tuple[np.ndarray, np.ndarray]:

    B = X.shape[0]
    max_age = np.max(T)
    min_age = max(np.min(T), 0)
    age_range = max_age - min_age
    N = int(age_range // interval)

    if mode == "random":
        no_event_T = rng.integers(min_age, int(max_age - interval), (B, N))
    elif mode == "regular":
        no_event_T = np.linspace(min_age, int(max_age) - interval, num=N) * np.ones(
            (B, 1)
        )
    else:
        raise ValueError

    no_event_T = no_event_T.astype(np.float32)
    no_event_X = np.full(no_event_T.shape, token)

    T_cp = T.copy()
    T_cp[T_cp < 0] = 0
    min_T = np.min(T_cp, axis=1, keepdims=True)
    max_T = np.max(T, axis=1, keepdims=True)
    out_of_time = (no_event_T <= min_T) | (no_event_T >= max_T)
    no_event_X[out_of_time] = 0
    no_event_T[out_of_time] = -1e4

    X = np.hstack((X, no_event_X))
    T = np.hstack((T, no_event_T))

    return X, T


def sort_by_time(T: np.ndarray, *args: np.ndarray):

    s = np.argsort(T, axis=1)
    T = np.take_along_axis(T, s, axis=1)

    if args and any(arr.shape != T.shape for arr in args):
        raise ValueError("all arrays must have the same shape as T")

    return T, *[np.take_along_axis(arr, s, axis=1) for arr in args]


def trim_margin(reference: np.ndarray, *args: np.ndarray, trim_val: Any):

    margin = np.min(np.sum(reference == trim_val, axis=1))

    return reference[:, margin:], *[arr[:, margin:] for arr in args]


def crop_contiguous(
    X: np.ndarray, *args: np.ndarray, block_size: int, rng: np.random.Generator
):

    L = X.shape[1]

    if L <= block_size:
        if args:
            return X, *args
        else:
            return X
    else:
        start = rng.integers(0, L - block_size + 1)
        cut = slice(start, start + block_size)
        if args:
            return X[:, cut], *[arr[:, cut] for arr in args]
        else:
            return X[:, cut]


def crop_priority(
    X: np.ndarray,
    T: np.ndarray,
    M: np.ndarray,
    biomarker_X: dict[Modality, np.ndarray],
    priority_tokens: Optional[np.ndarray],
    priority_modality: Optional[np.ndarray],
    block_size: int,
    rng: np.random.Generator,
):
    """
    crop the input data to a fixed block size, prioritizing certain tokens and modalities and preferentially crop out padding tokens
    """

    priority_np = np.zeros(M.shape, dtype=np.uint8)
    priority_np[M > 0] = 1
    if priority_tokens is not None:
        priority_np[np.isin(X, priority_tokens)] = 2
    if priority_modality is not None:
        priority_np[np.isin(M, priority_modality)] = 2

    tiebreaker = rng.integers(0, M.shape[1], size=M.shape, dtype=np.uint32)
    s = np.lexsort((tiebreaker, priority_np), axis=1)
    s_inv = np.argsort(s, axis=1)
    to_keep = np.zeros_like(M, dtype=bool)
    to_keep[:, -block_size:] = True
    to_keep = np.take_along_axis(to_keep, s_inv, axis=1)

    for modality, m_X in biomarker_X.items():
        sub_idx, pos_idx = np.where(M == modality.value)
        biomarker_X[modality] = m_X[to_keep[sub_idx, pos_idx], :]

    M = M[to_keep].reshape(M.shape[0], -1)
    X = X[to_keep].reshape(X.shape[0], -1)
    T = T[to_keep].reshape(T.shape[0], -1)

    return X, T, M, biomarker_X
