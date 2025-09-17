from typing import Any, Literal, Optional

import numpy as np

from delphi.multimodal import Modality


def append_no_event(
    x: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
    interval: float,
    mode: str = "random",
    token: int = 1,
) -> tuple[np.ndarray, np.ndarray]:

    max_age = np.max(t)
    # add 1e-6 to ensure no_event does not co-occur with first token
    min_age = max(np.min(t[t >= 0]), 0) + 1e-6
    if min_age >= max_age - interval:
        return x, t

    age_range = max_age - min_age
    n = int(age_range // interval) - 1

    if mode == "random":
        no_event_t = rng.uniform(min_age, max_age, size=(n,))
    elif mode == "regular":
        no_event_t = np.linspace(min_age, max_age, num=n)
    else:
        raise ValueError

    no_event_t = no_event_t.astype(np.float32)
    no_event_X = np.full(no_event_t.shape, token)

    x = np.concatenate((x, no_event_X))
    t = np.concatenate((t, no_event_t))

    return x, t


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
    x: np.ndarray,
    *args: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
    mode: Literal["left", "right", "random"] = "left",
):

    L = x.shape[0]

    if L <= block_size:
        return (x, *args) if args else x
    else:
        if mode == "left":
            start = 0
        elif mode == "right":
            start = L - block_size
        elif mode == "random":
            start = rng.integers(0, L - block_size + 1)
        else:
            raise ValueError
        cut = slice(start, start + block_size)
        if args:
            return x[cut], *[arr[cut] for arr in args]
        else:
            return x[cut]


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
