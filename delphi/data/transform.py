import itertools
from typing import Literal

import numpy as np


def identity_transform(*args):
    return args


def append_no_event(
    x: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
    interval: float,
    mode: str = "random",
    token: int = 1,
) -> tuple[np.ndarray, np.ndarray]:

    if mode == "random":
        max_age = np.max(t)
        # add 1e-6 to ensure no_event does not co-occur with first token
        min_age = max(np.min(t[t >= 0]), 0) + 1e-6
        age_range = max_age - min_age
        n = int(age_range // interval) - 1
        if n <= 0:
            no_event_t = np.array([])
        else:
            no_event_t = rng.uniform(min_age, max_age, size=(n,))
    elif mode == "regular":
        max_age = np.max(t)
        min_age = max(np.min(t[t >= 0]), 0) + 1e-6
        age_range = max_age - min_age
        n = int(age_range // interval) - 1
        if n <= 0:
            no_event_t = np.array([])
        else:
            no_event_t = np.linspace(min_age, max_age, num=n)
    elif mode == "legacy-random":
        min_age = np.min(t[t >= 0])
        max_age = np.max(t)
        no_event_t = rng.uniform(1, 36525, size=(int(36525 / interval),))
        no_event_t = no_event_t[
            np.logical_and(no_event_t >= min_age, no_event_t < max_age)
        ]
    elif mode == "exponential":
        rate = 1 / interval
        dt = np.diff(t)
        n_gaps = len(dt)
        max_per_gap = int(rate * dt.max() * 4)
        exp_samples = rng.exponential(1 / rate, size=(n_gaps, max_per_gap))
        cumsum = np.cumsum(exp_samples, axis=1)
        valid_mask = cumsum < dt[:, None]
        absolute_times = t[:-1, None] + cumsum
        no_event_t = absolute_times[valid_mask]
    else:
        raise ValueError

    no_event_t = no_event_t.astype(np.float32)
    no_event_x = np.full(no_event_t.shape, token)

    x = np.concatenate((x, no_event_x))
    t = np.concatenate((t, no_event_t))

    return x, t


def _crop_slice(mode, max_len, block_size, rng):
    if mode == "left":
        start = 0
    elif mode == "right":
        start = max_len - block_size
    elif mode == "random":
        start = rng.integers(0, max_len - block_size + 1)
    else:
        raise ValueError
    return slice(start, start + block_size)


def crop_contiguous(
    x: np.ndarray,
    *args: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
    mode: Literal["left", "right", "random"] = "left",
):
    """
    input sequences should be sorted according to time
    """

    L = x.shape[0]
    if L <= block_size:
        return (x, *args) if args else x
    else:
        cut = _crop_slice(mode, L, block_size, rng)
        if args:
            return x[cut], *[arr[cut] for arr in args]
        else:
            return x[cut]


def crop_contiguous_multimodal(
    x: np.ndarray,
    biomarker: dict,
    t: np.ndarray,
    m: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
    mode: Literal["left", "right", "random"] = "left",
):
    """
    input sequences should be sorted according to time
    """

    L = x.shape[0]
    if L <= block_size:
        return x, biomarker, t, m
    else:
        keep = _crop_slice(mode, L, block_size, rng)
        mask = np.zeros_like(m).astype(bool)
        mask[keep] = True
        biomarker_keep = dict()
        for modality in biomarker.keys():
            modality_mask = mask[m == modality.value]
            if modality_mask.sum() == 0:
                continue
            else:
                biomarker_keep[modality] = list(
                    itertools.compress(biomarker[modality], modality_mask)
                )

        return x[keep], biomarker_keep, t[keep], m[keep]


def sort_by_time(t: np.ndarray, *args: np.ndarray):
    s = np.argsort(t)
    t = t[s]
    return t, *[arg[s] for arg in args]


def perturb_time(
    x: np.ndarray,
    t: np.ndarray,
    tokens: np.ndarray,
    rng: np.random.Generator,
    low: float = -20 * 365.25,
    high: float = 40 * 365.25,
):
    to_perturb = np.isin(x, tokens)
    t[to_perturb] += rng.uniform(low=low, high=high, size=(to_perturb.sum(),))
    return x, t


def exclude_tokens(x: np.ndarray, t: np.ndarray, blacklist: np.ndarray):
    to_exclude = np.isin(x, blacklist)
    x = x[~to_exclude]
    t = t[~to_exclude]
    return x, t
