from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np

from delphi import DAYS_PER_YEAR
from delphi.multimodal import Modality
from delphi.tokenizer import Tokenizer


def add_no_event(
    X: np.ndarray,
    T: np.ndarray,
    rng: np.random.Generator,
    interval: float,
    mode: str,
    token: int = 1,
) -> tuple[np.ndarray, np.ndarray]:

    B = X.shape[0]
    max_age = np.max(T)
    N = int(max_age // interval)

    if mode == "random":
        no_event_T = rng.integers(1, int(max_age - interval), (B, N))
    elif mode == "regular":
        no_event_T = np.linspace(1, int(max_age) - interval, num=N) * np.ones((B, 1))
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
        return X, *args
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


@dataclass
class TransformArgs:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)


class AddNoEvent:

    def __init__(
        self,
        tokenizer: Tokenizer,
        seed: int,
        interval_in_years: int = 5,
        mode: Literal["regular", "random"] = "random",
    ):

        self.rng = np.random.default_rng(seed)
        self.no_event_interval = interval_in_years * DAYS_PER_YEAR

        assert mode in [
            "regular",
            "random",
        ], "mode must be either 'regular' or 'random'"
        self.mode = mode

        self.tokenizer = tokenizer

        pass

    def __call__(
        self,
        X: np.ndarray,
        T: np.ndarray,
        M: np.ndarray,
        biomarker_X: dict[str, np.ndarray],
    ):

        X, T = add_no_event(
            X=X,
            T=T,
            rng=self.rng,
            interval=self.no_event_interval,
            mode=self.mode,
            token=self.tokenizer["no_event"],
        )

        no_event_X = X[:, M.shape[1] :]
        M = np.hstack((M, (no_event_X != 0).astype(np.int8)))

        return X, T, M, biomarker_X


class CropBlockSize:

    def __init__(
        self,
        tokenizer: Tokenizer,
        seed: int,
        block_size: int,
        priority_tokens: Optional[list[str]] = None,
        priority_modality: Optional[list[str]] = None,
        **kwargs,
    ):

        self.rng = np.random.default_rng(seed)
        self.block_size = block_size
        if priority_tokens:
            self.priority_tokens = np.array(tokenizer.encode(priority_tokens))
        if priority_modality:
            self.priority_modality = np.array(
                [Modality[modality.upper()].value for modality in priority_modality]
            )

    def __call__(
        self,
        X: np.ndarray,
        T: np.ndarray,
        M: np.ndarray,
        biomarker_X: dict[Modality, np.ndarray],
    ):

        X, T, M, biomarker_X = crop_priority(
            X=X,
            T=T,
            M=M,
            biomarker_X=biomarker_X,
            priority_tokens=(
                self.priority_tokens if hasattr(self, "priority_tokens") else None
            ),
            priority_modality=(
                self.priority_modality if hasattr(self, "priority_modality") else None
            ),
            block_size=self.block_size,
            rng=self.rng,
        )

        return X, T, M, biomarker_X


transform_registry = {
    "no-event": AddNoEvent,
    "crop-block-size": CropBlockSize,
    # Add other transforms here
}


def parse_transform(
    transform_args: TransformArgs, tokenizer: Tokenizer, seed: int
) -> Any:

    transform_cls = transform_registry[transform_args.name]

    return transform_cls(seed=seed, tokenizer=tokenizer, **transform_args.args)
