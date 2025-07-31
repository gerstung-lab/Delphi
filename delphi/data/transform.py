from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np

from delphi import DAYS_PER_YEAR
from delphi.multimodal import Modality
from delphi.tokenizer import Tokenizer


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
        max_age_in_years: Optional[float] = None,
    ):

        self.rng = np.random.default_rng(seed)
        self.max_age_in_years = max_age_in_years
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

        n_participants = X.shape[0]
        if self.max_age_in_years is None:
            max_age = np.max(T)
        else:
            max_age = self.max_age_in_years * DAYS_PER_YEAR

        n_no_events = int(max_age // self.no_event_interval)
        if self.mode == "random":
            no_event_timesteps = self.rng.integers(
                1, int(max_age - self.no_event_interval), (n_participants, n_no_events)
            )
        elif self.mode == "regular":
            no_event_timesteps = np.linspace(
                1, int(max_age) - self.no_event_interval, num=n_no_events
            ) * np.ones((n_participants, 1))
        no_event_timesteps = no_event_timesteps.astype(np.float32)
        no_event_tokens = np.full(no_event_timesteps.shape, self.tokenizer["no_event"])

        T_cp = T.copy()
        T_cp[T_cp < 0] = 0
        min_T = np.min(T_cp, axis=1, keepdims=True)
        max_T = np.max(T, axis=1, keepdims=True)
        out_of_time_range = (no_event_timesteps <= min_T) | (
            no_event_timesteps >= max_T
        )
        no_event_tokens[out_of_time_range] = 0
        no_event_timesteps[out_of_time_range] = -1e4

        X = np.hstack((X, no_event_tokens))
        T = np.hstack((T, no_event_timesteps))
        M = np.hstack((M, (no_event_tokens != 0).astype(np.int8)))

        return X, T, M, biomarker_X


class AugmentLifestyle:

    def __init__(
        self,
        tokenizer: Tokenizer,
        seed: int,
        lifestyle_tokens: list[str],
        min_time: float = -20 * DAYS_PER_YEAR,
        max_time: float = 40 * DAYS_PER_YEAR,
    ):

        self.rng = np.random.default_rng(seed)
        self.lifestyle_tokens = np.array(tokenizer.encode(lifestyle_tokens))
        self.min_time = min_time
        self.max_time = max_time

    def __call__(
        self,
        X: np.ndarray,
        T: np.ndarray,
        M: np.ndarray,
        biomarker_X: dict[str, np.ndarray],
    ):

        is_lifestyle = np.isin(X, self.lifestyle_tokens)
        if is_lifestyle.sum() > 0:
            augment = self.rng.integers(
                int(self.min_time), int(self.max_time), (is_lifestyle.sum(),)
            )
            T[is_lifestyle] += augment

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
        """
        crop the input data to a fixed block size, prioritizing certain tokens and modalities and preferentially crop out padding tokens
        """

        priority_np = np.zeros(M.shape, dtype=np.uint8)
        priority_np[M > 0] = 1
        if hasattr(self, "priority_tokens"):
            priority_np[np.isin(X, self.priority_tokens)] = 2
        if hasattr(self, "priority_modality"):
            priority_np[np.isin(M, self.priority_modality)] = 2

        tiebreaker = self.rng.integers(0, M.shape[1], size=M.shape, dtype=np.uint32)
        s = np.lexsort((tiebreaker, priority_np), axis=1)
        s_inv = np.argsort(s, axis=1)
        to_keep = np.zeros_like(M, dtype=bool)
        to_keep[:, -self.block_size :] = True
        to_keep = np.take_along_axis(to_keep, s_inv, axis=1)

        for modality, m_X in biomarker_X.items():
            sub_idx, pos_idx = np.where(M == modality.value)
            biomarker_X[modality] = m_X[to_keep[sub_idx, pos_idx], :]

        M = M[to_keep].reshape(M.shape[0], -1)
        X = X[to_keep].reshape(X.shape[0], -1)
        T = T[to_keep].reshape(T.shape[0], -1)

        return X, T, M, biomarker_X


transform_registry = {
    "no-event": AddNoEvent,
    "augment-lifestyle": AugmentLifestyle,
    "crop-block-size": CropBlockSize,
    # Add other transforms here
}


def parse_transform(
    transform_args: TransformArgs, tokenizer: Tokenizer, seed: int
) -> Any:

    transform_cls = transform_registry[transform_args.name]

    return transform_cls(seed=seed, tokenizer=tokenizer, **transform_args.args)
