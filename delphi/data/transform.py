from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np

from delphi import DAYS_PER_YEAR
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

    def __call__(self, X, T):

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

        return X, T


class AugmentLifestyle:

    def __init__(
        self,
        tokenizer: Tokenizer,
        seed: int,
        min_time: float = -20 * DAYS_PER_YEAR,
        max_time: float = 40 * DAYS_PER_YEAR,
    ):

        self.rng = np.random.default_rng(seed)
        self.lifestyle_tokens = np.array(tokenizer.lifestyle_tokens)
        self.min_time = min_time
        self.max_time = max_time

    def __call__(self, X: np.ndarray, T: np.ndarray):

        is_lifestyle = np.isin(X, self.lifestyle_tokens)
        if is_lifestyle.sum() > 0:
            augment = self.rng.integers(
                int(self.min_time), int(self.max_time), (is_lifestyle.sum(),)
            )
            T[is_lifestyle] += augment

        return X, T


transform_registry = {
    "no-event": AddNoEvent,
    "augment-lifestyle": AugmentLifestyle,
    # Add other transforms here
}


def parse_transform(
    transform_args: TransformArgs, tokenizer: Tokenizer, seed: int
) -> Any:

    transform_cls = transform_registry[transform_args.name]

    return transform_cls(seed=seed, tokenizer=tokenizer, **transform_args.args)
