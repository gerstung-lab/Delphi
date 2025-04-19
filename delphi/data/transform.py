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
        tokenizer,
        interval_in_years: int = 5,
        mode: Literal["regular", "random"] = "random",
        max_age: Optional[float] = None,
    ):

        self.max_age = max_age
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
        min_T = np.min(T, axis=1, keepdims=True)
        max_T = np.max(T, axis=1, keepdims=True)
        max_age = np.max(max_T) if self.max_age is None else self.max_age

        n_no_events = int(max_age // self.no_event_interval)
        if self.mode == "random":
            no_event_timesteps = np.random.randint(
                1, int(max_age * DAYS_PER_YEAR), (n_participants, n_no_events)
            )
        elif self.mode == "regular":
            no_event_timesteps = np.linspace(
                1, int(max_age * DAYS_PER_YEAR), num=n_no_events
            ) * np.ones((n_participants, 1))
        no_event_tokens = np.full(no_event_timesteps.shape, self.tokenizer["no_event"])

        min_T = np.min(T, axis=1, keepdims=True)
        max_T = np.max(T, axis=1, keepdims=True)
        out_of_time_range = (no_event_timesteps <= min_T) * (
            no_event_timesteps >= max_T
        )
        no_event_tokens[out_of_time_range] = 0
        no_event_timesteps[out_of_time_range] = -1e4

        X = np.hstack((X, no_event_tokens))
        T = np.hstack((T, no_event_timesteps))

        return X, T


class AugmentLifestyle:

    def __init__(self):

        pass

    def __call__(self):

        pass


transform_registry = {
    "no-event": AddNoEvent,
    "augment-lifestyle": AugmentLifestyle,
    # Add other transforms here
}


def parse_transform(transform_args: TransformArgs, tokenizer: Tokenizer) -> Any:

    transform_cls = transform_registry[transform_args.name]

    return transform_cls(tokenizer=tokenizer, **transform_args.args)
