from typing import Literal

import numpy as np

from delphi import DAYS_PER_YEAR


def parse_transform():
    pass


class AddNoEvent:

    def __init__(
        self,
        tokenizer,
        no_event_interval: int = 5,
        mode: Literal["regular", "random"] = "random",
        max_age: int = 100,
    ):

        self.max_age = int(max_age)

        assert (
            no_event_interval <= self.max_age / 2
        ), "no_event_interval must be less than half of max_age"
        self.no_event_interval = no_event_interval

        assert mode in [
            "regular",
            "random",
        ], "mode must be either 'regular' or 'random'"
        self.mode = mode

        self.tokenizer = tokenizer

        pass

    def __call__(self, X, T):

        n_participants = X.shape[0]
        n_no_events = self.max_age // self.no_event_interval
        if self.mode == "random":
            no_event_time_steps = np.random.randint(
                1, int(self.max_age * DAYS_PER_YEAR), (n_participants, n_no_events)
            )
        elif self.mode == "regular":
            no_event_time_steps = np.linspace(
                1, int(self.max_age * DAYS_PER_YEAR), num=n_no_events
            ) * np.ones((n_participants, 1))

        no_event_tokens = np.full(self.tokenizer.no_event, no_event_time_steps.shape)

        min_T = np.min(T, axis=1, keepdims=True)
        max_T = np.max(T, axis=1, keepdims=True)
        out_of_time_range = (no_event_time_steps < min_T) * (
            no_event_time_steps > max_T
        )
        no_event_tokens[out_of_time_range] = 0
        no_event_time_steps[out_of_time_range] = -1e4

        X = np.hstack((X, no_event_tokens))
        T = np.hstack((T, no_event_time_steps))

        return X, T


class AugmentLifestyle:

    def __init__(self):

        pass

    def __call__(self):

        pass
