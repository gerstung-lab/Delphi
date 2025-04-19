import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_array

from delphi.data.dataset import get_p2i

OptionalTimeRange = Tuple[Optional[float], Optional[float]]


class Cohort:

    def __init__(
        self, participants: np.ndarray, tokens: np.ndarray, time_steps: np.ndarray
    ):

        self.participants = participants
        self.n_participants = len(participants)

        assert tokens.shape[0] == self.n_participants
        self.tokens = tokens

        assert time_steps.shape == self.tokens.shape
        self.time_steps = time_steps

    def __len__(self):

        return self.n_participants

    def __getitem__(self, key):

        return Cohort(
            participants=self.participants[key],
            tokens=self.tokens[key],
            time_steps=self.time_steps[key],
        )

    def token_counts(
        self,
        token: int,
        time_range: OptionalTimeRange = (None, None),
    ) -> np.ndarray:

        in_time_range = np.ones_like(self.time_steps, dtype=bool)
        start, end = time_range
        if start is not None:
            in_time_range &= self.time_steps >= start
        if end is not None:
            in_time_range &= self.time_steps < end

        return np.logical_and(self.tokens == token, in_time_range).sum(axis=1)

    def has_token(
        self,
        token: int,
        min_count: int = 1,
        time_range: OptionalTimeRange = (None, None),
    ) -> np.ndarray:

        counts = self.token_counts(token, time_range=time_range)

        return counts >= min_count

    def has_any_token(self, time_range: OptionalTimeRange = (None, None)) -> np.ndarray:

        in_time_range = np.ones_like(self.time_steps, dtype=bool)
        start, end = time_range
        if start is not None:
            in_time_range &= self.time_steps >= start
        if end is not None:
            in_time_range &= self.time_steps < end

        return in_time_range.any(axis=1)

    def survival_histogram(
        self, time_bins, clip_age: OptionalTimeRange = (None, None)
    ) -> Tuple[np.ndarray, np.ndarray]:

        age_at_death = self.time_steps.max(axis=1)
        if clip_age != (None, None):
            clip_min, clip_max = clip_age
            age_at_death = np.clip(age_at_death, a_min=clip_min, a_max=clip_max)

        death_hist, time_bin_edges = np.histogram(age_at_death, time_bins)

        surv_hist = self.n_participants - np.cumsum(death_hist)

        return surv_hist, time_bin_edges

    def token_incidence(
        self, token: int, time_bins: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        survival_participant_hist, _ = self.survival_histogram(time_bins)
        token_count_hist, time_bin_edges = np.histogram(
            self.time_steps[self.tokens == token], time_bins
        )

        return token_count_hist / survival_participant_hist, time_bin_edges

    def token_timestamps(self, token: int) -> np.ndarray:

        assert self.has_token(
            token
        ).all(), f"all participants must have token {token} in their token sequence"

        has_token = self.tokens == token

        # todo: sampling here

        return np.sum(has_token * self.time_steps, axis=1)


def cohort_from_ukb_data(
    data: Optional[np.ndarray] = None, data_path: Optional[str] = None
):

    if data is None:  # if raw data is not provided, load from file
        assert data_path is not None, "Either data_path or data must be provided"
        assert data_path.endswith(".bin"), "Cohort data must be a .bin NumPy memory bin"
        data = np.fromfile(data_path, dtype=np.uint32).reshape(-1, 3)
    else:
        data = data  # allow direct data injection

    participants = pd.unique(data[:, 0])

    p2i = get_p2i(data)
    lengths = p2i[:, 1]
    row_idx = np.repeat(np.arange(participants.size), lengths)
    col_idx = np.concatenate([np.arange(length) for length in lengths])
    tokens = coo_array(
        (data[:, 2] + 1, (row_idx, col_idx)), shape=(participants.size, lengths.max())
    ).toarray()
    time_steps = coo_array(
        (data[:, 1], (row_idx, col_idx)), shape=(participants.size, lengths.max())
    ).toarray()

    return Cohort(participants=participants, tokens=tokens, time_steps=time_steps)
