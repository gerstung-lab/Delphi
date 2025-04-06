import os
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_array

PathOrArray = Union[os.PathLike, np.ndarray]
OptionalSearchSpace = Optional[Literal["input", "target"]]
OptionalTimeRange = Tuple[Optional[float], Optional[float]]


def get_p2i(data):
    """
    Get the patient to index mapping. (patient index in data -> length of sequence)
    """

    px = data[:, 0].astype("int")
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
        if i == len(px) - 1:
            # add last participant
            p2i.append([j, i - j + 1])
    return np.array(p2i)


def data_loader(
    dataset: "Dataset",
    batch_size: int,
    transforms: Optional[List[Callable]] = None,
):

    while True:

        ix = np.random.randint(len(dataset), (batch_size,))
        _, X, T = dataset.get_batch(ix)

        if transforms is not None:
            for transform in transforms:
                X, T = transform(X, T)

        X = torch.tensor(X, dtype=torch.long)
        T = torch.tensor(T, dtype=torch.float)

        X_t0 = X[:, :-1]
        T_t0 = T[:, :-1]
        X_t1 = X[:, 1:]
        T_t1 = T[:, 1:]

        yield X_t0, T_t0, X_t1, T_t1


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


class Dataset:

    def __init__(
        self,
        participants: PathOrArray,
        tokens: PathOrArray,
        time_steps: PathOrArray,
        start_pos: PathOrArray,
        seq_len: PathOrArray,
    ):

        self.tokens = self._load_array(tokens, dtype=np.uint32)
        self.time_steps = self._load_array(time_steps, dtype=np.uint32)
        assert self.tokens.size == self.time_steps.size

        self.participants = self._load_array(participants, dtype=np.uint32)
        self.n_participants = self.participants.size

        self.start_pos = self._load_array(start_pos, dtype=np.uint32)
        assert self.start_pos.size == self.n_participants
        self.seq_len = self._load_array(seq_len, dtype=np.uint32)
        assert self.seq_len.size == self.n_participants

    def _load_array(self, data: PathOrArray, dtype) -> np.ndarray:
        if isinstance(data, os.PathLike):
            return np.fromfile(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            return data.astype(dtype, copy=False)
        else:
            raise ValueError("Input must be a file path or a NumPy array")

    def __len__(self):

        return self.n_participants

    def get_batch(self, batch_idx):

        batch_size = batch_idx.size
        batch_len = self.seq_len[batch_idx]
        batch_slice = np.concatenate(
            [np.arange(s, s + l) for s, l in zip(self.start_pos[batch_idx], batch_len)]
        )
        batch_tokens = self.tokens[batch_slice]
        batch_time_steps = self.time_steps[batch_slice]
        row_idx = np.repeat(np.arange(batch_size), batch_len)
        col_idx = np.concatenate([np.arange(length) for length in batch_len])

        P = self.participants[batch_idx]

        X = coo_array(
            (batch_tokens, (row_idx, col_idx)), shape=(batch_size, batch_len.max())
        ).toarray()
        T = coo_array(
            (batch_time_steps, (row_idx, col_idx)), shape=(batch_size, batch_len.max())
        ).toarray()

        return P, X, T


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


class DelphiTrajectories:

    def __init__(
        self,
        X_t0: np.ndarray,
        T_t0: np.ndarray,
        X_t1: np.ndarray,
        T_t1: np.ndarray,
        Y_t1: np.ndarray,
    ):

        assert X_t0.shape == T_t0.shape == X_t1.shape == T_t1.shape

        X0_pad_mask = X_t0 == 0
        assert (T_t0[X0_pad_mask] == -10000).all()
        X1_pad_mask = X_t1 == 0
        assert (T_t1[X1_pad_mask] == -10000).all()

        self.X0 = X_t0
        self.T0 = T_t0
        self.X1 = X_t1
        self.T1 = T_t1

        self.n_participants = X_t0.shape[0]
        self.max_seq_len = X_t0.shape[1]

        assert Y_t1.shape[:-1] == X_t0.shape
        self.Y_t1 = Y_t1

    def __getitem__(self, key):

        if isinstance(key, tuple):
            raise NotImplementedError(
                "DelphiTrajectories only supports slicing by participant index"
            )
        else:
            return DelphiTrajectories(
                X_t0=self.X0[key],
                T_t0=self.T0[key],
                X_t1=self.X1[key],
                T_t1=self.T1[key],
                Y_t1=self.Y_t1[key],
            )

    def _in_time_range(
        self,
        t0_range: OptionalTimeRange = (None, None),
        t1_range: OptionalTimeRange = (None, None),
    ) -> np.ndarray:

        in_t0_range = np.ones_like(self.T0, dtype=bool)
        t0_start, t0_end = t0_range
        if t0_start is not None:
            in_t0_range &= self.T0 >= t0_start
        if t0_end is not None:
            in_t0_range &= self.T0 < t0_end

        in_t1_range = np.ones_like(self.T1, dtype=bool)
        t0_start, t0_end = t1_range
        if t0_start is not None:
            in_t1_range &= self.T1 >= t0_start
        if t0_end is not None:
            in_t1_range &= self.T1 < t0_end

        return in_t0_range & in_t1_range

    def _sample_token_mask(
        self,
        token_mask: np.ndarray,
        keep: Literal["first", "average"] = "average",
    ):

        token_mask = token_mask.astype(float)
        if keep == "average":
            token_mask /= np.sum(token_mask, axis=1, keepdims=True)
        elif keep == "first":
            token_mask = token_mask * (np.cumsum(token_mask, axis=1) == 1).astype(float)
        else:
            raise ValueError("keep must be either 'first' or 'average'")

        return token_mask

    def has_any_token(
        self,
        t0_range: OptionalTimeRange = (None, None),
        t1_range: OptionalTimeRange = (None, None),
    ) -> np.ndarray:
        """
        checks if any input/target token is present in the specified time range

        t0_range: tuple of (start, end) for the input time range
        t1_range: tuple of (start, end) for the target time range

        returns: boolean array of shape (n_participants,)
        """

        in_time_range = self._in_time_range(t0_range=t0_range, t1_range=t1_range)

        return in_time_range.any(axis=1)

    def has_any_valid_predictions(
        self,
        min_time_gap: float,
        t0_range: OptionalTimeRange = (None, None),
        t1_range: OptionalTimeRange = (None, None),
    ):

        is_valid = np.cumsum(self.T0 <= (self.T1 - min_time_gap), axis=1) > 0
        in_time = self._in_time_range(t0_range=t0_range, t1_range=t1_range)

        return np.logical_and(in_time, is_valid).any(axis=1)

    def has_token(
        self,
        token: int,
        token_type: OptionalSearchSpace = "target",
        t0_range: OptionalTimeRange = (None, None),
        t1_range: OptionalTimeRange = (None, None),
        min_count: int = 1,
    ) -> np.ndarray:
        """
        checks if a specific input/target token is present in the specified time range

        token_type: 'input' or 'target'
        t0_range: tuple of (start, end) for the input time range
        t1_range: tuple of (start, end) for the target time range
        min_count: minimum number of tokens required to be present in the time range

        returns: boolean array of shape (n_participants,)
        """

        if token_type == "input":
            has_token = self.X0 == token
        elif token_type == "target":
            has_token = self.X1 == token
        elif token_type is None:
            has_token = np.logical_or(self.X0 == token, self.X1 == token)
        else:
            raise ValueError("token_type must be either 'input', 'target', or None")

        in_time = self._in_time_range(t0_range=t0_range, t1_range=t1_range)

        return np.logical_and(has_token, in_time).sum(axis=1) >= min_count

    def token_rates(
        self,
        token: int,
        t0_range: OptionalTimeRange = (None, None),
        t1_range: OptionalTimeRange = (None, None),
        keep: Literal["first", "average", "last", "random"] = "average",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        computes the token rates for a specific input/target token in the specified time range
        computes a single token rate for each participant in the specified time range

        token: token ID
        t0_range: tuple of (start, end) for the input time range
        t1_range: tuple of (start, end) for the target time range
        keep: 'first', 'average', 'last', or 'random'

        returns: tuple of (input time, token rates)
        input time: input timestamp array of shape (n_participants,)
        token rates: token rates array of shape (n_participants,)
        """

        assert (
            token < self.Y_t1.shape[-1]
        ), "token ID out of range; check that correct tokenizer is used"

        assert self.has_any_token(
            t0_range=t0_range, t1_range=t1_range
        ).all(), "no tokens in the specified time range"

        token_rates = self.Y_t1[:, :, token]

        in_time_range = self._in_time_range(t0_range=t0_range, t1_range=t1_range)
        in_time_range = in_time_range.astype(float)
        in_time_range = self._sample_token_mask(in_time_range, keep=keep)

        t0 = np.sum(self.T0 * in_time_range, axis=1)
        y1 = np.sum(token_rates * in_time_range, axis=1)

        return t0, y1

    def penultimate_token_rates(
        self,
        token: int,
    ) -> Tuple[np.ndarray, np.ndarray]:

        assert self.has_token(
            token, token_type="target"
        ).all(), f"all participants must have token {token} in their target sequence"

        assert (
            token < self.Y_t1.shape[-1]
        ), "token ID out of range; check that correct tokenizer is used"
        token_rates = self.Y_t1[:, :, token]

        token_mask = self.X1 == token
        token_mask = self._sample_token_mask(token_mask, keep="first")

        t0 = np.sum(self.T0 * token_mask, axis=1)
        y1 = np.sum(token_rates * token_mask, axis=1)

        return t0, y1
