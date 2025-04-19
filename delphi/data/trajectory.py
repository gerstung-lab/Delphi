from typing import Literal, Optional, Tuple

import numpy as np

OptionalSearchSpace = Optional[Literal["input", "target"]]
OptionalTimeRange = Tuple[Optional[float], Optional[float]]


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
        keep: Literal["first", "average", "last", "random"] = "average",
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
