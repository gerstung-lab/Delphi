from collections import defaultdict

import numpy as np
import torch
from scipy.stats import rankdata

from delphi.exponential import integrate_risk


def mann_whitney_auc(x1: np.ndarray, x2: np.ndarray) -> float:

    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    n1 = len(x1)
    n2 = len(x2)
    x12 = np.concatenate([x1, x2])
    ranks = rankdata(x12, method="average")

    R1 = ranks[:n1].sum()
    U1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    if n1 == 0 or n2 == 0:
        return np.nan
    return U1 / n1 / n2


def kaplan_meier_incidence(
    surv_prob: np.ndarray, surv_time: np.ndarray, start: float, end: float
):
    """
    assumes the same survival times across all diseases and all participants

    ! this can be vectorized but is actually faster AS IS when # participants is large

    inputs:
        – surv_prob [# participants, # tokens, # time_intervals]
        – surv_time [# time_intervals]
    output(s):
        – incidence [# participants, # tokens]
    """
    assert len(surv_prob.shape) == 3
    assert len(surv_time.shape) == 1
    assert surv_prob.shape[-1] == surv_time.size

    in_range = (surv_time >= start) & (surv_time <= end)
    incidence = list()
    for token in range(surv_prob.shape[1]):
        _estimator = surv_prob[:, token, in_range]
        start_surv, end_surv = _estimator.max(axis=-1), _estimator.min(axis=-1)
        incidence.append((start_surv - end_surv) / start_surv)

    return np.stack(incidence, axis=1)


class KaplanMeierEstimator:

    def __init__(self, surv_percent: list[np.ndarray], surv_time: list[np.ndarray]):
        self.surv_percent = surv_percent
        self.surv_time = surv_time

    @classmethod
    def from_population(cls, timestep: np.ndarray, tokens: np.ndarray, vocab_size: int):

        assert timestep.shape == tokens.shape
        n_subjects = tokens.shape[0]

        # surv_time[i, j] -> exit time for event i in subject j
        # note that exit time can be
        # – time of event (for first occurrence data)
        # – time of death
        # – time at last follow-up
        surv_time = timestep.max(axis=1)[:, None]
        surv_time = np.repeat(surv_time, vocab_size, axis=1)
        np.put_along_axis(arr=surv_time, indices=tokens, values=timestep, axis=1)
        surv_time = surv_time.transpose(1, 0)

        # occur[i, j] -> 1 if subject j experiences event i else 0
        occur = np.zeros((n_subjects, 1))
        occur = np.repeat(occur, vocab_size, axis=1)
        np.put_along_axis(arr=occur, indices=tokens, values=1, axis=1)
        occur = occur.transpose(1, 0)

        sort_surv_time = np.argsort(surv_time, axis=1)
        surv_time = np.take_along_axis(surv_time, indices=sort_surv_time, axis=1)
        occur = np.take_along_axis(occur, indices=sort_surv_time, axis=1)

        surv_percent = list()
        surv_timestep = list()
        for i in range(vocab_size):
            uniq_time, inverse_indices, n_exit = np.unique(
                surv_time[i, :], return_inverse=True, return_counts=True
            )
            n_exit = np.concatenate(([0], n_exit[:-1]))
            n_occur = np.bincount(inverse_indices, weights=occur[i, :])
            n_surv = n_subjects - np.cumsum(n_exit)
            surv_percent.append(np.cumprod(1 - n_occur / n_surv))
            surv_timestep.append(uniq_time)

        return cls(surv_percent, surv_timestep)

    def incidence(self, start_age: float, end_age: float) -> np.ndarray:

        incidence = list()
        for token in range(len(self.surv_percent)):
            in_range = (self.surv_time[token] >= start_age) & (
                self.surv_time[token] <= end_age
            )
            if in_range.sum() > 0:
                _prob = self.surv_percent[token][in_range]
                incidence.append((_prob.max() - _prob.min()) / _prob.max())
            else:
                incidence.append(float("nan"))
        return np.array(incidence)


class IntervalRiskCollector:

    def __init__(self, time_intervals: np.ndarray, n_repeats: int = 1):
        self.risk_per_interval = list()
        self.time_intervals = time_intervals
        self.n_intervals = len(time_intervals) - 1
        self.n_repeats = n_repeats

    def step(self, tokens: torch.Tensor, timestep: torch.Tensor, logits: torch.Tensor):

        _, _, vocab_size = logits.shape

        logits[logits == -torch.inf] = torch.nan
        hazard_rates = logits[:, :-1].exp()

        last_time_by_event = (
            timestep.max(dim=1, keepdim=True)[0].expand(-1, vocab_size).clone()
        )
        last_time_by_event = last_time_by_event.scatter_(
            index=tokens, src=timestep, dim=1
        )

        risk_per_timestep = list()
        for i in range(1, len(self.time_intervals)):
            risk = integrate_risk(
                hazard_rates,
                timestep,
                last_time_by_event,
                start=self.time_intervals[i - 1],
                end=self.time_intervals[i],
            )
            risk_per_timestep.append(risk)
        risk_per_timestep = torch.stack(risk_per_timestep, dim=1)
        # participants * # repeats, # age_groups, # vocab_size
        risk_per_timestep = torch.reshape(
            risk_per_timestep, (-1, self.n_repeats, self.n_intervals, vocab_size)
        )  # participants, # repeats, # age_groups, # vocab_size
        risk_per_timestep = torch.nanmean(risk_per_timestep, dim=1)
        # participants, # age_groups, # vocab_size
        risk_per_timestep = risk_per_timestep.transpose(-1, -2)
        # participants, # vocab_size, # age_groups

        self.risk_per_interval.append(risk_per_timestep.detach().cpu())

    def finalize(self):
        out = torch.cat(self.risk_per_interval, dim=0)
        self.risk_per_interval.clear()
        return out


class SamplingProbCollector:

    def __init__(
        self, vocab_size: int, time_horizon: list, start_age: float, n_repeats: int = 1
    ):
        self.vocab_size = vocab_size
        self.time_horizon = time_horizon
        self.start_age = start_age
        self.n_repeats = n_repeats
        self.prob_by_horizon = defaultdict(list)

    def step(self, tokens: torch.Tensor, timestep: torch.Tensor):

        batch_size, _ = tokens.shape

        occur_time = torch.full(
            (batch_size, self.vocab_size), fill_value=torch.nan, device=tokens.device
        )
        occur_time = (
            occur_time.scatter_(dim=1, index=tokens, src=timestep)
            .detach()
            .cpu()
            .numpy()
        )
        exit_time = timestep.detach().cpu().numpy().max(axis=1)

        for horizon in self.time_horizon:
            end_age = self.start_age + horizon
            occur = np.zeros((batch_size, self.vocab_size))
            occur[np.logical_and(occur_time > self.start_age, occur_time < end_age)] = 1
            occur[occur_time <= self.start_age] = float("nan")
            early_exit = exit_time < end_age
            early_exit = early_exit[:, None]
            occur[np.logical_and(early_exit, occur == 0)] = float("nan")
            occur = np.reshape(occur, (-1, self.n_repeats, occur.shape[-1]))
            occur = np.nanmean(occur, axis=1)
            self.prob_by_horizon[horizon].append(occur)

    def finalize(self):
        prob_by_horizon = dict()
        for horizon in self.prob_by_horizon.keys():
            prob_by_horizon[horizon] = np.concatenate(
                self.prob_by_horizon[horizon], axis=0
            )
        return prob_by_horizon


class EventTimeCollector:

    def __init__(self, vocab_size: int):
        self.exit_time = list()
        self.occur_time = list()
        self.vocab_size = vocab_size

    def step(self, tokens: torch.Tensor, timestep: torch.Tensor):
        batch_size, _ = tokens.shape
        self.exit_time.append(timestep.detach().cpu().numpy().max(axis=1))

        occur_time = torch.full((batch_size, self.vocab_size), fill_value=torch.nan)
        occur_time = occur_time.scatter_(dim=1, index=tokens, src=timestep)
        self.occur_time.append(occur_time.detach().cpu().numpy())

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:

        occur_time = np.concatenate(self.occur_time, axis=0)
        exit_time = np.concatenate(self.exit_time, axis=0)

        return occur_time, exit_time


class LogitCollector:

    def __init__(self, age: float, n_repeats: int = 1):
        self.logits = list()
        self.n_repeats = n_repeats
        self.age = age

    def step(self, tokens: torch.Tensor, timestep: torch.Tensor, logits: torch.Tensor):

        batch_size, _, vocab_size = logits.shape
        logits[logits == -torch.inf] = torch.nan

        collect_idx = torch.argmax(torch.clamp(timestep, max=self.age), dim=1)
        collect_logits = logits[torch.arange(batch_size), collect_idx, :]
        collect_logits = torch.reshape(collect_logits, (-1, self.n_repeats, vocab_size))
        collect_logits = torch.nanmean(collect_logits, dim=1)

        self.logits.append(collect_logits.detach().cpu())

    def finalize(self):

        return torch.cat(self.logits, dim=0).numpy()
