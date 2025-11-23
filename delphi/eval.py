from collections import defaultdict

import numpy as np
import torch
from scipy.stats import rankdata

from delphi.multimodal import Modality


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


def integrate_risk(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    timesteps: torch.Tensor,
    time_intervals: torch.Tensor,
):
    """
    not vectorized due to memory concerns when time_intervals are dense

    input(s):
        - hazard_rates: [# participants, # timesteps, # tokens]
        - timesteps: [# participants, # timesteps]
        - time_intervals: [# intervals]
        - last_time_by_event: [# participants, # tokens]
    output(s):
        - risk: [# participants, # tokens, # intervals]
    """
    _, _, vocab_size = logits.shape

    logits[logits == -torch.inf] = torch.nan
    hazard_rates = logits[:, :-1].exp()

    last_time_by_event = (
        timesteps.max(dim=1, keepdim=True)[0].expand(-1, vocab_size).clone()
    )
    last_time_by_event = last_time_by_event.scatter_(index=tokens, src=timesteps, dim=1)

    starts = time_intervals[:-1]
    ends = time_intervals[1:]
    risks = list()
    for start, end in zip(starts, ends):
        _timestep = timesteps.unsqueeze(-1)
        _timestep = torch.clamp(_timestep, min=start)
        _timestep = torch.clamp(_timestep, max=last_time_by_event.unsqueeze(1))
        _timestep = torch.clamp(_timestep, max=end)
        # _timestep: [# participants, # timesteps, # tokens]
        delta_t = torch.diff(_timestep, dim=1)
        not_enough_exposure = torch.nansum(delta_t, dim=1) < (end - start)

        cumul_hazard = delta_t * hazard_rates
        all_nan = torch.isnan(cumul_hazard).all(dim=1)
        cumul_hazard = torch.nansum(cumul_hazard, dim=1)
        # cumul_hazard: [# participants, # tokens]
        # manually set sum of NaNs to Nan because torch.nansum over all NaNs returns 0
        cumul_hazard[all_nan] = torch.nan
        cumul_hazard[not_enough_exposure] = torch.nan

        risk = 1 - torch.exp(-cumul_hazard)
        risks.append(risk)

    return torch.stack(risks, dim=-1)


class IntervalRiskCollator:

    def __init__(self, time_intervals: list[float], n_repeats: int = 1):
        self.risk_per_interval = list()
        self.time_intervals = time_intervals
        self.n_intervals = len(time_intervals) - 1
        self.n_repeats = n_repeats

    def step(self, tokens: torch.Tensor, timestep: torch.Tensor, logits: torch.Tensor):

        _, _, vocab_size = logits.shape

        risk_per_interval = integrate_risk(
            logits=logits,
            tokens=tokens,
            timesteps=timestep,
            time_intervals=torch.tensor(self.time_intervals).to(tokens.device),
        )
        risk_per_interval = torch.reshape(
            risk_per_interval,
            (-1, self.n_repeats, vocab_size, len(self.time_intervals) - 1),
        )  # participants, # repeats, # vocab_size, # time_intervals
        risk_per_interval = torch.nanmean(risk_per_interval, dim=1)
        # participants, # vocab_size, # time_intervals

        self.risk_per_interval.append(risk_per_interval.detach().cpu())

    def finalize(self):
        out = torch.cat(self.risk_per_interval, dim=0)
        self.risk_per_interval.clear()
        return out


class IntervalKaplanMeierCollator:

    def __init__(
        self,
        time_horizon: list[float],
        start_age: float,
        time_intervals: None | list[float] = None,
        n_repeats: int = 1,
    ):
        self.time_intervals = time_intervals
        self.time_horizon = time_horizon
        self.start_age = start_age
        self.n_repeats = n_repeats
        self.prob_by_horizon = defaultdict(list)

    def step(self, tokens: torch.Tensor, timestep: torch.Tensor, logits: torch.Tensor):

        if self.time_intervals is None:
            time_intervals = (
                torch.unique(torch.clamp(timestep, min=0), sorted=True)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            time_intervals = self.time_intervals

        risk_per_interval = integrate_risk(
            logits=logits,
            tokens=tokens,
            timesteps=timestep,
            time_intervals=torch.tensor(self.time_intervals).to(tokens.device),
        )
        risk_per_interval = torch.reshape(
            risk_per_interval,
            (-1, self.n_repeats, logits.shape[-1], len(time_intervals) - 1),
        )  # participants, # repeats, # vocab_size, # time_intervals
        risk_per_interval = torch.nanmean(risk_per_interval, dim=1)
        # participants, # vocab_size, # time_intervals

        surv_prob = torch.cumprod(1 - risk_per_interval, dim=-1)
        surv_time = np.array(time_intervals)[1:]
        for horizon in self.time_horizon:
            self.prob_by_horizon[horizon].append(
                kaplan_meier_incidence(
                    surv_prob=surv_prob.detach().cpu().numpy(),
                    surv_time=surv_time,
                    start=self.start_age,
                    end=self.start_age + horizon,
                )
            )

    def finalize(self):
        prob_by_horizon = dict()
        for horizon in self.prob_by_horizon.keys():
            prob_by_horizon[horizon] = np.concatenate(
                self.prob_by_horizon[horizon], axis=0
            )
        return prob_by_horizon


class SamplingProbCollator:

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


class EventTimeCollator:

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


def corrective_indices(T0: torch.Tensor, T1: torch.Tensor, offset: float):
    assert T0.shape == T1.shape  # (m, n)
    T0_expanded = T0.unsqueeze(1)  # (m, 1, n)
    T1_expanded = T1.unsqueeze(-1)  # (m, n, 1)
    C = (T0_expanded <= (T1_expanded - offset)).sum(dim=2) - 1

    return C.long()


def correct_time_offset(
    T0: torch.Tensor, T1: torch.Tensor, logits: torch.Tensor, offset: float
):
    corr_idx = corrective_indices(T0, T1, offset)
    invalid = corr_idx == -1
    corr_idx = torch.clamp(corr_idx, min=0)
    T0 = torch.gather(input=T0, index=corr_idx, dim=1)
    logits = torch.gather(
        input=logits,
        index=corr_idx.unsqueeze(-1).expand(-1, -1, logits.shape[-1]),
        dim=1,
    )

    T0[invalid] = -10000
    logits[invalid] = -torch.inf

    return T0, logits


def sample_boolean_mask(mask):
    """Sample one True value per row from a boolean mask (vectorized)."""
    n_rows = mask.shape[0]
    result = torch.zeros_like(mask).bool()

    # Count True values per row
    counts = mask.sum(dim=1)
    has_true = counts > 0

    if not has_true.any():
        return result

    # For rows with at least one True, generate random positions
    random_positions = torch.rand(n_rows, mask.shape[1])
    random_positions[~mask] = -torch.inf  # Mask out False positions

    # Select the position with max random value per row
    selected_cols = torch.argmax(random_positions, dim=1)
    result[torch.arange(n_rows), selected_cols] = has_true

    return result


class AgeStratRatesCollator:

    def __init__(self, age_groups: torch.Tensor):
        self.age_groups = age_groups
        self.ctl_rates = list()

    def step(self, timesteps, logits):
        bin_assignments = torch.searchsorted(self.age_groups, timesteps, right=True)
        bin_assignments -= 1

        ctl_rates = list()
        for bin_idx in range(len(self.age_groups) - 1):
            bin_mask = sample_boolean_mask(bin_assignments == bin_idx)
            ctl_rate = torch.full(
                (logits.shape[0], logits.shape[-1]),
                dtype=logits.dtype,
                fill_value=torch.nan,
            ).to(logits.device)
            ctl_rate[bin_mask.any(dim=-1)] = logits[bin_mask, :]
            ctl_rates.append(ctl_rate)
        ctl_rates = torch.stack(ctl_rates, dim=1)

        self.ctl_rates.append(ctl_rates.detach().cpu())

    def finalize(self):
        return torch.concat(self.ctl_rates)


class DiseaseRatesCollator:

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.dis_rates = list()
        self.dis_times = list()

    def step(self, tokens, timesteps, logits):

        dis_time = torch.full(
            (logits.shape[0], logits.shape[-1]),
            dtype=timesteps.dtype,
            fill_value=torch.nan,
        ).to(logits.device)
        dis_time.scatter_(index=tokens, src=timesteps, dim=1)
        self.dis_times.append(dis_time.detach().cpu())

        dis_rate = torch.full(
            (logits.shape[0], logits.shape[-1]),
            dtype=logits.dtype,
            fill_value=torch.nan,
        ).to(logits.device)
        uniq_tokens = torch.unique(tokens)
        uniq_tokens = uniq_tokens[uniq_tokens > 1]
        for token in uniq_tokens:
            have_disease = tokens == token
            dis_rate[have_disease.any(dim=1), token] = logits[have_disease][:, token]
        self.dis_rates.append(dis_rate.detach().cpu())

    def finalize(self):
        return torch.concat(self.dis_rates), torch.concat(self.dis_times)


class SexCollator:

    def __init__(self):
        self.is_female = list()

    def step(self, tokens):
        self.is_female.append((tokens == 2).any(dim=1).detach().cpu())

    def finalize(self):
        return torch.concat(self.is_female)


class ModalityCollator:

    def __init__(self, modalities: list[str]):
        self.modalities = [Modality[modality.upper()] for modality in modalities]
        self.max_mod = max([modality.value for modality in self.modalities])
        self.mod_timesteps = list()

    def step(self, mod_tokens, timesteps):
        assert mod_tokens.shape == timesteps.shape
        mod_timesteps = torch.full(
            (timesteps.shape[0], self.max_mod + 1), fill_value=torch.nan
        ).to(timesteps.device)
        mod_timesteps = mod_timesteps.scatter_(dim=1, src=timesteps, index=mod_tokens)
        self.mod_timesteps.append(mod_timesteps.detach().cpu())

    def finalize(self):
        return torch.cat(self.mod_timesteps, dim=0)
