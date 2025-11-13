import torch
import torch.nn.functional as F


def sample_competing_exponentials(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    t_next = torch.clamp(
        -torch.exp(-logits) * torch.rand(logits.shape, device=logits.device).log(),
        min=0,
        max=365.25 * 80.0,
    ).min(1)
    next_token = t_next[1][:, None]
    time_til_next = t_next[0][:, None]

    return next_token, time_til_next


def exponential_nll(
    delta_t: torch.Tensor,
    log_lambda: torch.Tensor,
    t_min: float,
):
    ldt = -torch.log(delta_t + t_min)
    lse = -torch.log(torch.exp(-log_lambda) + t_min)
    nll = -(lse - torch.exp(lse - ldt))
    return nll


def sample_zero_inflated_exponentials(
    logits: torch.Tensor, pi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    next_token, time_til_next = sample_competing_exponentials(logits)

    pi = torch.sigmoid(pi)
    is_comorbid = torch.bernoulli(pi).to(torch.bool)
    time_til_next[is_comorbid] = 0.0
    next_token[is_comorbid.squeeze(-1)] = torch.multinomial(
        F.softmax(logits[is_comorbid.squeeze(-1), :], dim=-1), num_samples=1
    )

    return next_token, time_til_next


def integrate_risk(
    hazard_rates: torch.Tensor,
    timesteps: torch.Tensor,
    last_time_by_event: torch.Tensor,
    start: float,
    end: float,
):
    _timestep = timesteps.unsqueeze(-1)
    _timestep = torch.clamp(_timestep, min=start)
    _timestep = torch.clamp(_timestep, max=last_time_by_event.unsqueeze(1))
    _timestep = torch.clamp(_timestep, max=end)
    delta_t = torch.diff(_timestep, dim=1)
    delta_t[delta_t == 0] = torch.nan
    not_enough_exposure = torch.nansum(delta_t, dim=1) < (end - start)

    cumul_hazard = delta_t * hazard_rates
    all_nan = torch.isnan(cumul_hazard).all(dim=1)
    cumul_hazard = torch.nansum(cumul_hazard, dim=1)
    # manually set sum of NaNs to Nan because torch.nansum over all NaNs returns 0
    cumul_hazard[all_nan] = torch.nan

    cumul_hazard[not_enough_exposure] = torch.nan

    risk = 1 - torch.exp(-cumul_hazard)

    return risk
