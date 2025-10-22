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
    log_lambda: torch.Tensor, age: torch.Tensor, start: float, end: float
):
    r"""
    Aggregate values x over time intervals t within a specified time window [start, end].
    As per the theory of non-homogeneous exponential distribution, the probability
    an event occurs in the time window [start, end] is given by:
    P(event in [start, end]) = 1 - exp(- \int_{start}^{end} \lambda(t) dt)
    where \lambda(t) is the disease rate at time t.
    This this function calculates the integral of the disease rate over the time window
    under that piecewise constant disease rate assumption, using the tokens that
    fall in the time window.

    Args:
        x: Disease rate to integrate, lambda_0, ...., lambda_n, [batch, block_size, disease]
        t: Time points, days since birth, t_0, ...., t_n, t_(n+1) [batch, block_size]
            (the last time point is needed to calculate the duration of the last event)
        start: Start of time window
        end: End of time window

    Returns:
        Aggregated risk values, normalized by time exposure
    """
    pad = torch.clamp(age[:, [-1]], min=end)
    age = torch.cat([age, pad], dim=1)

    t_clamped = age.clamp(start, end)
    dt = t_clamped.diff(1, dim=1)
    dt_norm = dt / (dt.sum(1, keepdim=True) + 1e-6) * (end - start)

    risk = log_lambda.exp() * dt_norm.unsqueeze(-1)
    risk = risk.sum(-2)

    risk[dt.sum(dim=1) == 0] = torch.nan

    return risk
