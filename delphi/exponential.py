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


# def integrate_risk(
#     logits: torch.Tensor,
#     tokens: torch.Tensor,
#     timesteps: torch.Tensor,
#     time_intervals: torch.Tensor,
# ):
#     '''
#     input(s):
#         - hazard_rates: [# participants, # timesteps, # tokens]
#         - timesteps: [# participants, # timesteps]
#         - time_intervals: [# intervals]
#         - last_time_by_event: [# participants, # tokens]
#     output(s):
#         - risk: [# participants, # tokens, # intervals]
#     '''
#     _, _, vocab_size = logits.shape
#
#     logits[logits == -torch.inf] = torch.nan
#     hazard_rates = logits[:, :-1].exp()
#
#     last_time_by_event = (
#         timesteps.max(dim=1, keepdim=True)[0].expand(-1, vocab_size).clone()
#     )
#     last_time_by_event = last_time_by_event.scatter_(
#         index=tokens, src=timesteps, dim=1
#     )
#
#     starts = time_intervals[:-1]
#     ends = time_intervals[1:]
#     _timestep = timesteps.unsqueeze(-1).unsqueeze(-1)
#     _timestep = torch.clamp(_timestep, min=starts.view(1, 1, 1, -1))
#     _timestep = torch.clamp(_timestep, max=last_time_by_event.unsqueeze(1).unsqueeze(-1))
#     _timestep = torch.clamp(_timestep, max=ends.view(1, 1, 1, -1))
#     # _timestep: [# participants, # timesteps, # tokens, # intervals]
#     delta_t = torch.diff(_timestep, dim=1)
#     not_enough_exposure = torch.nansum(delta_t, dim=1) < (ends - starts).view(1, 1, -1)
#
#     cumul_hazard = delta_t * hazard_rates.unsqueeze(-1)
#     all_nan = torch.isnan(cumul_hazard).all(dim=1)
#     cumul_hazard = torch.nansum(cumul_hazard, dim=1)
#     # cumul_hazard: [# participants, # tokens, # intervals]
#     # manually set sum of NaNs to Nan because torch.nansum over all NaNs returns 0
#     cumul_hazard[all_nan] = torch.nan
#
#     cumul_hazard[not_enough_exposure] = torch.nan
#
#     risk = 1 - torch.exp(-cumul_hazard)
#
#     return risk
