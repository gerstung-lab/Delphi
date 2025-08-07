from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from delphi import DAYS_PER_YEAR


def truncate_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    topk_value, _ = torch.topk(logits, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    logits[logits < min_value_top_k] = -torch.inf
    return logits


def sample_competing_exponentials(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    t_next = torch.clamp(
        -torch.exp(-logits) * torch.rand(logits.shape, device=logits.device).log(),
        min=0,
        max=DAYS_PER_YEAR * 80.0,
    ).min(1)
    next_token = t_next[1][:, None]
    time_til_next = t_next[0][:, None]

    return next_token, time_til_next


def sample_zero_inflated_exponentials(
    logits: torch.Tensor,
    pi: torch.Tensor,
    always_single_tokens: list,
) -> tuple[torch.Tensor, torch.Tensor]:

    next_token, time_til_next = sample_competing_exponentials(logits)

    pi = torch.sigmoid(pi)
    is_comorbid = torch.bernoulli(pi).to(torch.bool)
    should_be_single = torch.isin(
        next_token, torch.tensor(always_single_tokens, device=logits.device)
    )
    time_til_next[is_comorbid & ~should_be_single] = 0.0

    return next_token, time_til_next


def sample_comorbid_based_on_cutoff(
    logits: torch.Tensor,
    comorbid_cutoff: float,
    always_single_tokens: list,
) -> tuple[torch.Tensor, torch.Tensor]:

    single_token, time_til_single_token = sample_competing_exponentials(logits)

    # logits[..., always_single_tokens] = -torch.inf
    probs = F.softmax(logits, dim=-1)
    probs[..., always_single_tokens] = 0.0
    n_abv_cutoff = (probs >= comorbid_cutoff).sum(-1)
    max_comorbid = max(int(n_abv_cutoff.max().item()), 1)
    if max_comorbid == 1:
        return single_token, time_til_single_token

    has_comorbid = n_abv_cutoff > 1
    comorbid_tokens = torch.zeros(
        logits.shape[0], max_comorbid, device=logits.device, dtype=torch.long
    )
    topk_probs, topk_tokens = torch.topk(probs, k=max_comorbid, dim=-1)
    topk_logits = torch.gather(logits, index=topk_tokens, dim=-1)
    is_comorbid = torch.logical_and(
        has_comorbid.unsqueeze(-1), topk_probs >= comorbid_cutoff
    )
    comorbid_tokens[is_comorbid] = topk_tokens[is_comorbid]
    comorbid_tokens[~has_comorbid, 0] = single_token[~has_comorbid].squeeze()
    _, time_til_comorbid_tokens = sample_competing_exponentials(topk_logits)
    time_til_tokens = time_til_comorbid_tokens.repeat(1, max_comorbid)
    time_til_tokens[~has_comorbid, 0] = time_til_single_token[~has_comorbid].squeeze()
    time_til_tokens[comorbid_tokens == 0] = -10000

    return comorbid_tokens, time_til_tokens


@dataclass
class CausalSamplerConfig:
    seed: int = 42
    no_repeat: bool = True
    top_k: Optional[int] = None
    temperature: float = 1.0
    max_age_in_years: float = 80
    termination_tokens: list[str] = field(default_factory=list)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    model_input: list | tuple,
    seed: int,
    max_age: float,  # in days
    termination_tokens: torch.Tensor,
    no_repeat: bool = True,
    top_k: Optional[int] = None,
    temperature: float = 1.0,
):

    torch.manual_seed(seed)

    while True:

        model_input = model.next_token(
            *model_input, no_repeat=no_repeat, top_k=top_k, temperature=temperature
        )

        idx, age = model_input[0], model_input[1]
        is_termination_token = torch.isin(idx, termination_tokens)
        if torch.logical_or(
            is_termination_token.any(-1), (age[:, -1] > max_age).any(-1)
        ).all():
            break

    exceed_max_age = age > max_age
    # mask all tokens after the *second* occurrence of a termination token
    beyond_termination = (
        torch.cumsum(torch.cumsum(is_termination_token.int(), 1), 1) > 1
    )

    pad = exceed_max_age | beyond_termination
    idx[pad] = 0
    age[pad] = -10000

    logits, _, _ = model(*model_input)

    return idx, age, logits
