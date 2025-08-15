from typing import Optional

import torch

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


def homogenize_piecewise_lambda(
    log_lambda: torch.Tensor, piece_edges: torch.Tensor, horizon: float
) -> torch.Tensor:

    right_edges = piece_edges[:-1]
    too_far = right_edges > horizon
    piece_edges = piece_edges.clamp(None, horizon)
    piece_interval = torch.diff(piece_edges)
    piece_interval[too_far] = 0
    weighted_avg_lamba = (
        torch.exp(log_lambda) * piece_interval.view(1, 1, -1, 1) / piece_interval.sum()
    ).sum(dim=-2, keepdim=False)
    eps = 1e-6

    return torch.log(weighted_avg_lamba + eps)


def sample_piecewise_exponentials(
    log_lambda: torch.Tensor, piece_edges: torch.Tensor, task_tokens: torch.Tensor
):

    B, P, K = log_lambda.shape

    piece_interval = torch.diff(piece_edges)
    if piece_edges[-1] == float("inf"):
        piece_interval = piece_interval[:-1]
        log_lambda = log_lambda[:, :-1, :]
    hazard_rates = (
        torch.exp(log_lambda) * piece_interval.view(1, -1, 1) / piece_interval.sum()
    )
    cumul_hazard_rates = torch.cumsum(hazard_rates, dim=1)

    u = torch.rand((B, K), device=log_lambda.device).log().unsqueeze(1)
    residue_u = u - cumul_hazard_rates
    _, in_piece = torch.max((residue_u > 0).float(), dim=1)
    in_piece = in_piece.unsqueeze(1)
    piece_log_lambda = torch.gather(input=log_lambda, index=in_piece, dim=1).squeeze(1)
    piece_residue_u = torch.gather(input=residue_u, index=in_piece, dim=1).squeeze(1)
    piece_start = piece_edges[:-1][in_piece].squeeze(1)

    ranked_task_tokens, _ = torch.sort(task_tokens)
    next_sample = torch.clamp(
        -torch.exp(-piece_log_lambda) * piece_residue_u, min=0, max=DAYS_PER_YEAR * 80
    ).min(1)
    next_task = next_sample[1][:, None]
    next_time = next_sample[0][:, None] + torch.gather(
        input=piece_start, index=next_task, dim=1
    )
    next_token = ranked_task_tokens[next_task].long()

    return next_token, next_time


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    age: torch.Tensor,
    seed: int,
    max_age: float,  # in days
    termination_tokens: torch.Tensor,
    no_repeat: bool = True,
    top_k: Optional[int] = None,
    temperature: float = 1.0,
):

    torch.manual_seed(seed)
    next_token_generator = model.next_token(
        idx=idx, age=age, no_repeat=no_repeat, top_k=top_k, temperature=temperature
    )

    n, l = idx.shape
    has_termin_token = torch.zeros((n, 1), device=idx.device).bool()
    out_of_time = torch.zeros_like(has_termin_token).bool()

    gen_idx_lst = []
    gen_age_lst = []
    while True:
        terminated = torch.logical_or(has_termin_token, out_of_time)
        if terminated.all() or l > model.config.block_size:
            break

        next_idx, next_age, next_logits = next(next_token_generator)
        gen_idx_lst.append(next_idx)
        gen_age_lst.append(next_age)

        has_termin_token = torch.logical_or(
            has_termin_token, torch.isin(next_idx, termination_tokens)
        )
        out_of_time = torch.logical_or(out_of_time, next_age >= max_age)
        l += 1

    idx = torch.cat(gen_idx_lst, dim=1)
    age = torch.cat(gen_age_lst, dim=1)

    # mask all tokens after the *second* occurrence of a termination token
    exceed_max_age = age > max_age
    is_termination_token = torch.isin(idx, termination_tokens)
    beyond_termination = (
        torch.cumsum(torch.cumsum(is_termination_token.int(), 1), 1) > 1
    )

    pad = exceed_max_age | beyond_termination
    idx[pad] = 0
    age[pad] = -10000

    return idx, age
