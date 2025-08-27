from typing import Optional

import torch
from transformers import DynamicCache

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
    max_age: Optional[float] = None,  # in days
    max_time: Optional[float] = None,
    termination_tokens: Optional[torch.Tensor] = None,
    no_repeat: bool = True,
    top_k: Optional[int] = None,
    temperature: float = 1.0,
    stop_at_block_size: bool = True,
    token_budget: Optional[int] = None,
):

    device = idx.device
    if termination_tokens is None:
        termination_tokens = torch.Tensor([model.config.vocab_size], device=device)

    torch.manual_seed(seed)

    budget = 0
    n, l = idx.shape
    has_termin_token = torch.zeros(n, device=device).bool()
    out_of_time = torch.zeros_like(has_termin_token).bool()
    start_age = age[:, [-1]]
    if max_age is None:
        assert max_time is not None
        max_age = start_age + max_time  # type: ignore
    else:
        assert max_time is None

    position_ids = model.position_ids(idx=idx)
    attention_mask = (idx > 0).long()
    past_key_values = None
    next_idx = idx
    next_age = age

    has_occurred = torch.zeros((n, model.config.vocab_size), device=device).int()
    has_occurred = has_occurred.scatter_(dim=1, index=idx, value=1)

    gen_idx_lst = []
    gen_age_lst = []
    gen_logits_lst = []
    active_pos = torch.arange(n).to(device)
    while True:
        terminated = torch.logical_or(has_termin_token, out_of_time)
        if terminated.all():
            break
        if l >= model.config.block_size and stop_at_block_size:
            break
        if (token_budget is not None) and (budget >= token_budget):
            break

        active_pos = active_pos[~terminated]
        next_idx = next_idx[~terminated]
        next_age = next_age[~terminated]
        position_ids = position_ids[~terminated]
        attention_mask = attention_mask[~terminated]
        has_occurred = has_occurred[~terminated]
        if isinstance(past_key_values, DynamicCache):
            kv_cache = DynamicCache()
            for i in range(len(past_key_values.layers)):
                if (
                    past_key_values.layers[i].keys is not None
                    and past_key_values.layers[i].values is not None
                ):
                    kv_cache.update(
                        past_key_values.layers[i].keys[~terminated],
                        past_key_values.layers[i].values[~terminated],
                        i,
                    )
            past_key_values = kv_cache

        logits, _, hf_output_dict = model(
            idx=next_idx,
            age=next_age,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        next_raw_logits = logits[:, [-1], :].clone()
        next_logits = logits[:, -1, :]
        next_logits[..., 0] = -torch.inf
        next_logits /= temperature
        if top_k is not None:
            next_logits = truncate_top_k(next_logits, top_k)
        if no_repeat:
            has_occurred[:, next_idx[:, -1]] = 1
            has_occurred[:, 1] = 0
            if hasattr(model, "time_tokens"):
                has_occurred[:, model.time_tokens] = 0
            next_logits[has_occurred.bool()] = -torch.inf
        next_idx, time_til_next = model.sample_next(next_logits)
        next_age = next_age[..., [-1]] + time_til_next
        past_key_values = hf_output_dict.past_key_values
        if isinstance(past_key_values, tuple):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        position_ids = position_ids[:, [-1]] + 1
        attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)),
            dim=1,
        )

        batch_next_idx = torch.zeros((n, 1), device=device).long()
        batch_next_age = torch.zeros_like(batch_next_idx).float()
        batch_next_logits = torch.zeros(
            (n, 1, model.config.vocab_size), device=device
        ).float()
        batch_next_idx[active_pos, :] = next_idx
        batch_next_age[active_pos, :] = next_age
        batch_next_logits[active_pos, ...] = next_raw_logits
        gen_idx_lst.append(batch_next_idx)
        gen_age_lst.append(batch_next_age)
        gen_logits_lst.append(batch_next_logits)

        has_termin_token = torch.isin(next_idx, termination_tokens).squeeze(1)
        out_of_time = (next_age >= max_age).squeeze(1)
        l += 1
        budget += 1

    idx = torch.cat(gen_idx_lst, dim=1)
    age = torch.cat(gen_age_lst, dim=1)
    logits = torch.cat(gen_logits_lst, dim=1)

    exceed_max_age = age > max_age
    is_termination_token = torch.isin(idx, termination_tokens)
    beyond_termination = (
        torch.cumsum(torch.cumsum(is_termination_token.int(), 1), 1) > 1
    )

    pad = exceed_max_age | beyond_termination
    idx[pad] = 0
    age[pad] = -10000

    return idx, age, logits
