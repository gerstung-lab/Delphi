from typing import Optional

import torch
from transformers import DynamicCache

from delphi import DAYS_PER_YEAR


def truncate_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    topk_value, _ = torch.topk(logits, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    logits[logits < min_value_top_k] = -torch.inf
    return logits


@torch.no_grad()
def legacy_generate(
    model,
    idx,
    age,
    max_new_tokens=100,
    max_age=85 * 365.25,
    no_repeat=True,
    termination_tokens=None,
    top_k=None,
):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.

    Selected parameters:
    --------------------

    termination_tokens: list[int] -  a list of tokens that indicate the and of the trajectory.
    Usually it is the "Death" token, but could be several tokens e.g. to indicate different
    death reasons.
    """
    if termination_tokens is None:
        import warnings

        warnings.warn(
            "When using a custem dataset, consider changing the `termination_tokens` argument."
        )
        termination_tokens = [1269]

    termination_tokens = torch.tensor(
        termination_tokens, dtype=torch.int64, device=idx.device
    )
    mask_time = -10000

    if max_new_tokens == -1:
        max_new_tokens = 128

    for _ in range(max_new_tokens):
        logits, _, _ = model(idx, age)
        logits = logits[:, -1, :]
        if hasattr(model.config, "ignore_tokens"):
            ignore_tokens = model.config.ignore_tokens
        else:
            ignore_tokens = [0]
        logits[:, ignore_tokens] = -torch.inf

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -torch.inf

        if no_repeat:
            fill = idx.clone()
            fill[fill == 1] = 0
            logits = logits.scatter_(1, fill, -torch.inf)

        # sample from exponential distributions for each disease using the inverse CDF method, then take min
        t_next = torch.clamp(
            -torch.exp(-logits) * torch.rand(logits.shape, device=idx.device).log(),
            min=0,
            max=365 * 80,
        ).min(1)
        idx_next = t_next[1][:, None]  # the index of the min sampled time
        age_next = (
            age[..., [-1]] + t_next[0][:, None]
        )  # the value of the min sampled time

        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        age = torch.cat((age, age_next), dim=1)

        if torch.logical_or(
            torch.isin(idx, termination_tokens).any(-1), age_next > max_age
        ).all():
            break

    pad = (
        torch.cumsum(
            torch.cumsum(torch.isin(idx, termination_tokens), 1).bool().int(), 1
        )
        > 1
    ) + (age > max_age)

    logits, _, _ = model(idx, age)
    idx[pad] = 0
    age[pad] = mask_time

    if no_repeat:
        fill = idx + 0
        fill[fill == 1] = 0
        logits = torch.stack(
            [
                logits[:, j].scatter_(1, fill[:, : j + 1], -torch.inf)
                for j in range(fill.shape[1])
            ]
        ).transpose(0, 1)

    return idx, age, logits


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    age: torch.Tensor,
    seed: int,
    max_age: Optional[float] = None,  # in days
    max_time: Optional[float] = None,
    termination_tokens: Optional[list | torch.Tensor] = None,
    no_repeat: bool = True,
    top_k: Optional[int] = None,
    temperature: float = 1.0,
    stop_at_block_size: bool = True,
    token_budget: Optional[int] = None,
):

    device = idx.device
    if termination_tokens is None:
        termination_tokens = torch.Tensor([model.config.vocab_size], device=device)
    else:
        termination_tokens = torch.tensor(termination_tokens).to(device)

    torch.manual_seed(seed)

    budget = 0
    n, l = idx.shape
    has_termin_token = torch.zeros(n, device=device).bool()
    out_of_time = torch.zeros_like(has_termin_token).bool()

    position_ids = model.position_ids(idx=idx)
    attention_mask = (idx > 0).long()
    past_key_values = None
    prompt_idx = idx.clone()
    prompt_age = age.clone()
    next_idx = idx
    next_age = age

    has_occurred = torch.zeros((n, model.config.vocab_size), device=device).int()
    has_occurred = has_occurred.scatter_(dim=1, index=idx, value=1)

    gen_idx_lst = []
    gen_age_lst = []
    gen_logits_lst = []
    active_pos = torch.arange(n).to(device)
    time_elapsed = torch.zeros((n,)).to(device)
    while True:

        logits, _, hf_output_dict = model(
            idx=next_idx,
            age=next_age,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        next_raw_logits = logits[:, -1, :].clone()
        if no_repeat:
            has_occurred = has_occurred.scatter_(index=next_idx, dim=1, value=1)
            has_occurred[:, 1] = 0
            if hasattr(model, "time_tokens"):
                has_occurred[:, model.time_tokens] = 0
            next_raw_logits[has_occurred.bool()] = -torch.inf
        next_logits = next_raw_logits.clone()
        next_logits[..., 0] = -torch.inf
        next_logits /= temperature
        if top_k is not None:
            next_logits = truncate_top_k(next_logits, top_k)
        next_idx, time_til_next = model.sample_next(
            logits=next_logits, output_dict=hf_output_dict
        )
        next_age = next_age[..., [-1]] + time_til_next
        time_elapsed += time_til_next.squeeze(-1)

        batch_next_idx = torch.zeros((n, 1), device=device).long()
        batch_next_age = torch.full_like(batch_next_idx, fill_value=-1e4).float()
        batch_next_logits = torch.full(
            (n, 1, model.config.vocab_size), fill_value=-torch.inf, device=device
        ).float()
        batch_next_idx[active_pos, :] = next_idx
        batch_next_age[active_pos, :] = next_age
        batch_next_logits[active_pos, ...] = next_raw_logits.unsqueeze(1)
        gen_idx_lst.append(batch_next_idx)
        gen_age_lst.append(batch_next_age)
        gen_logits_lst.append(batch_next_logits)

        out_of_time = torch.zeros_like(time_elapsed).bool()
        if max_time is not None:
            out_of_time = torch.logical_or(out_of_time, time_elapsed >= max_time)
        if max_age is not None:
            out_of_time = torch.logical_or(
                out_of_time, (next_age >= max_age).squeeze(1)
            )
        has_termin_token = torch.isin(next_idx, termination_tokens).squeeze(1)
        terminated = torch.logical_or(has_termin_token, out_of_time)
        if terminated.all():
            break
        l += 1
        if l >= model.config.block_size and stop_at_block_size:
            break
        budget += 1
        if (token_budget is not None) and (budget >= token_budget):
            break

        if l >= model.config.block_size:
            past_key_values = None
            next_idx = torch.cat((prompt_idx, *gen_idx_lst), dim=1)
            next_idx = next_idx[active_pos, -model.config.block_size :]
            next_age = torch.cat((prompt_age, *gen_age_lst), dim=1)
            next_age = next_age[active_pos, -model.config.block_size :]
            position_ids = model.position_ids(idx=next_idx)
            attention_mask = (next_idx > 0).long()
        else:
            past_key_values = hf_output_dict.past_key_values
            if isinstance(past_key_values, tuple):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            position_ids = position_ids[:, [-1]] + 1
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=device),
                ),
                dim=1,
            )

        active_pos = active_pos[~terminated]
        time_elapsed = time_elapsed[~terminated]
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
                        past_key_values.layers[i].keys[~terminated],  # type: ignore
                        past_key_values.layers[i].values[~terminated],  # type: ignore
                        i,
                    )
            past_key_values = kv_cache

    idx = torch.cat(gen_idx_lst, dim=1)
    age = torch.cat(gen_age_lst, dim=1)
    logits = torch.cat(gen_logits_lst, dim=1)

    return idx, age, logits
