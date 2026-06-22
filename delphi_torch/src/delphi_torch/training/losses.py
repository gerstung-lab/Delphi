from __future__ import annotations

import torch
from torch.nn import functional as F

from delphi_torch.config import ModelConfig


def compute_losses(
    logits: torch.Tensor,
    *,
    idx: torch.Tensor,
    age: torch.Tensor,
    targets: torch.Tensor,
    targets_age: torch.Tensor,
    attn_mask: torch.Tensor,
    cfg: ModelConfig,
    validation: bool = False,
) -> dict[str, torch.Tensor]:
    ignored_tokens = list(cfg.ignore_tokens)
    if validation:
        ignored_tokens.append(1)
        logits = logits.clone()
        # Use large negative instead of -inf to avoid numerical issues
        logits[..., ignored_tokens] = -1e9

    targets_flat = targets.reshape(-1)
    pass_tokens = targets_flat != -1
    for k in ignored_tokens:
        pass_tokens &= targets_flat != k

    if pass_tokens.any():
        loss_ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1))[pass_tokens],
            targets_flat[pass_tokens],
            ignore_index=-1,
        )
    else:
        loss_ce = torch.tensor(0.0, device=logits.device)

    lse = torch.logsumexp(logits, -1)
    lse = -torch.log(torch.exp(-lse) + cfg.t_min)
    
    # Clamp dt to reasonable range (max ~100 years in days)
    # This prevents numerical explosion when targets_age or age have masked values
    dt = torch.clamp(targets_age - age, min=1.0, max=36525.0)

    if cfg.mask_ties:
        seq_len = idx.size(1)
        arange = torch.arange(0, seq_len, device=idx.device, dtype=torch.float32).view(1, 1, 1, -1)
        dt = torch.gather(dt, -1, (attn_mask * arange).max(-1).indices.squeeze((1, 2)))

    ldt = -torch.log(dt + cfg.t_min).view(-1)
    
    # Exponential log-likelihood: loss = -(lse - exp(lse - ldt))
    # Clamp the exponent to prevent numerical explosion
    # With lse bounded by t_min (~2.3) and dt bounded above, this should rarely trigger
    diff = lse.reshape(-1) - ldt.reshape(-1)
    exp_term = torch.exp(torch.clamp(diff, max=20.0))
    loss_dt = -(lse.reshape(-1) - exp_term)

    if pass_tokens.any():
        loss_dt = loss_dt[pass_tokens].mean()
    else:
        loss_dt = torch.tensor(0.0, device=logits.device)

    return {"loss_ce": loss_ce, "loss_dt": loss_dt, "loss": loss_ce + loss_dt}
