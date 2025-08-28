from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from transformers import DynamicCache, GPT2Config, GPT2LMHeadModel

from delphi import DAYS_PER_YEAR
from delphi.model import config
from delphi.model.components import (
    AgeEncoding,
    CompetingExpHead,
    CrossEntropyHead,
    target_mask,
)
from delphi.model.transformer import initialize_weights


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

    return next_token, time_til_nex


@dataclass
class ModelConfig(config.GPT2Config):
    age_as_position: bool = True
    time_scale: str = "year"  # year or day
    interval: str = "day"  # day or min
    ce_beta: float = 1.0
    dt_beta: float = 1.0
    zero_inflate: bool = False
    zero_inflate_projector: str = "linear"


class Model(torch.nn.Module):
    model_type = "delphi"

    def __init__(self, config: ModelConfig):

        super().__init__()
        self.config = config

        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.block_size,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            resid_pdrop=config.resid_pdrop,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
        )
        self.gpt2 = GPT2LMHeadModel(gpt2_config)
        self.token_embed = self.gpt2.transformer.wte

        self.ce_head = CrossEntropyHead(config)
        self.dt_head = CompetingExpHead(
            n_input=config.vocab_size,
            zero_inflate=config.zero_inflate,
            pi_head=config.zero_inflate_projector,
        )

        initialize_weights(self, config=config)
        if self.config.age_as_position:
            if config.time_scale == "year":
                # pd.to_timedelta does not support 'year' as a time unit
                time_scale = DAYS_PER_YEAR * pd.to_timedelta(f"1 day").total_seconds()
            else:
                time_scale = pd.to_timedelta(f"1 {config.time_scale}").total_seconds()
            norm_factor = (
                time_scale / pd.to_timedelta(f"1 {config.interval}").total_seconds()
            )
            self.pos_emb = AgeEncoding(n_embd=config.n_embd, norm_factor=norm_factor)
            self.gpt2.transformer.wpe.weight.data *= 0
            for param in self.gpt2.transformer.wpe.parameters():
                param.requires_grad = False

    def inputs_embeds(self, idx: torch.Tensor, age: torch.Tensor):

        token_emb = self.token_embed(idx)
        if self.config.age_as_position:
            age_emb = self.pos_emb(age.unsqueeze(-1))
            x = token_emb + age_emb
        else:
            x = token_emb

        return x

    @staticmethod
    def position_ids(idx: torch.Tensor):

        batch_size, seq_len = idx.shape
        pos = torch.arange(seq_len, device=idx.device).view(1, -1).repeat(batch_size, 1)
        is_pad = idx == 0
        offset = is_pad.sum(dim=1, keepdim=True)
        pos = torch.clamp(pos - offset, min=0)

        return pos

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        targets_age: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[DynamicCache | tuple] = None,
    ):

        _, seq_len = idx.shape
        assert seq_len <= self.config.block_size

        x = self.inputs_embeds(idx=idx, age=age)
        if position_ids is None:
            position_ids = self.position_ids(idx)
        if attention_mask is None:
            attention_mask = (idx > 0).long()

        output_dict = self.gpt2(
            inputs_embeds=x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        logits = output_dict.logits

        if targets is not None:
            is_valid_target = target_mask(x1=targets, ignore_tokens=[])

            loss_ce = self.ce_head(logits=logits, targets=targets)
            loss_ce = torch.mean(loss_ce[is_valid_target])

            assert targets_age is not None
            loss_dt = self.dt_head(logits=logits, delta_t=targets_age - age)
            loss_dt = torch.mean(loss_dt[is_valid_target])

            loss = {
                "loss_ce": loss_ce * self.config.ce_beta,
                "loss_dt": loss_dt * self.config.dt_beta,
            }
        else:
            loss = None

        return logits, loss, output_dict

    def sample_next(self, logits: torch.Tensor):
        idx, time_til_next = sample_competing_exponentials(logits)
        return idx, time_til_next


def integrate_risk(
    log_lambda: torch.Tensor, age: torch.Tensor, start: float, end: float | torch.Tensor
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
    B, L = age.shape
    _, _, V = log_lambda.shape
    age = age.unsqueeze(-1).broadcast_to((B, L, V))
    pad = age[:, [-1], :]
    if not isinstance(end, torch.Tensor):
        end = torch.full_like(pad, fill_value=end)
    else:
        assert end.shape == (B, V)
        end = end.unsqueeze(1)
    age = torch.cat([age, torch.maximum(pad, end)], dim=1)

    start = torch.full_like(end, fill_value=start)
    t_clamped = age.clamp(start, end)
    dt = t_clamped.diff(1, dim=1)
    dt_norm = dt / (dt.sum(1, keepdim=True) + 1e-6) * (end - start)

    risk = log_lambda.exp() * dt_norm
    risk = risk.sum(-2)

    return risk
