from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache, GPT2Config, GPT2LMHeadModel

from delphi import DAYS_PER_YEAR
from delphi.model import config
from delphi.model.components import (
    AgeEncoding,
    CompetingExpHead,
    CrossEntropyHead,
    PiecewiseAgeEncoding,
    causal_attention_mask,
    target_mask,
    ties_adjusted_delta_t,
)
from delphi.model.transformer import Block, LayerNorm, initialize_weights


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


@dataclass
class ModelConfig(config.GPT2Config):
    age_as_position: bool = True
    max_wavelen: float = 10000.0
    absolute_position: bool = False
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
            if config.time_scale == "year-day":
                self.pos_emb = PiecewiseAgeEncoding(
                    n_embd=config.n_embd, max_wavelen=config.max_wavelen
                )
            else:
                if config.time_scale == "year":
                    # pd.to_timedelta does not support 'year' as a time unit
                    time_scale = (
                        DAYS_PER_YEAR * pd.to_timedelta(f"1 day").total_seconds()
                    )
                else:
                    time_scale = pd.to_timedelta(
                        f"1 {config.time_scale}"
                    ).total_seconds()
                norm_factor = (
                    time_scale / pd.to_timedelta(f"1 {config.interval}").total_seconds()
                )
                self.pos_emb = AgeEncoding(
                    n_embd=config.n_embd,
                    norm_factor=norm_factor,
                    max_wavelen=config.max_wavelen,
                )

        if not self.config.absolute_position:
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

    def sample_next(self, logits: torch.Tensor, output_dict: dict):
        if self.config.zero_inflate:
            pi = self.dt_head.pi_head(output_dict["logits"][:, -1, :])
            idx, time_til_next = sample_zero_inflated_exponentials(logits=logits, pi=pi)
        else:
            idx, time_til_next = sample_competing_exponentials(logits)
        return idx, time_til_next


@dataclass
class Delphi2MConfig(config.GPT2Config):
    dropout: float = 0.1
    token_dropout: float = 0.0
    ignore_tokens: list = field(default_factory=list)
    ce_beta: float = 1.0
    dt_beta: float = 1.0
    mask_ties: bool = True
    t_min: float = 1.0


class Delphi2M(torch.nn.Module):
    model_type = "delphi-2m"

    def __init__(self, config: Delphi2MConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wae=AgeEncoding(n_embd=config.n_embd, norm_factor=DAYS_PER_YEAR),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.ce_head = CrossEntropyHead(config)
        self.dt_head = CompetingExpHead(zero_inflate=False)

        initialize_weights(self, config=config)

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        targets_age: Optional[torch.Tensor] = None,
        validation_loss_mode: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]], torch.Tensor]:

        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age.unsqueeze(-1))

        x = tok_emb + age_emb
        x = self.transformer.drop(x)

        attn_mask = causal_attention_mask(
            pad=(idx == 0), t1=targets_age, t0=age, mask_ties=self.config.mask_ties
        )

        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        if (targets is not None) and (targets_age is not None):
            logits = self.lm_head(x)

            logits_cp = logits.clone()
            ignored_tokens = self.config.ignore_tokens.copy()
            if validation_loss_mode:
                ignored_tokens += [1]
                logits_cp[..., ignored_tokens] = -torch.inf

            dt = ties_adjusted_delta_t(
                t0=age,
                t1=targets_age,
                attn_mask=attn_mask,
                mask_ties=self.config.mask_ties,
                eps=self.config.t_min,
            )

            is_valid_target = target_mask(targets, ignore_tokens=ignored_tokens)
            loss_ce = self.ce_head(logits=logits_cp, targets=targets)
            loss_ce = torch.mean(loss_ce[is_valid_target])
            loss_dt = self.dt_head(logits=logits, delta_t=dt)
            loss_dt = torch.mean(loss_dt[is_valid_target])

            loss = {
                "loss_ce": loss_ce * self.config.ce_beta,
                "loss_dt": loss_dt * self.config.dt_beta,
            }
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, :, :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, att


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
