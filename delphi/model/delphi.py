from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from transformers import DynamicCache, GPT2Config, GPT2LMHeadModel

from delphi import DAYS_PER_YEAR
from delphi.exponential import (
    sample_competing_exponentials,
    sample_zero_inflated_exponentials,
)
from delphi.model import config
from delphi.model.components import (
    AgeEncoding,
    CompetingExpHead,
    CrossEntropyHead,
    PiecewiseAgeEncoding,
    Time2Vec,
    target_mask,
)
from delphi.model.transformer import initialize_weights


@dataclass
class ModelConfig(config.GPT2Config):
    encode_time: str = "sin"
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
        if config.encode_time is not None:
            if config.time_scale == "year":
                # pd.to_timedelta does not support 'year' as a time unit
                time_scale = DAYS_PER_YEAR * pd.to_timedelta(f"1 day").total_seconds()
            else:
                time_scale = pd.to_timedelta(f"1 {config.time_scale}").total_seconds()
            norm_factor = (
                time_scale / pd.to_timedelta(f"1 {config.interval}").total_seconds()
            )

            if config.encode_time == "sin":
                self.pos_emb = AgeEncoding(
                    n_embd=config.n_embd,
                    norm_factor=norm_factor,
                    max_wavelen=config.max_wavelen,
                )
            elif config.encode_time == "piecewise":
                self.pos_emb = PiecewiseAgeEncoding(
                    n_embd=config.n_embd,
                    max_wavelen=config.max_wavelen,
                    norm_factors=[norm_factor, 1.0],
                )
            elif config.encode_time == "time2vec":
                self.pos_emb = Time2Vec(n_embd=config.n_embd, norm_factor=norm_factor)
            else:
                raise ValueError(f"unknown time encoding: {config.encode_time}")

        if not self.config.absolute_position:
            self.gpt2.transformer.wpe.weight.data *= 0
            for param in self.gpt2.transformer.wpe.parameters():
                param.requires_grad = False

    def inputs_embeds(self, idx: torch.Tensor, age: torch.Tensor):

        token_emb = self.token_embed(idx)
        if hasattr(self, "pos_emb"):
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
