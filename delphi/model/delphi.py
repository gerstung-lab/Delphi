from dataclasses import dataclass
from typing import Optional

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from delphi.model import config
from delphi.model.components import AgeEncoding, target_mask
from delphi.model.loss import CompetingExpHead, CrossEntropyHead
from delphi.model.transformer import initialize_weights
from delphi.sampler import sample_competing_exponentials, truncate_top_k


@dataclass
class ModelConfig(config.GPT2Config):
    age_as_position: bool = True
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
        self.dt_head = CompetingExpHead(config)  # type: ignore

        initialize_weights(self, config=config)
        if self.config.age_as_position:
            self.pos_emb = AgeEncoding(config=config)
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
    ):

        _, seq_len = idx.shape
        assert seq_len <= self.config.block_size

        pos = self.position_ids(idx=idx)
        x = self.inputs_embeds(idx=idx, age=age)

        output_dict = self.gpt2(
            inputs_embeds=x, position_ids=pos, attention_mask=(idx > 0).long()
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

        return logits, loss

    @torch.no_grad()
    def next_token(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        no_repeat: bool = True,
    ):

        position_ids = self.position_ids(idx=idx)
        inputs_embeds = self.inputs_embeds(idx=idx, age=age)
        age_next = age
        past_key_values = None
        cache_position = None
        attention_mask = (idx > 0).long()

        while True:
            output_dict = self.gpt2(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                cache_position=cache_position,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

            logits = output_dict.logits
            raw_logits = logits[:, [-1], :].clone()
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = truncate_top_k(logits, top_k)
            if no_repeat:
                fill = idx + 0
                fill[fill == 1] = 0
                logits = logits.scatter_(1, fill, -torch.inf)
            idx_next, time_til_next = sample_competing_exponentials(logits)
            age_next = age_next[..., [-1]] + time_til_next

            yield idx_next, age_next, raw_logits

            past_key_values = output_dict.past_key_values
            position_ids = position_ids[:, [-1]] + 1
            inputs_embeds = self.inputs_embeds(idx=idx_next, age=age_next)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1), device=attention_mask.device
                    ),
                ),
                dim=1,
            )
