from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from delphi.model.components import AgeEncoding, target_mask
from delphi.model.config import GPT2Config
from delphi.model.loss import CompetingExpHead, CrossEntropyHead
from delphi.model.transformer import (
    Block,
    LayerNorm,
    causal_attention_mask,
    count_params,
    initialize_weights,
)


@dataclass
class ModelConfig(GPT2Config):
    t_min: float = 1.0
    ce_beta: float = 1.0
    dt_beta: float = 1.0
    zero_inflate: bool = False
    zero_inflate_projector: str = "linear"


class Model(torch.nn.Module):
    model_type = "delphi"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.build_model(config)
        initialize_weights(self, config=config)
        n_params = count_params(self)
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def build_model(self, config: ModelConfig):

        assert config.vocab_size is not None
        self.token_embed = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.n_embd
        )
        self.age_embed = AgeEncoding(config=config)
        self.dropout = nn.Dropout(p=config.dropout)
        self.transformer_blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight
        self.ce_head = CrossEntropyHead(config)
        self.dt_head = CompetingExpHead(config)  # type: ignore

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        targets_age: Optional[torch.Tensor] = None,
    ):

        _, seq_len = idx.shape
        assert seq_len <= self.config.block_size
        token_emb = self.token_embed(idx)
        age_emb = self.age_embed(age.unsqueeze(-1))
        x = token_emb + age_emb

        attn_mask = causal_attention_mask(pad=(idx > 0))
        att = []
        for block in self.transformer_blocks:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.ln_f(x)
        att = torch.stack(att)

        if targets is not None:
            logits = self.lm_head(x)
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
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, att

    def eval_step(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
    ):

        _, seq_len = idx.shape
        assert seq_len <= self.config.block_size
        token_emb = self.token_embed(idx)
        age_emb = self.age_embed(age.unsqueeze(-1))
        x = token_emb + age_emb

        attn_mask = causal_attention_mask(pad=(idx > 0))
        att = []
        for block in self.transformer_blocks:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.ln_f(x)
        att = torch.stack(att)

        logits = self.lm_head(x)

        return logits, idx, age
