from dataclasses import dataclass
from typing import Optional

import torch
import transformers

from delphi.model.components import AgeEncoding, target_mask
from delphi.model.config import GPT2Config
from delphi.model.loss import CompetingExpHead, CrossEntropyHead
from delphi.model.transformer import (
    count_params,
    initialize_weights,
)
from delphi.sampler import sample_competing_exponentials, truncate_top_k


@dataclass
class ModelConfig(GPT2Config):
    ce_beta: float = 1.0
    dt_beta: float = 1.0
    zero_inflate: bool = False
    zero_inflate_projector: str = "linear"


class Model(torch.nn.Module):
    model_type = "delphi"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.age_embed = AgeEncoding(config=config)

        gpt2_config = transformers.GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.block_size,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            resid_pdrop=config.resid_pdrop,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
        )
        self.gpt2 = transformers.GPT2LMHeadModel(gpt2_config)
        self.token_embed = self.gpt2.transformer.wte

        self.ce_head = CrossEntropyHead(config)
        self.dt_head = CompetingExpHead(config)  # type: ignore

        initialize_weights(self, config=config)
        self.gpt2.transformer.wpe.weight.data *= 0
        for param in self.gpt2.transformer.wpe.parameters():
            param.requires_grad = False

        n_params = count_params(self)
        print("number of parameters: %.2fM" % (n_params / 1e6,))

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

        attn_mask = idx > 0
        output_dict = self.gpt2(inputs_embeds=x, attention_mask=attn_mask.long())
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

        logits, _ = self.forward(idx, age)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            logits = truncate_top_k(logits, top_k)

        if no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = logits.scatter_(1, fill, -torch.inf)

        idx_next, time_til_next = sample_competing_exponentials(logits)
        age_next = age[..., [-1]] + time_til_next

        idx = torch.cat((idx, idx_next), dim=1)
        age = torch.cat((age, age_next), dim=1)

        logits = self.forward(idx=idx, age=age)

        return idx, age
