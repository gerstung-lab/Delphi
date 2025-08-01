import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn import functional as F

from delphi.config import dataclass_from_dict
from delphi.model.components import (
    DelphiEmbedding,
    causal_attention_mask,
    target_mask,
    ties_adjusted_delta_t,
)
from delphi.model.config import DelphiConfig, GPT2Config
from delphi.model.loss import CompetingExpHead, CrossEntropyHead, MotorHead


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = False  # hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0

    def forward(self, x, attn_mask):
        B, T, C = x.size()
        # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = att.masked_fill(attn_mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        y, att = self.attn(self.ln_1(x), attn_mask)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att


def count_params(model: torch.nn.Module):
    n_params = sum(p.numel() for p in model.parameters())
    return n_params


def initialize_weights(model: torch.nn.Module, config: GPT2Config):

    def _init_weights(module: torch.nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    model.apply(_init_weights)
    # apply special scaled init to the residual projections, per GPT-2 paper
    for pn, p in model.named_parameters():
        if pn.endswith("c_proj.weight"):
            torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))


class Delphi(torch.nn.Module):

    def __init__(self, config: DelphiConfig):
        super().__init__()
        self.config = config
        self.build_model(config)
        initialize_weights(self, config=config)
        n_params = count_params(self)
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def build_model(self, config: DelphiConfig):

        self.transformer = nn.ModuleDict(
            dict(
                embed=DelphiEmbedding(config),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        assert config.vocab_size is not None
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.embed.token_embedding.weight = self.lm_head.weight

        self.ce_head = CrossEntropyHead(config)
        if config.loss.motor:
            self.dt_head = MotorHead(config)
        else:
            self.dt_head = CompetingExpHead(config)

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        modality: torch.Tensor,
        biomarker: Optional[dict[str, torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
        targets_age: Optional[torch.Tensor] = None,
        validation_loss_mode: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]], torch.Tensor]:

        x = self.transformer.embed(x0=idx, t0=age, M=modality, biomarker_x=biomarker)
        x = self.transformer.drop(x)

        attn_mask = causal_attention_mask(
            pad=(modality > 0), t1=targets_age, t0=age, mask_ties=self.config.mask_ties
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
                eps=0.0 if self.config.loss.zero_inflate else 1.0,
            )

            is_valid_target = target_mask(targets, ignore_tokens=ignored_tokens)
            loss_ce = self.ce_head(logits=logits_cp, targets=targets)
            loss_ce = torch.mean(loss_ce[is_valid_target])
            if self.config.loss.motor:
                loss_dt = self.dt_head(
                    h=x, age=age, targets_age=targets_age, targets=targets
                )
            else:
                loss_dt = self.dt_head(logits=logits, delta_t=dt)
                loss_dt = torch.mean(loss_dt[is_valid_target])

            loss = {
                "loss_ce": loss_ce * self.config.loss.ce_beta,
                "loss_dt": loss_dt * self.config.loss.dt_beta,
            }
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, :, :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, att


def load_model(
    ckpt_path,
    model_cls=Delphi,
    model_cfg_cls=DelphiConfig,
):

    ckpt_path = Path(ckpt_path)
    train_cfg = OmegaConf.load(ckpt_path / "config.yaml")
    ckpt_dict = torch.load(
        ckpt_path / "ckpt.pt",
        map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    )

    param_dtype = dict(
        float32=torch.float32,
        float64=torch.float64,
        float16=torch.float16,
        bfloat16=torch.bfloat16,
    )[train_cfg.dtype]
    model_cfg = dataclass_from_dict(model_cfg_cls, train_cfg.model, strict=False)
    model = model_cls(model_cfg)
    model.load_state_dict(ckpt_dict["model"])
    model = model.eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)

    return model, train_cfg
