from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from delphi_torch.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, T, C) means: batch size, sequence length, embedding dimension (d_model)
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        # (B, T, d_model) -> (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        # transpose to (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # scaled dot-product attention: softmax(Q·Kᵀ / √head_dim) · V
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        # att matrix: (B, n_head, T, T)
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = att @ v
        # transpose to (B, T, n_head, head_dim) and flatten last two dimensions
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # (B, T, d_model) -> (B, T, d_model)
        y = self.resid_dropout(self.c_proj(y))
        # y: (B, T, d_model)
        # att: (B, n_head, T, T)
        return y, att


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # From the original Delphi paper:
        # "The MLP has a dimension of 4 * d_model and uses GELU activation."
        dim_feedforward = 4 * cfg.d_model
        self.ln_1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, dim_feedforward, bias=cfg.bias),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim_feedforward, cfg.d_model, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y, att = self.attn(self.ln_1(x), attn_mask)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att


class AgeEncoding(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        div_term = torch.exp(torch.arange(0, cfg.d_model, 2) * (-math.log(10000.0) / cfg.d_model))
        self.register_buffer("div_term", div_term)
        self.d_model = cfg.d_model
        self.linear = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(x.shape[0], x.shape[1], self.d_model, device=x.device)
        y[..., 0::2] = torch.sin(x / 365.25 * self.div_term)
        y[..., 1::2] = torch.cos(x / 365.25 * self.div_term)
        return self.linear(y)


def build_attention_mask(
    idx: torch.Tensor,
    age: torch.Tensor,
    targets_age: torch.Tensor | None,
    *,
    mask_ties: bool,
) -> torch.Tensor:
    device = idx.device
    bsz, seq_len = idx.size()

    # Use float tensors throughout for ONNX compatibility
    # (ONNX doesn't support boolean arithmetic)
    idx_valid = (idx > 0).float()
    attn_mask = idx_valid.view(bsz, 1, 1, seq_len) * idx_valid.view(bsz, 1, seq_len, 1)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device))[None, None, :, :]
    attn_mask = attn_mask * causal

    # Use torch.eye instead of torch.diag for ONNX compatibility
    eye = torch.eye(seq_len, device=device)

    if targets_age is not None and mask_ties:
        tie_mask = (age.view(bsz, 1, 1, seq_len) != targets_age.view(bsz, 1, seq_len, 1)).float()
        attn_mask = attn_mask * tie_mask
        # Add self-attention for positions with no attention
        no_attn = (attn_mask.sum(-1, keepdim=True) == 0).float()
        attn_mask = attn_mask + no_attn * eye

    # Add self-attention for padding positions
    idx_pad = (idx == 0).float().view(bsz, 1, 1, seq_len)
    attn_mask = attn_mask + idx_pad * eye
    attn_mask = attn_mask * causal
    
    # Convert back to boolean for masked_fill compatibility
    return attn_mask > 0


class DelphiModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Simple token embedding - the original Delphi approach
        # Hierarchical embeddings were removed: added complexity without proven benefit,
        # and created weight-tying asymmetry with lm_head
        wte = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Age encoding: sinusoidal encoding is sufficient for multi-scale temporal patterns
        wae = AgeEncoding(cfg)

        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                wae=wae,
                token_drop=nn.Dropout(cfg.token_dropout),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.d_model, bias=cfg.bias),
            )
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # Weight tying: use token embedding weights for output projection
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        scaled_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("mlp.2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=scaled_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        *,
        targets_age: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age.unsqueeze(-1))
        x = self.transformer.token_drop(tok_emb) * (1 - self.cfg.token_dropout)
        x = x + age_emb
        x = self.transformer.drop(x)

        attn_mask = build_attention_mask(idx, age, targets_age, mask_ties=self.cfg.mask_ties)

        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if return_attn:
            return logits, attn_mask, torch.stack(att)
        return logits, attn_mask, None
