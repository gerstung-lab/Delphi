import math
from typing import Optional

import torch
import torch.nn as nn


class AgeEncoding(nn.Module):

    def __init__(self, config, max_dim: int = 1024):
        super().__init__()
        div_term = torch.exp(
            torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd)
        )
        self.register_buffer("div_term", div_term)
        self.n_embd = config.n_embd
        self.linear = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(x / 365.25 * self.div_term)  # * (1-self.div_term)
        y[..., 1::2] = torch.cos(x / 365.25 * self.div_term)  # * (1-self.div_term)
        y = self.linear(y)

        # x = self.wae[:x.size(0)]
        return y  # self.dropout(x)


class DelphiEmbedding(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd, padding_idx=0
        )
        self.age_encoding = AgeEncoding(config)
        self.token_drop = nn.Dropout(config.token_dropout)

    def attention_mask(
        self,
        x0: torch.Tensor,
        t0: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        t1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size = x0.shape[0]
        seq_len = x0.shape[1]
        lower_tri_mask = torch.tril(torch.ones((seq_len, seq_len), device=x0.device))
        lower_tri_mask = lower_tri_mask.view(1, seq_len, seq_len)
        pad_mask = (x0 > 0).view(batch_size, 1, seq_len).to(torch.int)
        attn_mask = pad_mask * lower_tri_mask

        if self.config.mask_ties:
            if x1 is not None and t1 is not None:
                ties_mask = (
                    t1.view(batch_size, seq_len, 1) != t0.view(batch_size, 1, seq_len)
                ).to(torch.int)
                attn_mask *= ties_mask

        # attn_mask = torch.logical_or(
        #     attn_mask.to(torch.bool),
        #     torch.eye(seq_len, device=x0.device).unsqueeze(0).to(torch.bool)
        # )

        return attn_mask.unsqueeze(1)

    def forward(
        self,
        x0: torch.Tensor,
        t0: torch.Tensor,
    ) -> torch.Tensor:

        token_emb = self.token_embedding(x0)
        token_emb = self.token_drop(token_emb) * (1 - self.config.token_dropout)
        age_emb = self.age_encoding(t0.unsqueeze(-1))
        x = token_emb + age_emb

        return x


class ZeroTimeInflationPiProjector(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(config.vocab_size, 32, bias=False),
                nn.ReLU(),
                nn.Linear(32, 1, bias=False),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = l(x)
        return x
