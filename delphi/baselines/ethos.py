from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from delphi.model.components import target_mask
from delphi.model.config import GPT2Config
from delphi.model.loss import CrossEntropyHead
from delphi.model.transformer import (
    Block,
    LayerNorm,
    causal_attention_mask,
    count_params,
    initialize_weights,
)
from delphi.tokenizer import Tokenizer


def is_strictly_ascending(arr: list):
    return np.all(arr[1:] > arr[:-1])


def create_ethos_sequence(
    X: torch.Tensor, T: torch.Tensor, offset: int, time_bins: torch.Tensor
):

    is_pad = T[:, :-1] == -1e4
    deltas = torch.diff(T, dim=1)
    same_time = deltas == 0
    deltas = torch.bucketize(deltas, boundaries=time_bins, right=True)
    deltas += offset
    deltas[is_pad | same_time] = 0

    B = X.shape[0]
    L = X.shape[1] + deltas.shape[1]
    device = time_bins.device

    event_pos = torch.arange(0, L, step=2, device=device)
    event_pos = torch.broadcast_to(event_pos, (B, event_pos.shape[0]))
    is_event = torch.ones_like(event_pos, device=device)

    time_pos = torch.arange(1, L, step=2, device=device)
    time_pos = torch.tile(time_pos, (B, 1))
    time_pos[is_pad | same_time] = -1
    is_time = torch.zeros_like(time_pos)

    pos = torch.hstack((event_pos, time_pos))
    tokens = torch.hstack((X, deltas))
    is_event = torch.hstack([is_event, is_time]).bool()

    _, sorted_idx = torch.sort(pos, dim=1)
    tokens = torch.take_along_dim(tokens, indices=sorted_idx, dim=1)
    is_event = torch.take_along_dim(is_event, indices=sorted_idx, dim=1)

    return tokens, is_event


def estimate_time_bins(sample_t: np.ndarray, n_tokens: int):

    delta = np.diff(sample_t)
    delta = delta[delta > 0]

    percentiles = np.linspace(0, 100, num=n_tokens + 1)
    percentiles = percentiles[:-1]

    time_bins = np.round(np.percentile(delta, q=percentiles), decimals=1)
    n_uniq = len(np.unique(time_bins))
    assert n_uniq == len(time_bins), "too many time tokens!"

    return time_bins


def parse_time_bins(tokenizer: Tokenizer):

    time_bins = list()
    token_map = tokenizer.to_dict()
    for token_key, token_val in token_map.items():
        token_key = str(token_key)
        if "time" in token_key:
            start = float(token_key.split("-")[1])
            end = float(token_key.split("-")[2])
            time_bins.append(start)

    return np.array(time_bins)


@dataclass
class ModelConfig(GPT2Config):
    base_vocab_size: int = 1270
    n_time_tokens: int = 10
    time_bins: list = field(default_factory=list)


class Model(torch.nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.build_model(config)
        initialize_weights(self, config=config)
        n_params = count_params(self)
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def build_model(self, config: ModelConfig):

        self.token_embed = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.n_embd
        )
        self.pos_embed = nn.Embedding(
            num_embeddings=config.block_size, embedding_dim=config.n_embd
        )
        self.dropout = nn.Dropout(p=config.dropout)
        self.transformer_blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight
        self.ce_head = CrossEntropyHead(config)

        assert is_strictly_ascending(config.time_bins)
        self.register_buffer("time_bins", torch.Tensor(config.time_bins))

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        targets_age: Optional[torch.Tensor] = None,
    ):

        if targets is not None:
            assert targets_age is not None
            all_idx = torch.hstack((idx, targets[:, [-1]]))
            all_age = torch.hstack((age, targets_age[:, [-1]]))
            ethos_idx, _ = create_ethos_sequence(
                X=all_idx,
                T=all_age,
                offset=self.config.base_vocab_size - 1,
                time_bins=self.time_bins,
            )
            idx, targets = ethos_idx[:, :-1], ethos_idx[:, 1:]
        else:
            idx, _ = create_ethos_sequence(
                X=idx,
                T=age,
                offset=self.config.base_vocab_size - 1,
                time_bins=self.time_bins,
            )

        batch_size, seq_len = idx.shape
        assert seq_len <= self.config.block_size
        x = self.token_embed(idx)

        pos = torch.arange(seq_len, device=idx.device).view(1, -1).repeat(batch_size, 1)

        is_pad = idx == 0
        offset = is_pad.sum(dim=1, keepdim=True)
        pos = torch.clamp(pos - offset, min=0)
        x += self.pos_embed(pos)

        attn_mask = causal_attention_mask(pad=(idx > 0))

        att = []
        for block in self.transformer_blocks:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.ln_f(x)
        att = torch.stack(att)

        if targets is not None:
            logits = self.lm_head(x)
            loss_ce = self.ce_head(logits=logits, targets=targets)
            is_valid_target = target_mask(x1=targets, ignore_tokens=[])
            loss_ce = torch.mean(loss_ce[is_valid_target])
            loss = {"loss_ce": loss_ce}
        else:
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, att

    def eval_step(self, idx: torch.Tensor, age: torch.Tensor):

        raw_idx = idx.clone()
        raw_seq_len = idx.shape[1]

        ethos_idx, is_event = create_ethos_sequence(
            X=idx,
            T=age,
            time_bins=self.time_bins,
            offset=self.config.base_vocab_size - 1,
        )
        idx = torch.tensor(ethos_idx).to(idx.device)

        batch_size, seq_len = idx.shape
        assert seq_len <= self.config.block_size
        x = self.token_embed(idx)

        pos = torch.arange(seq_len, device=idx.device).view(1, -1).repeat(batch_size, 1)

        is_pad = idx == 0
        offset = is_pad.sum(dim=1, keepdim=True)
        pos = torch.clamp(pos - offset, min=0)
        x += self.pos_embed(pos)

        attn_mask = causal_attention_mask(pad=(idx > 0))

        att = []
        for block in self.transformer_blocks:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.ln_f(x)
        att = torch.stack(att)

        logits = self.lm_head(x)

        logits = logits[is_event, :]
        logits = logits.reshape(batch_size, raw_seq_len, -1)

        return logits, raw_idx, age
