from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from delphi.model.components import target_mask
from delphi.model.config import GPT2Config
from delphi.model.loss import CrossEntropyHead
from delphi.model.transformer import (
    initialize_weights,
)
from delphi.sampler import truncate_top_k
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
    model_type = "ethos"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        gpt2_config = transformers.GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.block_size,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
        )
        self.gpt2 = transformers.GPT2LMHeadModel(gpt2_config)
        self.token_embed = self.gpt2.transformer.wte

        self.ce_head = CrossEntropyHead(config)

        initialize_weights(self, config=config)
        self.gpt2.transformer.wpe.weight.data *= 0
        for param in self.gpt2.transformer.wpe.parameters():
            param.requires_grad = False
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

        pos = torch.arange(seq_len, device=idx.device).view(1, -1).repeat(batch_size, 1)
        is_pad = idx == 0
        offset = is_pad.sum(dim=1, keepdim=True)
        pos = torch.clamp(pos - offset, min=0)

        attn_mask = idx > 0
        output_dict = self.gpt2(
            input_ids=idx, attention_mask=attn_mask, position_ids=pos
        )
        logits = output_dict["logits"]

        if targets is not None:
            loss_ce = self.ce_head(logits=logits, targets=targets)
            is_valid_target = target_mask(x1=targets, ignore_tokens=[])
            loss_ce = torch.mean(loss_ce[is_valid_target])
            loss = {"loss_ce": loss_ce}
        else:
            loss = None

        return logits, loss

    def eval_step(self, idx: torch.Tensor, age: torch.Tensor):

        raw_idx = idx.clone()
        batch_size, raw_seq_len = idx.shape

        ethos_idx, is_event = create_ethos_sequence(
            X=idx,
            T=age,
            time_bins=self.time_bins,
            offset=self.config.base_vocab_size - 1,
        )
        idx = torch.tensor(ethos_idx).to(idx.device)

        logits = self.forward_backbone(idx=idx)

        logits = logits[is_event, :]
        logits = logits.reshape(batch_size, raw_seq_len, -1)

        return logits, raw_idx, age

    @torch.no_grad
    def next_token(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        no_repeat: bool = False,
    ):

        max_event_token = self.config.base_vocab_size - 1

        logits, _ = self.forward(idx, age)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            logits = truncate_top_k(logits, top_k)

        if no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = logits.scatter_(1, fill, -torch.inf)
        probs = F.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1)
        next_is_time = tokens > max_event_token
        next_is_event = ~next_is_time
        next_event = tokens.clone()
        next_event[next_is_time] = 0
        idx = torch.concat((idx, next_event), dim=-1)

        last_age = age[:, [-1]]
        next_time = tokens.clone()
        next_time -= max_event_token
        next_time = torch.clamp(next_time, min=0)
        # select left edge of each time bin
        next_time = self.time_bins[next_time - 1]
        next_time[next_is_event] = 0
        next_age = next_time + last_age
        age = torch.concat((age, next_age), dim=-1)

        return idx, age
