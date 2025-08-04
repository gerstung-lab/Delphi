from dataclasses import dataclass, field
from functools import partial
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn

from delphi.data.core import (
    BaseDataConfig,
    collate_batch_data,
    collate_batch_time,
    load_core_data_package,
)
from delphi.data.transform import add_no_event, crop_contiguous, sort_by_time
from delphi.experiment import BaseTrainer
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
from delphi.tokenizer import Tokenizer, update_tokenizer


@dataclass
class DataConfig(BaseDataConfig):
    n_time_tokens: int = 10
    time_bins: list = field(default_factory=list)


def is_strictly_ascending(arr: np.ndarray):
    return np.all(arr[1:] > arr[:-1])


def create_ethos_sequence(
    X: np.ndarray, T: np.ndarray, offset: int, time_bins: np.ndarray
):

    T, X = sort_by_time(T, X)

    is_pad = T[:, :-1] == -1e4
    deltas = T[:, 1:] - T[:, :-1]
    same_time = deltas == 0
    deltas = np.digitize(deltas, bins=time_bins)
    deltas += offset
    deltas[is_pad | same_time] = 0

    B = X.shape[0]
    L = X.shape[1] + deltas.shape[1]

    event_pos = np.arange(0, L, step=2)
    event_pos = np.broadcast_to(event_pos, (B, event_pos.size))
    is_event = np.ones_like(event_pos)

    time_pos = np.arange(1, L, step=2)
    time_pos = np.tile(time_pos, (B, 1))
    time_pos[is_pad | same_time] = -1
    is_time = np.zeros_like(time_pos)

    pos = np.hstack((event_pos, time_pos))
    tokens = np.hstack((X, deltas))
    is_event = np.hstack([is_event, is_time]).astype(bool)

    _, tokens, is_event = sort_by_time(pos, tokens, is_event)

    return tokens, is_event


def _estimate_time_bins(sample_t: np.ndarray, n_tokens: int):

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


class EthosDataset:

    def __init__(self, cfg: DataConfig):

        self.cfg = cfg

        (
            base_tokenizer,
            self.start_pos,
            self.seq_len,
            self.participants,
            self.tokens,
            self.time_steps,
            self.rng,
        ) = load_core_data_package(cfg=cfg, memmap=True)

        if len(cfg.time_bins) == 0:
            assert cfg.n_time_tokens is not None
            print(f"\t- time bins not defined; estimating time bins...")
            self.time_bins = self.estimate_time_bins(n_tokens=cfg.n_time_tokens)
        else:
            print(f"\t- time bins found!")
            self.time_bins = np.array(cfg.time_bins)
            assert is_strictly_ascending(self.time_bins)
        print(f"\t- time bins:")
        for i in range(len(self.time_bins) - 1):
            print(f"\t\t- {self.time_bins[i]} – {self.time_bins[i+1]}")

        n_bins = len(self.time_bins)
        time_tokenizer = dict()
        for i in range(n_bins):
            start = self.time_bins[i]
            token = i + 1
            if i < n_bins - 1:
                end = self.time_bins[i + 1]
                time_tokenizer[f"time-{start}-{end}"] = token
            else:
                time_tokenizer[f"time-{start}-inf"] = token

        tokenizer, self.time_token_offset = update_tokenizer(
            base_tokenizer=base_tokenizer, add_tokenizer=time_tokenizer
        )
        self.tokenizer = Tokenizer(tokenizer)

        if cfg.no_event_interval is not None:
            self.add_no_event = partial(
                add_no_event,
                mode="random",
                interval=cfg.no_event_interval,
                token=self.tokenizer["no_event"],
                rng=self.rng,
            )
        else:
            self.add_no_event = lambda *args: args
        self.ethos_sequence = partial(
            create_ethos_sequence,
            offset=self.time_token_offset,
            time_bins=self.time_bins,
        )
        if cfg.block_size is not None:
            self.crop_block_size = partial(
                crop_contiguous, block_size=cfg.block_size, rng=self.rng
            )
        else:
            self.crop_block_size = lambda *args: args

    def __len__(self):
        return self.participants.size

    def estimate_time_bins(self, n_tokens: int, sample_size: int = 100000):

        print(f"\t- estimating based on {sample_size} samples...")

        pids = self.participants[
            self.rng.permutation(np.arange(len(self)))[np.arange(sample_size)]
        ]

        sample = list()
        for pid in pids:
            i = self.start_pos[pid]
            l = self.seq_len[pid]
            sample.extend(range(i, i + l))
        sample = np.array(sample)
        sample_t = self.time_steps[sample].astype(np.int32)

        return _estimate_time_bins(sample_t=sample_t, n_tokens=n_tokens)

    def get_batch(self, batch_idx: np.ndarray):

        P = self.participants[batch_idx]
        X = []
        T = []
        for i, pid in enumerate(P):
            i = self.start_pos[pid]
            l = self.seq_len[pid]
            x_pid = self.tokens[i : i + l]
            t_pid = self.time_steps[i : i + l]
            X.append(x_pid)
            T.append(t_pid)

        X = collate_batch_data(X)
        T = collate_batch_time(T)

        X, T = self.add_no_event(X, T)
        X = self.ethos_sequence(X=X, T=T)
        X = self.crop_block_size(X)

        return X


def build_datasets(train_cfg_dict: dict, val_cfg_dict: dict):

    train_cfg = DataConfig(**train_cfg_dict)
    val_cfg = DataConfig(**val_cfg_dict)

    train_ds = EthosDataset(train_cfg)

    val_cfg.no_event_interval = train_cfg.no_event_interval
    val_cfg.block_size = train_cfg.block_size
    val_cfg.time_bins = train_ds.time_bins.tolist()

    val_ds = EthosDataset(val_cfg)

    return train_ds, val_ds


def load_sequences(
    it: Iterator,
    dataset: EthosDataset,
) -> Iterator:

    for idx in it:

        X = dataset.get_batch(idx)
        X = torch.tensor(X, dtype=torch.long)

        yield X


class Model(torch.nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.build_model(config)
        initialize_weights(self, config=config)
        n_params = count_params(self)
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def build_model(self, config: GPT2Config):

        assert config.vocab_size is not None
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

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):

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
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss_ce = self.ce_head(logits=logits, targets=targets)
            is_valid_target = target_mask(x1=targets, ignore_tokens=[])
            loss_ce = torch.mean(loss_ce[is_valid_target])
            loss = {"loss_ce": loss_ce}
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def eval_step(self, idx: torch.Tensor, age: torch.Tensor, time_bins: np.ndarray):

        raw_idx = idx.clone()
        raw_seq_len = idx.shape[1]
        assert self.config.vocab_size is not None
        max_event_token = self.config.vocab_size - len(time_bins) - 1

        ethos_idx, is_event = create_ethos_sequence(
            X=idx.numpy(), T=age.numpy(), time_bins=time_bins, offset=max_event_token
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


class Trainer(BaseTrainer):

    def mini_step(self, loader: Iterator) -> dict[str, torch.Tensor]:

        X = next(loader)
        X = X.to(self.device)
        X_t0, X_t1 = X[:, :-1], X[:, 1:]
        with self.ctx:
            _, loss = self.model(idx=X_t0, targets=X_t1)

        return loss
