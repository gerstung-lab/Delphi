from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from delphi.data.ukb import UKBDataset
from delphi.model.components import target_mask
from delphi.model.config import GPT2Config
from delphi.model.loss import CrossEntropyHead
from delphi.model.transformer import initialize_weights
from delphi.sampler import truncate_top_k
from delphi.tokenizer import Tokenizer, update_tokenizer


def is_strictly_ascending(arr: list):
    return np.all(arr[1:] > arr[:-1])


def create_ethos_sequence(
    x: np.ndarray, t: np.ndarray, offset: int, time_bins: np.ndarray
):

    deltas = np.diff(t)
    same_time = deltas == 0
    deltas = np.digitize(deltas, bins=time_bins)
    deltas += offset
    deltas[same_time] = 0

    total_len = len(x) + len(deltas)
    event_pos = np.arange(0, total_len, step=2)
    time_pos = np.arange(1, total_len, step=2)
    pos = np.concatenate((event_pos, time_pos))
    idx = np.concatenate((x, deltas))
    age = np.concatenate((t, t[1:]))

    sort_by_pos = np.argsort(pos)
    idx = idx[sort_by_pos]
    age = age[sort_by_pos]

    reject = idx == 0
    idx = idx[~reject]
    age = age[~reject]

    return idx, age


class EthosUKBDataset(UKBDataset):

    def __init__(self, cfg, memmap, time_bins):

        super().__init__(cfg, memmap)
        self.time_bins = time_bins

        self.base_vocab_size = self.tokenizer.vocab_size
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
            base_tokenizer=self.tokenizer.to_dict(), add_tokenizer=time_tokenizer
        )
        self.tokenizer = Tokenizer(tokenizer)

    def __getitem__(self, idx: int):

        pid = self.participants[idx]
        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l]
        t_pid = self.time_steps[i : i + l]
        x_pid, t_pid = self.add_no_event(x_pid, t_pid)
        x_pid, t_pid = create_ethos_sequence(
            x_pid, t_pid, time_bins=self.time_bins, offset=self.base_vocab_size
        )
        x_pid, t_pid = self.crop_block_size(x_pid, t_pid)

        return x_pid, t_pid


class Model(torch.nn.Module):
    model_type = "ethos"

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
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
        self.ce_head = CrossEntropyHead(config)
        initialize_weights(self, config=config)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):

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
