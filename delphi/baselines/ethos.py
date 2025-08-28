from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import DynamicCache

from delphi.data.ukb import UKBDataset as BaseUKBDataset
from delphi.data.utils import update_tokenizer
from delphi.model.components import CrossEntropyHead, target_mask
from delphi.model.config import GPT2Config
from delphi.model.transformer import initialize_weights


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


class UKBDataset(BaseUKBDataset):

    def __init__(
        self,
        data_dir: str,
        subject_list: str,
        time_bins: list,
        no_event_interval: Optional[float] = None,
        block_size: Optional[int] = None,
        seed: int = 42,
        memmap: bool = False,
    ):

        super().__init__(
            data_dir=data_dir,
            subject_list=subject_list,
            no_event_interval=no_event_interval,
            block_size=block_size,
            seed=seed,
            memmap=memmap,
        )
        self.time_bins = np.array(time_bins)

        self.base_vocab_size = len(self.tokenizer)
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

        self.tokenizer, self.time_token_offset = update_tokenizer(
            base_tokenizer=self.tokenizer, add_tokenizer=time_tokenizer
        )

    def __getitem__(self, idx: int):

        pid = self.participants[idx]
        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l]
        t_pid = self.time_steps[i : i + l]
        x_pid, t_pid = self.add_no_event(x_pid, t_pid)
        x_pid, t_pid = create_ethos_sequence(
            x_pid, t_pid, time_bins=self.time_bins, offset=self.base_vocab_size - 1
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

    def set_time(self, time_tokens: list, time_intervals: list):
        time_tokens = torch.tensor(time_tokens)
        time_intervals = torch.tensor(time_intervals)
        token_to_time = torch.zeros((self.config.vocab_size,))
        token_to_time[time_tokens] = time_intervals
        self.register_buffer("time_tokens", time_tokens)
        self.register_buffer("token_to_time", token_to_time)

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
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[DynamicCache | tuple] = None,
    ):

        _, seq_len = idx.shape
        assert seq_len <= self.config.block_size

        if position_ids is None:
            position_ids = self.position_ids(idx)
        if attention_mask is None:
            attention_mask = (idx > 0).long()
        output_dict = self.gpt2(
            input_ids=idx,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        logits = output_dict["logits"]

        if targets is not None:
            loss_ce = self.ce_head(logits=logits, targets=targets)
            is_valid_target = target_mask(x1=targets, ignore_tokens=[])
            loss_ce = torch.mean(loss_ce[is_valid_target])
            loss = {"loss_ce": loss_ce}
        else:
            loss = None

        return logits, loss, output_dict

    def sample_next(self, logits: torch.Tensor, output_dict: dict):
        assert hasattr(self, "token_to_time")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        time_tile_next = self.token_to_time[idx_next]
        return idx_next, time_tile_next
