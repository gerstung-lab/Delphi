from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
import yaml

from delphi import DAYS_PER_YEAR
from delphi.model.components import AgeEncoding, CrossEntropyHead
from delphi.model.config import GPT2Config, parse_token_list
from delphi.model.transformer import (
    initialize_weights,
)


def estimate_pieces(
    X: np.ndarray, T: np.ndarray, task_tokens: list, vocab_size: int, n_pieces: int
):

    targets = torch.from_numpy(X[:, 1:])
    age, targets_age = torch.from_numpy(T[:, :-1]), torch.from_numpy(T[:, 1:])

    tte, _, _ = time_to_event(
        age=age,
        targets_age=targets_age,
        targets=targets,
        task_tokens=torch.Tensor(task_tokens),
        vocab_size=vocab_size,
    )

    pieces = np.percentile(tte.ravel(), np.linspace(0, 100, n_pieces + 1))
    pieces[0] = 0
    pieces[-1] = float("inf")

    return pieces


def homogenize_piecewise_lambda(
    log_lambda: torch.Tensor, piece_edges: torch.Tensor, horizon: float
) -> torch.Tensor:

    right_edges = piece_edges[:-1]
    too_far = right_edges > horizon
    piece_edges = piece_edges.clamp(None, horizon)
    piece_interval = torch.diff(piece_edges)
    piece_interval[too_far] = 0
    weighted_avg_lamba = (
        torch.exp(log_lambda) * piece_interval.view(1, 1, -1, 1) / piece_interval.sum()
    ).sum(dim=-2, keepdim=False)
    eps = 1e-6

    return torch.log(weighted_avg_lamba + eps)


@dataclass
class ModelConfig(GPT2Config):
    time_scale: str = "year"  # year or day
    interval: str = "day"  # day or mi
    ce_beta: float = 0.0
    motor_beta: float = 1.0
    motor_n_hidden: int = 32
    motor_pieces: Optional[list] = None
    motor_task_tokens: list = field(default_factory=list)


class MotorHead(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        with open(self.config.motor_pieces, "r") as f:
            motor_pieces = yaml.safe_load(f)
        self.register_buffer("time_bins", torch.tensor(motor_pieces))
        if self.config.motor_task_tokens == "all":
            task_tokens = np.arange(1, self.config.vocab_size)
        else:
            task_tokens = parse_token_list(self.config.motor_task_tokens)
        self.register_buffer("task_tokens", torch.tensor(task_tokens))
        self.n_bins = len(self.time_bins) - 1

        assert config.vocab_size is not None
        self.V = self.task_tokens.shape[0]
        self.head = nn.ModuleList(
            [
                nn.Linear(
                    config.n_embd,
                    self.n_bins * config.motor_n_hidden,
                    bias=False,
                ),
                nn.Linear(config.motor_n_hidden, self.V, bias=False),
            ]
        )

    def forward(
        self,
        h: torch.Tensor,
        age: Optional[torch.Tensor] = None,
        targets_age: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:

        batch_size, seq_len = h.shape[0], h.shape[1]
        # h: [B, L, H]
        h = self.head[0](h)
        # h: [B, L, P, M]
        h = torch.reshape(
            h, (batch_size, seq_len, self.n_bins, self.config.motor_n_hidden)
        )
        # h: [B, L, P, V]
        log_lambda = self.head[1](h)

        if targets is not None:
            assert age is not None
            assert targets_age is not None
            # tte: [B, L, V]
            tte, no_future_event, _ = time_to_event(
                age=age,
                targets_age=targets_age,
                targets=targets,
                task_tokens=self.task_tokens,
                vocab_size=self.config.vocab_size,  # type: ignore
            )

            # last_censor: [B, L]
            last_censor = targets_age[:, -1].view(-1, 1) - age

            surv_or_event_time, occur_in_piece = piecewise_time(
                tte=tte,
                no_future_event=no_future_event,
                last_censor=last_censor,
                time_bins=self.time_bins,
            )

            ll_survival = (-torch.exp(log_lambda) * surv_or_event_time).mean()
            ll_to_event = (log_lambda * occur_in_piece.float()).mean()
            nll_loss = -(ll_survival + ll_to_event)

        else:
            nll_loss = None

        return nll_loss, log_lambda


def piecewise_time(
    tte: torch.Tensor,
    no_future_event: torch.Tensor,
    last_censor: torch.Tensor,
    time_bins: torch.Tensor,
):

    occur_mask_per_bin = []
    surv_or_to_event_time_per_bin = []
    for i in range(len(time_bins) - 1):
        start, end = time_bins[i], time_bins[i + 1]
        # occur_in_bin: [B, L, V]
        occur_in_bin = (
            (tte > start).bool() & (tte <= end).bool() & ~no_future_event.bool()
        )
        # surv_or_event_time_in_bin: [B, L, V]
        surv_or_event_time_in_bin = torch.where(
            occur_in_bin.bool(),
            input=tte,
            other=torch.clamp(last_censor.unsqueeze(-1), min=0, max=float(end - start)),
        )
        occur_mask_per_bin.append(occur_in_bin.bool())
        surv_or_to_event_time_per_bin.append(surv_or_event_time_in_bin)
    # occur_in_piece: [B, L, P, V]
    occur_in_piece = torch.stack(occur_mask_per_bin, dim=-2)
    surv_or_event_time = torch.stack(surv_or_to_event_time_per_bin, dim=-2)

    return surv_or_event_time, occur_in_piece


def time_to_event(
    age: torch.Tensor,
    targets_age: torch.Tensor,
    targets: torch.Tensor,
    task_tokens: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    notation:
        - i: participant index
        - j: the j th event in trajectory
        - k: the k th task token in the sorted order
    returns:
        for any participant i,
        - tte:
            (i, j, k) -> time from j to the next k
        - no_event:
            (i, j, k) -> 1 if k does not appear after (not including) j else 0
        - token_index
            (i, j, k) -> n if [the n th event] == k after (not including) j
    """

    assert age.shape == targets_age.shape == targets.shape
    B = age.shape[0]
    L = age.shape[1]
    V = task_tokens.shape[0]
    device = targets.device

    # time_to_next_token: [B, L, L]
    time_to_next_token = broadcast_delta_t(age, targets_age)

    targets_clone = targets.clone()
    targets_exp = targets_clone.view(-1, 1, L).expand(B, L, L)
    targets_exp = torch.triu(
        targets_exp, diagonal=0
    )  # invalidate retrospective targets

    token_map = torch.arange(vocab_size, device=device)
    token_map[~torch.isin(token_map, task_tokens)] = 0  # invalidate non-task targets
    token_map[token_map > 0] = torch.arange(1, V + 1, device=device)
    targets_exp = token_map[targets_exp]

    L_idx = torch.arange(L, device=device).view(1, L).expand(L, L)
    L_idx_exp = L_idx.unsqueeze(0).expand(B, L, L)

    # token_index: [B, L, V+1]
    # token_index: (i, j, k) -> y where y = argmin(targets[i][y] == the k-1 th task token) s.t. (j < y) or -1
    token_index = torch.full((B, L, V + 1), -1, device=device)
    # token_index[i][j1][targets[i][j1][j2]] = L_idx_exp[i][j1][j2]
    # -> token_index[i][j1][targets[i][j1][j2]] = j2
    token_index = token_index.scatter_reduce(
        dim=-1, index=targets_exp, src=L_idx_exp, reduce="amin", include_self=False
    )
    # remove the 0th position in the V dimension
    token_index = token_index[:, :, 1:]
    no_event = token_index == -1

    # tte: [B, L, V]
    # tte: (i, j, k) -> delta_t (if event happens) or time_to_final_token (if no event)
    token_index[token_index == -1] = L - 1
    tte = torch.gather(input=time_to_next_token, index=token_index, dim=-1)

    return tte, no_event, token_index


def broadcast_delta_t(
    age: torch.Tensor, targets_age: torch.Tensor, eps: float = 1.0
) -> torch.Tensor:

    delta_t = targets_age.unsqueeze(-2) - age.unsqueeze(-1)  # [B, L, L]
    return torch.clamp(delta_t, min=eps)  # [B, L, L]


class Model(torch.nn.Module):
    model_type = "motor"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.time_scale == "year":
            # pd.to_timedelta does not support 'year' as a time unit
            time_scale = DAYS_PER_YEAR * pd.to_timedelta(f"1 day").total_seconds()
        else:
            time_scale = pd.to_timedelta(f"1 {config.time_scale}").total_seconds()
        norm_factor = (
            time_scale / pd.to_timedelta(f"1 {config.interval}").total_seconds()
        )
        self.age_embed = AgeEncoding(n_embd=config.n_embd, norm_factor=norm_factor)

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
        self.gpt2 = transformers.GPT2Model(gpt2_config)
        self.token_embed = self.gpt2.wte
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight

        self.ce_head = CrossEntropyHead(config)
        self.motor_head = MotorHead(config)

        initialize_weights(self, config=config)
        self.gpt2.wpe.weight.data *= 0
        for param in self.gpt2.wpe.parameters():
            param.requires_grad = False

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
        output_dict = self.gpt2(inputs_embeds=x, attention_mask=attn_mask)
        x = output_dict["last_hidden_state"]

        loss_motor, log_lambda = self.motor_head(h=x, age=age, targets=targets, targets_age=targets_age)

        if targets is not None:
            assert targets_age is not None
            loss = {"loss_motor": loss_motor}
        else:
            loss = None

        return log_lambda, loss, output_dict

