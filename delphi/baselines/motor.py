from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from delphi.model.components import AgeEncoding, target_mask
from delphi.model.config import GPT2Config
from delphi.model.loss import CrossEntropyHead
from delphi.model.transformer import (
    Block,
    LayerNorm,
    causal_attention_mask,
    count_params,
    initialize_weights,
)


@dataclass
class ModelConfig(GPT2Config):
    ce_beta: float = 0.0
    motor_beta: float = 1.0
    motor_n_hidden: int = 32
    motor_time_bins: list = field(default_factory=list)
    motor_task_tokens: list = field(default_factory=list)


class MotorHead(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.time_bins = torch.Tensor(self.config.motor_time_bins)
        self.register_buffer("task_tokens", torch.Tensor(self.config.motor_task_tokens))
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
        age: torch.Tensor,
        targets_age: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len = h.shape[0], h.shape[1]
        # h: [B, L, H]
        h = self.head[0](h)
        # h: [B, L, P, M]
        h = torch.reshape(
            h, (batch_size, seq_len, self.n_bins, self.config.motor_n_hidden)
        )
        # h: [B, L, P, V]
        log_lambda = self.head[1](h)

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

        return nll_loss


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
    age: torch.Tensor,
    targets_age: torch.Tensor,
) -> torch.Tensor:

    delta_t = targets_age.unsqueeze(-2) - age.unsqueeze(-1)  # [B, L, L]
    return delta_t  # [B, L, L]


class Model(torch.nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.build_model(config)
        initialize_weights(self, config=config)
        n_params = count_params(self)
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def build_model(self, config: ModelConfig):

        assert config.vocab_size is not None
        self.token_embed = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.n_embd
        )
        self.age_embed = AgeEncoding(config=config)
        self.dropout = nn.Dropout(p=config.dropout)
        self.transformer_blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight
        self.ce_head = CrossEntropyHead(config)
        self.motor_head = MotorHead(config)

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        targets_age: Optional[torch.Tensor] = None,
    ):

        batch_size, seq_len = idx.shape
        assert seq_len <= self.config.block_size
        token_emb = self.token_embed(idx)
        age_emb = self.age_embed(age.unsqueeze(-1))
        x = token_emb + age_emb

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

            assert targets_age is not None
            loss_motor = self.motor_head(
                h=x, age=age, targets=targets, targets_age=targets_age
            )

            loss = {
                "loss_ce": loss_ce * self.config.ce_beta,
                "loss_motor": loss_motor * self.config.motor_beta,
            }
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, att
