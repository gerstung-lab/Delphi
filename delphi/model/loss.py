import torch
import torch.nn as nn
import torch.nn.functional as F

from delphi.model.components import ZeroInflateProjector
from delphi.model.config import DelphiConfig


class CrosseEntropyHead(nn.Module):
    def __init__(self, config: DelphiConfig):
        super().__init__()
        self.config = config

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        logits = logits.permute(0, 2, 1)  # (b, l, n_vocab) -> (b, n_vocab, l)
        loss_ce = F.cross_entropy(logits, targets, ignore_index=-1, reduction="none")

        return loss_ce


class CompetingExpHead(nn.Module):
    def __init__(self, config: DelphiConfig):
        super().__init__()
        self.config = config

        if config.loss.zero_inflate:
            assert config.vocab_size is not None
            if config.loss.zero_inflate_projector == "linear":
                self.pi_head = nn.Linear(config.vocab_size, 1, bias=False)
            elif config.loss.zero_inflate_projector == "mlp":
                self.pi_head = ZeroInflateProjector(config)
            else:
                raise ValueError(
                    f"Unknown pi_projector: {config.loss.zero_inflate_projector}"
                )

    def forward(
        self,
        logits: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:

        lse = torch.logsumexp(logits, -1)
        lse = -torch.log(torch.exp(-lse) + self.config.t_min)
        ldt = -torch.log(delta_t + torch.tensor(self.config.t_min))
        exp_log_likelihood = lse - torch.exp(lse - ldt)

        if self.config.loss.zero_inflate:
            pi = self.pi_head(logits).squeeze()
            zero_case = -(F.softplus(-pi + lse) - F.softplus(-pi))
            nonzero_case = -(exp_log_likelihood - pi - F.softplus(-pi))
            loss_dt = zero_case * (delta_t == 0) + nonzero_case * (delta_t > 0)
        else:
            loss_dt = -exp_log_likelihood

        return loss_dt


class MotorHead(nn.Module):

    def __init__(self, config: DelphiConfig):
        super().__init__()
        self.config = config
        self.n_bins = config.loss.motor_n_bin
        self.time_horizon = config.loss.motor_time_horizon
        self.time_bins = torch.linspace(0, self.time_horizon, self.n_bins + 1)[:-1]

        assert config.vocab_size is not None
        self.V = config.vocab_size
        self.head = nn.ModuleList(
            [
                nn.Linear(
                    config.n_embd,
                    config.loss.motor_n_bin * config.loss.motor_n_hidden,
                    bias=False,
                ),
                nn.Linear(config.loss.motor_n_hidden, config.vocab_size, bias=False),
            ]
        )

    def forward(
        self,
        h: torch.Tensor,
        age: torch.Tensor,
        targets_age: torch.Tensor,
        delta_t: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        h = self.head[0](h)
        h = torch.reshape(
            h, (-1, -1, self.config.loss.motor_n_bin, self.config.loss.motor_n_hidden)
        )
        h = self.head[1](h)

        tte, no_event, _ = time_to_event(
            age=age, targets_age=targets_age, targets=targets, vocab_size=self.V
        )

        censor_masks = []
        for i in range(self.n_bins):
            is_censored = (tte >= self.time_bins[i]) * (tte < self.time_bins[i + 1])
            is_censored = torch.logical_or(is_censored, no_event)
            censor_masks.append(is_censored)
        censor_mask = torch.stack(censor_masks, dim=-2)  # [B, L, K, V]

        # compute loss
        # loss = loss_fn(censor_mask, h, tte)

        return h


def time_to_event(
    age: torch.Tensor,
    targets_age: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    notation:
        - i: participant index
        - j: position index (i.e. the j th event in trajectory)
        - k: event index
    returns:
        for any participant i,
        - tte:
            (i, j, k) -> time from the j th event to the next event k
        - no_event:
            (i, j, k) -> 1 if event k does not appear after (not including) the j th event else 0
        - token_index
            (i, j, k) -> n if the n th event is the next event k after (not including) the j th event
    """

    assert age.shape == targets_age.shape == targets.shape
    B = age.shape[0]
    L = age.shape[1]
    V = vocab_size

    # time_to_next_token: [B, L, L]
    time_to_next_token = broadcast_delta_t(age, targets_age)

    # token_index: [B, L, V]; targets: [B, L]
    # token_index: (i, j, k) -> y where y = argmin(targets[i][y] == k) s.t. (j < y) or -1
    targets_exp = targets.clone().view(-1, 1, L).expand(B, L, L)
    targets_exp = torch.triu(
        targets_exp, diagonal=0
    )  # invalidate retrospective targets
    L_idx = torch.arange(L, device=targets.device).view(1, L).expand(L, L)
    L_idx_exp = L_idx.unsqueeze(0).expand(B, L, L)
    # token_index[i][j1][targets[i][j1][j2]] = L_idx_exp[i][j1][j2]
    # -> token_index[i][j1][targets[i][j1][j2]] = j2
    token_index = torch.full((B, L, V), -1, device=targets.device)
    token_index = token_index.scatter_reduce(
        dim=-1, index=targets_exp, src=L_idx_exp, reduce="amin", include_self=False
    )
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
