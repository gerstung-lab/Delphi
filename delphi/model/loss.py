from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        logits = logits.permute(0, 2, 1)  # (b, l, n_vocab) -> (b, n_vocab, l)
        loss_ce = F.cross_entropy(logits, targets, reduction="none")

        return loss_ce


class CompetingExpHead(nn.Module):
    def __init__(self, n_embd: int, zero_inflate: bool, pi_head: Optional[str] = None):
        super().__init__()

        self.zero_inflate = zero_inflate
        if zero_inflate:
            assert pi_head is not None
            if pi_head == "linear":
                self.pi_head = nn.Linear(n_embd, 1, bias=False)
            elif pi_head == "mlp":
                self.pi_head = nn.ModuleList(
                    [
                        nn.Linear(n_embd, 32, bias=False),
                        nn.ReLU(),
                        nn.Linear(32, 1, bias=False),
                    ]
                )
            else:
                raise ValueError(f"Unknown pi_head: {pi_head}")

    def forward(
        self,
        logits: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:

        lse = torch.logsumexp(logits, -1)
        lse = -torch.log(torch.exp(-lse) + 1.0)
        ldt = -torch.log(delta_t + 1.0)
        exp_log_likelihood = lse - torch.exp(lse - ldt)

        if self.zero_inflate:
            pi = self.pi_head(logits).squeeze()
            zero_case = -(F.softplus(-pi + lse) - F.softplus(-pi))
            nonzero_case = -(exp_log_likelihood - pi - F.softplus(-pi))
            loss_dt = zero_case * (delta_t == 0) + nonzero_case * (delta_t > 0)
        else:
            loss_dt = -exp_log_likelihood

        return loss_dt
