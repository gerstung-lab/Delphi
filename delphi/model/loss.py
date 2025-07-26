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
