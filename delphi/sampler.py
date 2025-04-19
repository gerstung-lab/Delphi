from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from delphi import DAYS_PER_YEAR
from delphi.eval import clock
from delphi.model.transformer import Delphi
from delphi.tokenizer import Tokenizer


def truncate_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    topk_value, _ = torch.topk(logits, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    logits[logits < min_value_top_k] = -torch.inf
    return logits


def sample_competing_exponentials(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    t_next = torch.clamp(
        -torch.exp(-logits) * torch.rand(logits.shape, device=logits.device).log(),
        min=0,
        max=DAYS_PER_YEAR * 80.0,
    ).min(1)
    next_token = t_next[1][:, None]
    time_til_next = t_next[0][:, None]

    return next_token, time_til_next


def sample_comorbid_based_on_cutoff(
    logits: torch.Tensor,
    comorbid_cutoff: float,
    always_single_tokens: list,
) -> tuple[torch.Tensor, torch.Tensor]:

    single_token, time_til_single_token = sample_competing_exponentials(logits)

    logits[..., always_single_tokens] = -torch.inf
    probs = F.softmax(logits, dim=-1)
    n_abv_cutoff = (probs >= comorbid_cutoff).sum(-1)
    max_comorbid = max(int(n_abv_cutoff.max().item()), 1)
    if max_comorbid == 1:
        return single_token, time_til_single_token

    has_comorbid = n_abv_cutoff > 1
    comorbid_tokens = torch.zeros(
        logits.shape[0], max_comorbid, device=logits.device, dtype=torch.long
    )
    topk_probs, topk_tokens = torch.topk(probs, k=max_comorbid, dim=-1)
    topk_logits = torch.gather(logits, index=topk_tokens, dim=-1)
    is_comorbid = torch.logical_and(
        has_comorbid.unsqueeze(-1), topk_probs >= comorbid_cutoff
    )
    comorbid_tokens[is_comorbid] = topk_tokens[is_comorbid]
    comorbid_tokens[~has_comorbid, 0] = single_token[~has_comorbid].squeeze()
    _, time_til_comorbid_tokens = sample_competing_exponentials(topk_logits)
    time_til_tokens = time_til_comorbid_tokens.repeat(1, max_comorbid)
    time_til_tokens[~has_comorbid, 0] = time_til_single_token[~has_comorbid].squeeze()
    time_til_tokens[comorbid_tokens == 0] = -10000

    return comorbid_tokens, time_til_tokens


@dataclass
class CausalSamplerConfig:
    seed: int = 42
    no_repeat: bool = True
    top_k: Optional[int] = None
    temperature: float = 1.0
    max_age_in_years: float = 80
    max_new_tokens: int = 64
    max_total_tokens: int = 128
    termination_tokens: list[str] = field(default_factory=list)
    simulate_comorbid: bool = True
    comorbid_cutoff: float = 0.2
    always_single_tokens: list[str] = field(default_factory=list)


class CausalSampler:

    def __init__(self, cfg: CausalSamplerConfig, model: Delphi, tokenizer: Tokenizer):

        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

    @torch.no_grad()
    def next_token(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        logits, _, _ = self.model(idx, age)

        logits = logits[:, -1, :] / self.cfg.temperature

        logits[:, self.model.config.ignore_tokens] = -torch.inf

        if self.cfg.top_k is not None:
            logits = truncate_top_k(logits, self.cfg.top_k)

        if self.cfg.no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = logits.scatter_(1, fill, -torch.inf)

        if self.cfg.simulate_comorbid:
            always_single_tokens = [
                self.tokenizer[disease] for disease in self.cfg.always_single_tokens
            ]
            idx_next, time_til_next = sample_comorbid_based_on_cutoff(
                logits=logits,
                comorbid_cutoff=self.cfg.comorbid_cutoff,
                always_single_tokens=always_single_tokens,
            )
        else:
            idx_next, time_til_next = sample_competing_exponentials(logits)

        return idx_next, time_til_next

    @clock
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        termination_tokens = (
            torch.Tensor(
                [self.tokenizer[token] for token in self.cfg.termination_tokens]
            )
            .to(idx.device)
            .long()
        )

        max_age = self.cfg.max_age_in_years * DAYS_PER_YEAR
        n_new = 0
        n_total = 0
        while True:

            idx_next, time_til_next = self.next_token(
                idx=idx,
                age=age,
            )
            age_next = age[..., [-1]] + time_til_next
            age_next[time_til_next == -10000] = -10000

            n_new += idx_next.shape[1]
            too_many_new = n_new > self.cfg.max_new_tokens
            n_total = idx_next.shape[1] + idx.shape[1]
            too_many_total = n_total > self.cfg.max_total_tokens
            if too_many_new or too_many_total:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)

            sort_by_time = age.argsort(1)
            idx = idx.gather(1, sort_by_time)
            age = age.gather(1, sort_by_time)

            is_termination_token = torch.isin(idx, termination_tokens)
            if torch.logical_or(
                is_termination_token.any(-1), (age_next > max_age).any(-1)
            ).all():
                break

        exceed_max_age = age > max_age
        # mask all tokens after the *second* occurrence of a termination token
        beyond_termination = (
            torch.cumsum(torch.cumsum(is_termination_token.int(), 1), 1) > 1
        )

        pad = exceed_max_age | beyond_termination
        idx[pad] = 0
        age[pad] = -10000

        logits, _, _ = self.model(idx, age)

        if self.cfg.no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = torch.stack(
                [
                    logits[:, j].scatter_(1, fill[:, : j + 1], float("NaN"))
                    for j in range(fill.shape[1])
                ]
            ).transpose(0, 1)

        return idx, age, logits
