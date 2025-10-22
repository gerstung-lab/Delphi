import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

from delphi.exponential import (
    sample_competing_exponentials,
    sample_zero_inflated_exponentials,
)


def causal_attention_mask(
    pad: torch.Tensor,
    mask_ties: bool = False,
    t0: None | torch.Tensor = None,
    t1: None | torch.Tensor = None,
) -> torch.Tensor:

    b, l = pad.shape
    device = pad.device

    lower_tri_mask = torch.tril(torch.ones((l, l), device=device))
    lower_tri_mask = lower_tri_mask.view(1, l, l)
    pad_mask = pad.view(b, 1, l).to(torch.int)
    attn_mask = pad_mask * lower_tri_mask

    if mask_ties:
        assert t0 is not None
        if t1 is not None:
            ties_mask = (t1.view(b, l, 1) != t0.view(b, 1, l)).to(torch.int)
            attn_mask *= ties_mask

    attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(
        torch.ones(l, device=device)
    ) > 0

    return attn_mask.unsqueeze(1)


def ties_adjusted_delta_t(
    t0: torch.Tensor,
    t1: torch.Tensor,
    mask_ties: bool,
    attn_mask: torch.Tensor | None = None,
    eps: float = 1.0,
) -> torch.Tensor:

    delta_t = t1 - t0
    delta_t = torch.clamp(delta_t, min=eps)

    if mask_ties:
        assert attn_mask is not None
        delta_t = torch.gather(
            delta_t,
            -1,
            (
                attn_mask
                * torch.arange(
                    0, t0.size(1), device=t0.device, dtype=torch.float32
                ).view(1, 1, 1, -1)
            )
            .max(-1)
            .indices.squeeze((1, 2)),
        )

    return delta_t


class AgeEncoding(nn.Module):

    def __init__(
        self, n_embd: int, norm_factor: float = 365.25, max_wavelen: float = 10000.0
    ):
        super().__init__()
        div_term = torch.exp(
            torch.arange(0, n_embd, 2) * (-math.log(max_wavelen) / n_embd)
        )
        self.register_buffer("div_term", div_term)
        self.n_embd = n_embd
        self.linear = torch.nn.Linear(n_embd, n_embd, bias=False)

        self.norm_factor = norm_factor

    def forward(self, x: torch.Tensor):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        time_years = x / self.norm_factor
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(time_years * self.div_term)  # * (1-self.div_term)
        y[..., 1::2] = torch.cos(time_years * self.div_term)  # * (1-self.div_term)
        y = self.linear(y)

        return y


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x, attn_mask):
        B, T, C = x.size()
        # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        y, att = self.attn(self.ln_1(x), attn_mask)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att


@dataclass
class Delphi2MConfig:
    # defaults to config of the OG delphi-2m ckpt
    block_size: int = 48
    vocab_size: int = 1270
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    dropout: float = 0.0
    token_dropout: float = 0.0
    t_min: float = 0.1
    bias: bool = False
    mask_ties: bool = True
    ignore_tokens: None | list = field(
        default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )  # 0 always ignored
    zero_inflate: bool = False
    no_event_rate: None | float = None
    mask_no_event_attention: bool = False


class Delphi2M(nn.Module):
    """
    slightly cleaned up version of delphi-2m with extra features:
        - zero inflation
        - fix no-event rate as a model parameter
        - mask attention to previous no-event tokens
    """

    model_type = "delphi-2m"

    def __init__(self, config: Delphi2MConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wae=AgeEncoding(n_embd=config.n_embd),
                token_drop=nn.Dropout(config.token_dropout),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        if config.zero_inflate:
            assert not config.mask_ties, "mask_ties must be False for zero inflation"
            self.pi_head = nn.Sequential(
                nn.Linear(config.n_embd, 32, bias=False),
                nn.ReLU(),
                nn.Linear(32, 1, bias=False),
            )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx, age, targets=None, targets_age=None, validation_loss_mode=False
    ):
        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age.unsqueeze(-1))
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + age_emb
        x = self.transformer.drop(x)

        pad = idx > 0
        if self.config.mask_no_event_attention:
            pad = idx > 1
        attn_mask = causal_attention_mask(
            pad=pad, mask_ties=self.config.mask_ties, t0=age, t1=targets_age
        )

        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        logits = self.lm_head(x)
        if self.config.no_event_rate is not None:
            logits[..., 1] = math.log(self.config.no_event_rate)
        output = {"logits": logits}
        if self.config.zero_inflate:
            pi = self.pi_head(x).squeeze(-1)
            output["pi"] = pi

        if targets is not None:
            assert targets_age is not None

            ignored_tokens = [0]
            if self.config.ignore_tokens is not None:
                ignored_tokens += self.config.ignore_tokens.copy()
            if validation_loss_mode:
                ignored_tokens += [1]
                logits[..., ignored_tokens] = -torch.inf
            targets = targets.reshape(-1)
            pass_tokens = targets != -1
            for k in ignored_tokens:  # and gender
                pass_tokens *= targets != k

            loss_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1))[pass_tokens],
                targets[pass_tokens],
                ignore_index=-1,
            )

            lse = torch.logsumexp(logits, -1)
            lse = -torch.log(torch.exp(-lse) + self.config.t_min)
            dt = ties_adjusted_delta_t(
                t0=age,
                t1=targets_age,
                attn_mask=attn_mask,
                mask_ties=self.config.mask_ties,
                eps=self.config.t_min,
            ).view(-1)
            ldt = -torch.log(dt + self.config.t_min)

            if not self.config.zero_inflate:
                loss_dt = -(
                    lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1))
                )  ## Exponential log-likelihood (real statistics, TM)
            else:
                log_likelihood = lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt)
                lse = lse.view(-1)
                zero_case_nll = -(
                    F.softplus(-pi.view(-1) + lse) - F.softplus(-pi.view(-1))
                )
                nonzero_case_nll = -(
                    log_likelihood - pi.view(-1) - F.softplus(-pi.view(-1))
                )
                loss_dt = (
                    zero_case_nll * (dt == 0).float()
                    + nonzero_case_nll * (dt > 0).float()
                )
            loss_dt = torch.mean(loss_dt[pass_tokens])

            loss = {"loss_ce": loss_ce, "loss_dt": loss_dt}

        else:
            loss = None

        return output, loss, att

    @torch.no_grad()
    def generate(
        self,
        idx,
        age,
        max_new_tokens=100,
        max_age=85 * 365.25,
        no_repeat=True,
        termination_tokens=None,
        top_k=None,
        stop_at_block_size: bool = True,
    ):
        if termination_tokens is None:
            import warnings

            warnings.warn(
                "When using a custem dataset, consider changing the `termination_tokens` argument."
            )
            termination_tokens = [1269]

        termination_tokens = torch.tensor(
            termination_tokens, dtype=torch.int64, device=idx.device
        )
        mask_time = -10000

        if max_new_tokens == -1:
            max_new_tokens = 128

        for _ in range(max_new_tokens):
            output, _, _ = self.forward(idx, age)
            logits = output["logits"]
            logits = logits[:, -1, :]
            ignore_tokens = [0]
            if (
                hasattr(self.config, "ignore_tokens")
                and self.config.ignore_tokens is not None
            ):
                ignore_tokens += self.config.ignore_tokens
            logits[:, ignore_tokens] = -torch.inf

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -torch.inf

            if no_repeat:
                fill = idx.clone()
                fill[fill == 1] = 0
                logits = logits.scatter_(1, fill, -torch.inf)

            if not self.config.zero_inflate:
                idx_next, time_til_next = sample_competing_exponentials(logits=logits)
            else:
                idx_next, time_til_next = sample_zero_inflated_exponentials(
                    logits=logits, pi=output["pi"]
                )

            age_next = age[..., [-1]] + time_til_next

            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)

            if torch.logical_or(
                torch.isin(idx, termination_tokens).any(-1), age_next > max_age
            ).all():
                break

            if (idx.shape[1] > self.config.block_size) and stop_at_block_size:
                break

        pad = (
            torch.cumsum(
                torch.cumsum(torch.isin(idx, termination_tokens), 1).bool().int(), 1
            )
            > 1
        ) + (age > max_age)

        outputs, _, _ = self.forward(idx, age)
        logits = outputs["logits"]
        idx[pad] = 0
        age[pad] = mask_time

        if no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = torch.stack(
                [
                    logits[:, j].scatter_(1, fill[:, : j + 1], -torch.inf)
                    for j in range(fill.shape[1])
                ]
            ).transpose(0, 1)

        return idx, age, logits
