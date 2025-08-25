import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from delphi.model.config import BiomarkerEmbedConfig, DelphiConfig
from delphi.multimodal import Modality, module_name


class AgeEncoding(nn.Module):

    def __init__(self, n_embd: int, norm_factor: float):
        super().__init__()
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
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


class BiomarkerEmbedding(nn.Module):

    def __init__(self, config: BiomarkerEmbedConfig, n_embed: int) -> None:

        super().__init__()
        self.config = config
        assert config.input_size is not None, "input_size must be specified"
        if config.projector == "linear":
            self.projector = nn.Linear(config.input_size, n_embed, bias=False)
        elif config.projector == "mlp":
            layers = []
            assert (
                config.n_layers is not None
            ), "n_layers must be specified for mlp projector"
            assert (
                config.n_hidden is not None
            ), "n_hidden must be specified for mlp projector"
            for i in range(config.n_layers):
                in_size = config.input_size if i == 0 else config.n_hidden
                out_size = n_embed if i == config.n_layers - 1 else config.n_hidden
                layers.append(nn.Linear(in_size, out_size, bias=False))
                if i < config.n_layers - 1:
                    layers.append(nn.ReLU())
            self.projector = nn.Sequential(*layers)
        else:
            raise ValueError(f"unknown projector type: {config.projector}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.projector == "embed":
            x = x[x != 0]  # remove padding
            return self.projector(x)

        return self.projector(x)


class DelphiEmbedding(nn.Module):

    def __init__(self, config: DelphiConfig) -> None:
        super().__init__()
        self.config = config
        assert config.vocab_size is not None
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd, padding_idx=0
        )
        self.age_encoding = AgeEncoding(n_embd=config.n_embd)
        self.token_drop = nn.Dropout(config.token_dropout)

        self.biomarker_embed = nn.ModuleDict()
        biomarker_modalities = []
        for biomarker, biomarker_cfg in config.biomarkers.items():
            bm_key = module_name(Modality[biomarker.upper()])
            self.biomarker_embed[bm_key] = BiomarkerEmbedding(
                config=biomarker_cfg, n_embed=config.n_embd
            )
            biomarker_modalities.append(Modality[biomarker.upper()])

        if config.modality_emb:
            max_modality_idx = (
                max([modality.value for modality in biomarker_modalities])
                if len(biomarker_modalities) > 0
                else 1
            )
            self.mod_embedding = nn.Embedding(
                max_modality_idx + 1, config.n_embd, padding_idx=0
            )

    def forward(
        self,
        x0: torch.Tensor,
        t0: torch.Tensor,
        M: torch.Tensor,
        biomarker_x: dict[Modality, torch.Tensor] = {},
    ) -> torch.Tensor:

        token_emb = self.token_embedding(x0)

        token_emb = self.token_drop(token_emb) * (1 - self.config.token_dropout)
        age_emb = self.age_encoding(t0.unsqueeze(-1))

        for modality in biomarker_x.keys():
            m_pos = torch.nonzero(M == modality.value)  # N * 2
            if m_pos.size == 0:
                continue
            m_emb = self.biomarker_embed[module_name(modality)](
                biomarker_x[modality]
            )  # N * H
            assert m_emb.shape[0] == m_pos.shape[0]

            token_emb[m_pos[:, 0], m_pos[:, 1], :] *= 0
            token_emb[m_pos[:, 0], m_pos[:, 1], :] += m_emb

        x = token_emb + age_emb

        if self.config.modality_emb:
            mod_emb = self.mod_embedding(M)
            x += mod_emb

        return x


class CrossEntropyHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        logits = logits.permute(0, 2, 1)  # (b, l, n_vocab) -> (b, n_vocab, l)
        loss_ce = F.cross_entropy(logits, targets, reduction="none")

        return loss_ce


class CompetingExpHead(nn.Module):
    def __init__(self, n_input: int, zero_inflate: bool, pi_head: Optional[str] = None):
        super().__init__()

        self.zero_inflate = zero_inflate
        if zero_inflate:
            assert pi_head is not None
            if pi_head == "linear":
                self.pi_head = nn.Linear(n_input, 1, bias=False)
            elif pi_head == "mlp":
                self.pi_head = nn.ModuleList(
                    [
                        nn.Linear(n_input, 32, bias=False),
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
        t_min = 0.0 if self.zero_inflate else 1.0
        delta_t = torch.clamp(delta_t, min=t_min)
        ldt = -torch.log(delta_t + 1.0)
        exp_log_likelihood = lse - torch.exp(lse - ldt)

        if self.zero_inflate:
            pi = self.pi_head(logits).squeeze()
            pi = F.sigmoid(pi)
            loss_dt = -exp_log_likelihood * (1 - pi) + pi * (delta_t == 0).float()
        else:
            loss_dt = -exp_log_likelihood

        return loss_dt


def causal_attention_mask(
    pad: torch.Tensor,
    mask_ties: bool = False,
    t0: Optional[torch.Tensor] = None,
    t1: Optional[torch.Tensor] = None,
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


def target_mask(
    x1: torch.Tensor,
    ignore_tokens: list[int],
) -> torch.Tensor:

    is_valid_target = x1 != 0
    for k in ignore_tokens:
        is_valid_target *= x1 != k

    return is_valid_target


def ties_adjusted_delta_t(
    t0: torch.Tensor,
    t1: torch.Tensor,
    attn_mask: torch.Tensor,
    mask_ties: bool,
    eps: float = 1.0,
) -> torch.Tensor:

    delta_t = t1 - t0
    delta_t = torch.clamp(delta_t, min=eps)

    if mask_ties:
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
