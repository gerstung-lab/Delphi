import math
from typing import Optional

import torch
import torch.nn as nn
import yaml

from delphi.model.config import DelphiConfig
from delphi.multimodal import Modality, module_name


class AgeEncoding(nn.Module):

    def __init__(self, config, max_dim: int = 1024):
        super().__init__()
        div_term = torch.exp(
            torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd)
        )
        self.register_buffer("div_term", div_term)
        self.n_embd = config.n_embd
        self.linear = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(x / 365.25 * self.div_term)  # * (1-self.div_term)
        y[..., 1::2] = torch.cos(x / 365.25 * self.div_term)  # * (1-self.div_term)
        y = self.linear(y)

        # x = self.wae[:x.size(0)]
        return y  # self.dropout(x)


class DelphiEmbedding(nn.Module):

    def __init__(self, config: DelphiConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd, padding_idx=0
        )
        self.age_encoding = AgeEncoding(config)
        self.token_drop = nn.Dropout(config.token_dropout)

        self.biomarker_embed = nn.ModuleDict()
        n_modality = 2  # pad + event
        if config.prs:
            if config.prs_projector == "linear":
                self.biomarker_embed[module_name(Modality.PRS)] = nn.Linear(
                    config.prs_size, config.n_embd, bias=False
                )
            elif config.prs_projector == "mlp":
                self.biomarker_embed[module_name(Modality.PRS)] = nn.Sequential(
                    nn.Linear(config.prs_size, 64, bias=False),
                    nn.ReLU(),
                    nn.Linear(64, config.n_embd, bias=False),
                )
            else:
                raise ValueError(f"unknown prs_projector: {config.prs_projector}")
            n_modality += 1

        if config.family_hx:
            with open(config.family_hx_map_yaml, "r") as f:
                map_config = yaml.safe_load(f)
            family_hx_vocab_size = max(map_config.values()) + 1
            self.biomarker_embed[module_name(Modality.FAMILY_HX)] = nn.Embedding(
                family_hx_vocab_size, config.n_embd, padding_idx=0
            )

        if config.modality_emb:
            self.mod_embedding = nn.Embedding(n_modality, config.n_embd, padding_idx=0)

    def ties_adjusted_delta_t(
        self, t0: torch.Tensor, t1: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:

        delta_t = t1 - t0
        if not self.config.loss.zero_inflate:
            delta_t = torch.clamp(delta_t, min=1.0)

        if self.config.mask_ties:
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


class ZeroInflateProjector(nn.Module):
    def __init__(self, config: DelphiConfig) -> None:
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(config.vocab_size, 32, bias=False),
                nn.ReLU(),
                nn.Linear(32, 1, bias=False),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = l(x)
        return x


def build_zero_inflate_projector(config: DelphiConfig):

    if config.loss.zero_inflate_projector == "linear":
        return nn.Linear(config.vocab_size, 1, bias=False)
    elif config.loss.zero_inflate_projector == "mlp":
        return ZeroInflateProjector(config)
    else:
        raise ValueError(f"Unknown pi_projector: {config.loss.zero_inflate_projector}")


def attention_mask(
    t0: torch.Tensor,
    m0: torch.Tensor,
    mask_ties: bool,
    t1: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    b, l = m0.shape[0], m0.shape[1]

    lower_tri_mask = torch.tril(torch.ones((l, l), device=m0.device))
    lower_tri_mask = lower_tri_mask.view(1, l, l)
    pad_mask = (m0 > 0).view(b, 1, l).to(torch.int)
    attn_mask = pad_mask * lower_tri_mask

    if mask_ties:
        if t1 is not None:
            ties_mask = (t1.view(b, l, 1) != t0.view(b, 1, l)).to(torch.int)
            attn_mask *= ties_mask

    attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(
        torch.ones(m0.size(1), device=m0.device)
    ) > 0

    return attn_mask.unsqueeze(1)


def target_mask(
    x1: torch.Tensor,
    ignore_tokens: list[int],
) -> torch.Tensor:

    is_valid_target = x1 != -1
    for k in ignore_tokens:
        is_valid_target *= x1 != k

    return is_valid_target


def mask_biomarker_only_predicate(
    x1: torch.Tensor,
    m0: torch.Tensor,
):

    first_non_pad = torch.argmax(m0 > 0, dim=1)
    first_biomarker = torch.argmax(m0 > 1, dim=1)

    biomarker_only_predicate = first_non_pad == first_biomarker
    mask_positions = first_biomarker[biomarker_only_predicate]

    is_valid_target = torch.ones_like(x1, dtype=torch.bool)
    is_valid_target[torch.arange(mask_positions.shape[0]), mask_positions] = False

    return is_valid_target
