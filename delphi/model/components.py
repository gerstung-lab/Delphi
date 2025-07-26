import math
from typing import Optional

import torch
import torch.nn as nn

from delphi.model.config import BiomarkerEmbedConfig, DelphiConfig
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
        self.age_encoding = AgeEncoding(config)
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
        assert config.vocab_size is not None
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
