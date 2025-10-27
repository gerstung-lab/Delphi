import math
from dataclasses import dataclass, field
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from delphi.exponential import exponential_nll
from delphi.model.transformer import (
    AgeEncoding,
    Block,
    LayerNorm,
    causal_attention_mask,
    ties_adjusted_delta_t,
)
from delphi.multimodal import Modality, module_name

tensor_dict = dict[str, torch.Tensor]


class BiomarkerEmbedConfig(TypedDict, total=False):
    input_size: int
    projector: str
    n_layers: None | int
    n_hidden: None | int
    bias: bool


class BiomarkerEmbedding(nn.Module):

    def __init__(
        self,
        n_embed: int,
        input_size: int,
        projector: str,
        n_layers: None | int = None,
        n_hidden: None | int = None,
        bias: bool = False,
    ) -> None:

        super().__init__()
        if projector == "linear":
            self.projector = nn.Linear(input_size, n_embed, bias=bias)
        elif projector == "mlp":
            layers = []
            assert n_layers is not None, "n_layers must be specified for mlp projector"
            assert n_hidden is not None, "n_hidden must be specified for mlp projector"
            for i in range(n_layers):
                in_size = input_size if i == 0 else n_hidden
                out_size = n_embed if i == n_layers - 1 else n_hidden
                layers.append(nn.Linear(in_size, out_size, bias=bias))
                if i < n_layers - 1:
                    layers.append(nn.ReLU())
            self.projector = nn.Sequential(*layers)
        else:
            raise ValueError(f"unknown projector type: {projector}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class DelphiEmbedding(nn.Module):

    def __init__(self, config) -> None:
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
                n_embed=config.n_embd, **biomarker_cfg
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
            assert (
                m_emb.shape[0] == m_pos.shape[0]
            ), f"modality {modality}: m_emb {m_emb.shape} does not match m_pos {m_pos.shape}"

            token_emb[m_pos[:, 0], m_pos[:, 1], :] *= 0
            token_emb[m_pos[:, 0], m_pos[:, 1], :] += m_emb

        x = token_emb + age_emb

        if self.config.modality_emb:
            mod_emb = self.mod_embedding(M)
            x += mod_emb

        return x


@dataclass
class DelphiM4Config:
    block_size: int = 48
    vocab_size: int = 1270
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    dropout: float = 0.1
    token_dropout: float = 0.1
    t_min: float = 0.1
    bias: bool = False
    mask_ties: bool = False
    ignore_tokens: list = field(
        default_factory=lambda: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    biomarkers: dict[str, BiomarkerEmbedConfig] = field(default_factory=dict)
    modality_emb: bool = False
    ce_beta: float = 1.0
    dt_beta: float = 1.0


class DelphiM4(torch.nn.Module):
    model_type = "delphi-m4"

    def __init__(self, config: DelphiM4Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                embed=DelphiEmbedding(config),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        assert config.vocab_size is not None
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.embed.token_embedding.weight = self.lm_head.weight

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

    def loss(self, logits, targets, age, targets_age, attn_mask, targets_mask):

        loss_ce = F.cross_entropy(
            # (b, l, n_vocab) -> (b, n_vocab, l)
            logits.permute(0, 2, 1),
            targets,
            reduction="none",
        )
        loss_ce = torch.mean(loss_ce[targets_mask])
        dt = ties_adjusted_delta_t(
            t0=age,
            t1=targets_age,
            attn_mask=attn_mask,
            mask_ties=self.config.mask_ties,
            eps=self.config.t_min,
        )
        loss_dt = exponential_nll(
            delta_t=dt,
            log_lambda=torch.logsumexp(logits, -1),
            t_min=self.config.t_min,
        )
        loss_dt = torch.mean(loss_dt[targets_mask])
        return {
            "loss_ce": loss_ce * self.config.ce_beta,
            "loss_dt": loss_dt * self.config.dt_beta,
        }

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        modality: torch.Tensor,
        biomarker: None | tensor_dict = None,
        targets: None | torch.Tensor = None,
        targets_age: None | torch.Tensor = None,
    ) -> tuple[tensor_dict, None | tensor_dict, torch.Tensor]:

        x = self.transformer.embed(x0=idx, t0=age, M=modality, biomarker_x=biomarker)
        x = self.transformer.drop(x)

        attn_mask = causal_attention_mask(
            pad=(modality > 0), t1=targets_age, t0=age, mask_ties=self.config.mask_ties
        )

        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        logits = self.lm_head(x)

        if (targets is not None) and (targets_age is not None):
            is_valid_target = targets != 0
            for k in self.config.ignore_tokens:
                is_valid_target *= targets != k
            loss = self.loss(
                logits=logits,
                targets=targets,
                age=age,
                targets_age=targets_age,
                attn_mask=attn_mask,
                targets_mask=is_valid_target,
            )
        else:
            loss = None

        return {"logits": logits, "attn_mask": attn_mask}, loss, att
