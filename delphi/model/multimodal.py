import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Required, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from delphi.data.ukb import collate_batch
from delphi.exponential import exponential_nll
from delphi.model.transformer import (
    MLP,
    AgeEncoding,
    Block,
    LayerNorm,
    causal_attention_mask,
)
from delphi.multimodal import Modality, module_name

tensor_dict = dict[str, torch.Tensor]


class BiomarkerEmbedConfig(TypedDict, total=False):
    input_size: Required[int]
    projector: Required[str]
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
        idx: torch.Tensor,
        age: torch.Tensor,
        mod_idx: torch.Tensor,
        mod_age: torch.Tensor,
        biomarker_x: dict[Modality, torch.Tensor],
    ):

        idx_emb = self.token_embedding(idx)
        idx_emb = self.token_drop(idx_emb) * (1 - self.config.token_dropout)
        age_emb = self.age_encoding(age.unsqueeze(-1))
        emb = idx_emb + age_emb

        biomarker_emb = dict()
        mod_age_emb = self.age_encoding(mod_age.unsqueeze(-1))
        for modality in biomarker_x.keys():
            biomarker_emb[modality] = self.biomarker_embed[module_name(modality)](
                biomarker_x[modality]
            )  # N * H
            mod_mask = mod_idx == modality.value
            biomarker_emb[modality] += mod_age_emb[mod_mask]
            if self.config.modality_emb:
                mod_emb = self.mod_embedding(
                    torch.tensor(modality.value).to(idx.device)
                )
                biomarker_emb[modality] += mod_emb.unsqueeze(0)

        raw = {
            "idx": idx_emb,
            "age": age_emb,
            "mod_age": mod_age_emb,
            "biomarker": biomarker_emb,
        }
        return emb, biomarker_emb, raw


class CrossAttention(nn.Module):

    def __init__(self, n_embd, n_head, bias, dropout):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head

        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        pass

    def forward(self, main_seq, side_seq, attn_mask):

        B, L_main, _ = main_seq.shape
        _, L_side, _ = side_seq.shape

        q = self.q_proj(main_seq)
        k = self.k_proj(side_seq)
        v = self.v_proj(side_seq)

        k = k.view(B, L_side, self.n_head, self.n_embd // self.n_head).transpose(
            1, 2
        )  # (B, nh, L_side, hs)
        q = q.view(B, L_main, self.n_head, self.n_embd // self.n_head).transpose(
            1, 2
        )  # (B, nh, L_main, hs)
        v = v.view(B, L_side, self.n_head, self.n_embd // self.n_head).transpose(
            1, 2
        )  # (B, nh, L_side, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        all_masked = torch.all(attn_mask == 0, dim=-1, keepdim=True)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        att = att.masked_fill(all_masked, 0)
        y = att @ v
        # (B, nh, L_main, L_side) x (B, nh, L_side, hs) -> (B, nh, L_main, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, L_main, self.n_embd)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att


class CrossBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_0 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CrossAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            bias=config.bias,
            dropout=config.dropout,
        )
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, side_x, attn_mask):
        y, att = self.attn(self.ln_1(x), self.ln_0(side_x), attn_mask)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att


def time_attention_mask(query_age: torch.Tensor, key_age: torch.Tensor):
    assert query_age.shape[0] == key_age.shape[0]
    attn_mask = query_age.unsqueeze(-1) >= key_age.unsqueeze(1)
    pad_mask = (key_age >= 0).view(key_age.shape[0], 1, key_age.shape[1])
    attn_mask *= pad_mask
    attn_mask = attn_mask.unsqueeze(1)
    return attn_mask


class CrossFusion(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fuse = CrossBlock(config)

    def forward(self, x, age, mod_idx, mod_age, mod_emb):
        mod_x, _ = fuse_embed(mod_idx=mod_idx, mod_age=mod_age, mod_emb=mod_emb)
        attn_mask = time_attention_mask(query_age=age, key_age=mod_age)
        x, _ = self.fuse(x=x, side_x=mod_x, attn_mask=attn_mask)
        return x, torch.ones_like(age)


class SelfFusion(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fuse = Block(config)

    def forward(
        self,
        x: torch.Tensor,
        age: torch.Tensor,
        targets_age: torch.Tensor,
        mod_idx: torch.Tensor,
        mod_age: torch.Tensor,
        mod_emb: dict[Modality, torch.Tensor],
    ):
        x, fused_mod_idx = fuse_embed(
            mod_idx=mod_idx, mod_age=mod_age, mod_emb=mod_emb, emb=x, age=age
        )
        fused_age = fuse_age(age, mod_age, fused_mod_idx)
        fused_targets_age = fuse_age(targets_age, mod_age, fused_mod_idx)

        attn_mask = causal_attention_mask(
            pad=fused_age != -1e4,
            mask_ties=self.config.mask_ties,
            t0=fused_age,
            t1=fused_targets_age,
        )
        x, _ = self.fuse(x=x, attn_mask=attn_mask)
        return x, fused_mod_idx


class ConcatFusion(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.raw = "raw" in config.fuse
        if self.raw:
            self.biomarkers = {
                Modality[k.upper()]: biomarker["input_size"]
                for k, biomarker in config.biomarkers.items()
            }
        else:
            self.biomarkers = {
                Modality[k.upper()]: config.n_embd for k in config.biomarkers.keys()
            }
        input_dim = sum(list(self.biomarkers.values()))
        input_dim += config.n_embd
        self.fuse = nn.Linear(input_dim, config.n_embd)

    def forward(
        self,
        age: torch.Tensor,
        mod_idx: torch.Tensor,
        mod_age: torch.Tensor,
        idx_emb: torch.Tensor,
        bio_emb: dict[Modality, torch.Tensor],
    ):
        all_bio_X = list()
        for modality, bio_dim in self.biomarkers.items():
            n_per_sample = torch.sum(mod_idx == modality.value, dim=1)
            assert torch.all(n_per_sample <= 1)

            bio_x = torch.zeros((idx_emb.shape[0], bio_dim), device=idx_emb.device)
            bio_x[n_per_sample > 0, :] = bio_emb[modality]
            bio_x = bio_x.unsqueeze(1).expand(-1, age.shape[1], -1).clone()

            bio_t = torch.full(
                (idx_emb.shape[0],), fill_value=-1e4, device=idx_emb.device
            )
            bio_t[n_per_sample > 0] = mod_age[mod_idx == modality.value]
            bio_x[age < bio_t.unsqueeze(-1)] = 0

            all_bio_X.append(bio_x)
        all_bio_X = torch.cat(all_bio_X, dim=-1)
        idx_emb = torch.cat((idx_emb, all_bio_X), dim=-1)
        x = self.fuse(idx_emb)

        return x, None


def forward_fill(tensor: torch.Tensor, fill_mask: torch.Tensor, dim: int):
    assert tensor.shape == fill_mask.shape
    idx = torch.where(
        fill_mask, 0, torch.arange(fill_mask.shape[dim], device=tensor.device)
    )
    idx = torch.cummax(idx, dim=dim)[0]
    return torch.gather(tensor, dim, idx)


def fuse_embed(
    mod_idx: torch.Tensor,
    mod_age: torch.Tensor,
    mod_emb: dict[Modality, torch.Tensor],
    emb: None | torch.Tensor = None,
    age: None | torch.Tensor = None,
):
    if emb is not None:
        assert age is not None
        pseudo_mod_idx = torch.ones_like(age)
        fused_mod_idx = torch.cat((mod_idx, pseudo_mod_idx), dim=1)
        fused_age = torch.cat((mod_age, age), dim=1)
        n_embd = emb.shape[-1]
    else:
        fused_mod_idx = mod_idx
        fused_age = mod_age
        n_embd = list(mod_emb.values())[0].shape[-1]

    # stable = True ensures biomarkers precede disease tokens
    time_sort = torch.argsort(fused_age, stable=True, dim=1)
    fused_mod_idx = torch.take_along_dim(fused_mod_idx, time_sort, dim=1)
    fused_emb = torch.zeros((*fused_mod_idx.shape, n_embd)).to(mod_idx.device)
    if emb is not None:
        m_pos = torch.nonzero(fused_mod_idx == 1)
        fused_emb[m_pos[:, 0], m_pos[:, 1]] = emb.view(-1, emb.shape[-1])
    for modality, m_emb in mod_emb.items():
        m_pos = torch.nonzero(fused_mod_idx == modality.value)  # N * 2
        assert (
            m_emb.shape[0] == m_pos.shape[0]
        ), f"modality {modality}: m_emb {m_emb.shape} does not match m_pos {m_pos.shape}"
        fused_emb[m_pos[:, 0], m_pos[:, 1], :] *= 0
        fused_emb[m_pos[:, 0], m_pos[:, 1], :] += m_emb

    return fused_emb, fused_mod_idx


def fuse_age(age: torch.Tensor, mod_age: torch.Tensor, fused_mod_idx: torch.Tensor):

    fused_age = torch.zeros_like(fused_mod_idx).to(age.dtype)
    fused_age[fused_mod_idx == 1] = age.view(-1)
    fused_age[fused_mod_idx != 1] = mod_age.view(-1)

    return fused_age


@dataclass
class DelphiM4Config:
    block_size: None | int = 256
    vocab_size: int = 1270
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    dropout: float = 0.1
    token_dropout: float = 0.0
    t_min: float = 0.1
    bias: bool = True
    mask_ties: bool = True
    weight_tying: bool = True
    ignore_tokens: list = field(
        default_factory=lambda: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    biomarkers: dict[str, BiomarkerEmbedConfig] = field(default_factory=dict)
    modality_emb: bool = True
    biomarker_only: bool = False
    ce_beta: float = 1.0
    dt_beta: float = 1.0
    fuse: str = "early"  # early, cross, concat, concat-raw


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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.weight_tying:
            self.transformer.embed.token_embedding.weight = self.lm_head.weight

        self.fuse_early = self.config.fuse == "early"
        if (not self.fuse_early) and config.biomarker_only:
            raise ValueError
        if self.config.fuse in {"concat", "concat-raw"}:
            raise NotImplementedError
        elif self.config.fuse == "cross":
            self.fuse = CrossFusion(config)
        elif self.config.fuse == "late":
            self.fuse = SelfFusion(config)

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

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        age: torch.Tensor,
        targets_age: torch.Tensor,
    ):

        loss_ce = F.cross_entropy(
            # (b, l, n_vocab) -> (b, n_vocab, l)
            logits.permute(0, 2, 1),
            targets,
            reduction="none",
        )

        dt = targets_age - age
        is_tie = dt == 0
        dt = torch.clamp(dt, min=self.config.t_min)
        is_tie[age == -1e4] = False
        if self.config.mask_ties:
            dt = forward_fill(dt, is_tie, dim=1)

        loss_dt = exponential_nll(
            delta_t=dt,
            log_lambda=torch.logsumexp(logits, -1),
            t_min=self.config.t_min,
        )

        return loss_ce, loss_dt

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        biomarker: dict[Modality, torch.Tensor],
        mod_age: torch.Tensor,
        mod_idx: torch.Tensor,
        targets: None | torch.Tensor = None,
        targets_age: None | torch.Tensor = None,
    ) -> tuple[tensor_dict, None | tensor_dict, torch.Tensor]:

        x, mod_emb, raw = self.transformer.embed(
            idx=idx, age=age, mod_idx=mod_idx, mod_age=mod_age, biomarker_x=biomarker
        )

        if self.config.biomarker_only:
            x = raw["age"]

        if self.fuse_early:
            x, fused_mod_idx = fuse_embed(
                emb=x, age=age, mod_idx=mod_idx, mod_age=mod_age, mod_emb=mod_emb
            )
            t0 = fuse_age(age, mod_age, fused_mod_idx)
            t1 = (
                fuse_age(targets_age, mod_age, fused_mod_idx)
                if targets_age is not None
                else None
            )
        else:
            t0 = age
            t1 = targets_age
        attn_mask = causal_attention_mask(
            pad=t0 != -1e4, t0=t0, t1=t1, mask_ties=self.config.mask_ties
        )
        if self.config.biomarker_only:
            attn_idx = torch.arange(attn_mask.shape[-1], device=attn_mask.device)
            attn_idx = attn_idx.view(1, 1, 1, -1) * attn_mask
            last_idx = attn_idx.max(dim=-1, keepdim=True)[0]
            attn_mask *= (fused_mod_idx != 1).view(
                fused_mod_idx.shape[0], 1, 1, fused_mod_idx.shape[-1]
            )
            attn_mask.scatter_(dim=-1, index=last_idx.long(), value=1)

        x = self.transformer.drop(x)
        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        if not self.fuse_early:
            if self.config.fuse == "cross":
                x, fused_mod_idx = self.fuse(
                    x=x, age=age, mod_idx=mod_idx, mod_age=mod_age, mod_emb=mod_emb
                )
            elif self.config.fuse == "late":
                x, fused_mod_idx = self.fuse(
                    x=x,
                    age=age,
                    targets_age=targets_age,
                    mod_idx=mod_idx,
                    mod_age=mod_age,
                    mod_emb=mod_emb,
                )
            else:
                raise NotImplementedError

        x = x[fused_mod_idx == 1].view(*idx.shape, -1)  # type: ignore
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
                targets_mask=is_valid_target,
            )
        else:
            loss = None

        return {"logits": logits, "attn_mask": attn_mask}, loss, att
