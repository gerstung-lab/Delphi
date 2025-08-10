from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class BiomarkerEmbedConfig:
    projector: str = "linear"  # "linear", "mlp", or "embed"
    n_layers: Optional[int] = None
    n_hidden: Optional[int] = None
    input_size: Optional[int] = None


@dataclass
class GPT2Config:
    vocab_size: int = 1270
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    bias: bool = True
    block_size: int = 64


@dataclass
class DelphiConfig(GPT2Config):
    vocab_size: Optional[int] = None
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    mask_ties: bool = False
    ignore_tokens: list = field(default_factory=lambda: [0])
    biomarkers: dict[str, BiomarkerEmbedConfig] = field(default_factory=dict)
    modality_emb: bool = False
    ce_beta: float = 1.0
    dt_beta: float = 1.0
    zero_inflate: bool = False
    zero_inflate_projector: str = "linear"


def validate_model_config(config: DelphiConfig):
    assert (
        config.mask_ties != config.zero_inflate
    ), "mask_ties and zero_inflate cannot be both True or both False"


def validate_model_config_for_finetuning(
    finetune_config: DelphiConfig, pretrain_config: DelphiConfig
) -> None:

    assert (
        finetune_config.vocab_size == pretrain_config.vocab_size
        and finetune_config.n_layer == pretrain_config.n_layer
        and finetune_config.n_head == pretrain_config.n_head
        and finetune_config.n_embd == pretrain_config.n_embd
        and finetune_config.bias == pretrain_config.bias
    ), "model dimensions must match between finetune and pretrain configs"

    finetune_biomarkers = set(finetune_config.biomarkers.keys())
    pretrain_biomarkers = set(pretrain_config.biomarkers.keys())
    assert pretrain_biomarkers.issubset(
        finetune_biomarkers
    ), "finetune config must have all biomarkers from pretrain config"

    intersect_biomarkers = finetune_biomarkers.intersection(pretrain_biomarkers)
    for biomarker in intersect_biomarkers:
        finetune_bm = finetune_config.biomarkers[biomarker]
        pretrain_bm = pretrain_config.biomarkers[biomarker]

        assert (
            finetune_bm.projector == pretrain_bm.projector
            and finetune_bm.n_layers == pretrain_bm.n_layers
            and finetune_bm.n_hidden == pretrain_bm.n_hidden
            and finetune_bm.input_size == pretrain_bm.input_size
        ), f"biomarker {biomarker} embed configs must match between finetune and pretrain configs"


def parse_token_list(token_list: list[str]) -> list:
    if not token_list:
        return []

    parsed = []
    for token in token_list:
        if token.endswith(".yaml") or token.endswith(".yml"):
            with open(token, "r") as f:
                tokens = yaml.safe_load(f)
            if not isinstance(tokens, list):
                raise ValueError(f"Expected a list of tokens in {token}")
            parsed.extend(tokens)
        else:
            parsed.append(token)

    return parsed
