from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class BiomarkerEmbedConfig:
    projector: str = "linear"  # "linear", "mlp", or "embed"
    n_token: Optional[int] = 1
    n_layers: Optional[int] = None
    n_hidden: Optional[int] = None
    input_size: Optional[int] = None
    vocab_size: Optional[int] = None


@dataclass
class LossConfig:
    ce_beta: float = 1.0
    dt_beta: float = 1.0
    zero_inflate: bool = False
    zero_inflate_projector: str = "linear"


@dataclass
class DelphiConfig:
    block_size: int = 1024
    vocab_size: Optional[int] = None
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    token_dropout: float = 0.0
    t_min: float = 1.0
    bias: bool = True
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    mask_ties: bool = False
    ignore_tokens: list = field(default_factory=lambda: [0])
    biomarkers: dict[str, BiomarkerEmbedConfig] = field(default_factory=dict)
    modality_emb: bool = False
    loss: LossConfig = field(default_factory=LossConfig)


def validate_model_config(config: DelphiConfig):
    assert (
        config.mask_ties != config.loss.zero_inflate
    ), "mask_ties and zero_inflate cannot be both True or both False"


def parse_ignore_tokens(ignore_tokens: list[str]) -> list:
    if not ignore_tokens:
        return []

    parsed = []
    for ignore_token in ignore_tokens:
        if ignore_token.endswith(".yaml") or ignore_token.endswith(".yml"):
            with open(ignore_token, "r") as f:
                tokens = yaml.safe_load(f)
            if not isinstance(tokens, list):
                raise ValueError(f"Expected a list of tokens in {ignore_token}")
            parsed.extend(tokens)
        else:
            parsed.append(ignore_token)

    return parsed
