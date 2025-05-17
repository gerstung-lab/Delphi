from dataclasses import dataclass, field


@dataclass
class LossConfig:
    ce_beta: float = 1.0
    dt_beta: float = 1.0
    zero_inflate: bool = False
    zero_inflate_projector: str = "linear"


@dataclass
class DelphiConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
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
    prs: bool = False
    prs_size: int = 36
    modality_emb: bool = False
    loss: LossConfig = field(default_factory=LossConfig)


def validate_config(config: DelphiConfig):
    assert (
        config.mask_ties != config.loss.zero_inflate
    ), "mask_ties and zero_inflate cannot be both True or both False"
