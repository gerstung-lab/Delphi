import inspect
import math
from dataclasses import dataclass
from functools import partial

import torch

from delphi.model.transformer import LayerNorm


@dataclass
class OptimConfig:
    # adamw optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 10000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 10000  # should be ~= max_iters per Chinchilla
    min_lr: float = (
        6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    )


# learning rate decay scheduler (cosine with warmup)
def get_lr(it: int, cfg: OptimConfig) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < cfg.warmup_iters:
        return it / cfg.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.lr_decay_iters:
        return cfg.min_lr / cfg.learning_rate
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return (cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)) / cfg.learning_rate


def configure_optimizers(
    model, cfg: OptimConfig, device_type
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
    # will appear in the no_decay and decay sets respectively after the above.
    # In addition, because named_parameters() doesn't return duplicates, it
    # will only return the first occurrence, key'd by 'transformer.wte.weight', below.
    # so let's manually remove 'lm_head.weight' from decay set. This will include
    # this tensor into optimization via transformer.wte.weight only, and not decayed.
    decay.remove("lm_head.weight")

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    use_fused = (device_type == "cuda") and (
        "fused" in inspect.signature(torch.optim.AdamW).parameters
    )
    print(f"using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), **extra_args
    )

    lr_schedule_fn = partial(
        get_lr,
        cfg=cfg,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_schedule_fn,
    )

    return optimizer, scheduler
