import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from omegaconf import OmegaConf

from delphi.data.dataset import (
    PromptDataset,
    UKBDataConfig,
    build_prefetch_loader,
    eval_iter,
    load_sequences,
)
from delphi.log import GenLogConfig, GenLogger
from delphi.model.transformer import Delphi, load_model
from delphi.sampler import (
    CausalSampler,
    CausalSamplerConfig,
)
from delphi.tokenizer import (
    Tokenizer,
    load_tokenizer_from_ckpt,
)


@dataclass
class GenConfig:
    name: str = "debug"
    device: str = "cpu"
    batch_size: int = 512
    subsample: Optional[int] = None
    start_age_in_years: float = 60.0
    data: UKBDataConfig = field(default_factory=UKBDataConfig)
    sampler: CausalSamplerConfig = field(default_factory=CausalSamplerConfig)
    log: GenLogConfig = field(default_factory=GenLogConfig)


def gen(
    gen_cfg: GenConfig,
    ckpt: str,
    model: Optional[Delphi] = None,
    tokenizer: Optional[Tokenizer] = None,
) -> None:
    """
    Generate data using the generator.
    """

    if model is None:
        model, _ = load_model(ckpt)
    model.eval()
    model.to(gen_cfg.device)

    if tokenizer is None:
        tokenizer = load_tokenizer_from_ckpt(ckpt)

    dump_dir = os.path.join(ckpt, gen_cfg.name)
    os.makedirs(dump_dir, exist_ok=True)
    logger = GenLogger(cfg=gen_cfg.log, dump_dir=dump_dir)
    with open(os.path.join(dump_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=gen_cfg, f=f)

    sampler = CausalSampler(cfg=gen_cfg.sampler, model=model, tokenizer=tokenizer)

    ds = PromptDataset(gen_cfg.data, start_age_in_years=gen_cfg.start_age_in_years)

    n_participants = len(ds) if gen_cfg.subsample is None else gen_cfg.subsample

    it = eval_iter(total_size=n_participants, batch_size=gen_cfg.batch_size)
    total_batch = math.ceil(n_participants / gen_cfg.batch_size)
    loader = load_sequences(it=it, dataset=ds)
    loader = build_prefetch_loader(loader=loader)

    n_max_token = n_participants * gen_cfg.sampler.max_new_tokens
    logger.init_memmaps(
        n_max_token=n_max_token,
        # +1 to account for 0 padding
        n_vocab=tokenizer.vocab_size + 1,
    )

    batch = 1
    for P, X, T in loader:

        print(f"generating batch {batch}/{total_batch}...")
        idx, age, logits = sampler.generate(
            X.to(gen_cfg.device),
            T.to(gen_cfg.device),
        )

        tokens = idx.cpu().numpy().astype(np.uint16)
        timesteps = age.cpu().numpy().astype(np.int32)
        logits = logits.cpu().numpy().astype(np.float16)

        logger.write_memmaps(
            participants=P.cpu().numpy(),
            tokens=tokens,
            timesteps=timesteps,
            logits=logits,
        )

        batch += 1

    logger.close()


def main():

    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config
    ckpt = cli_args.ckpt
    del cli_args.ckpt
    # remove 'config' and 'ckpt' attributes as the underlying dataclass does not have it

    default_cfg = OmegaConf.structured(GenConfig)
    gen_cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    gen_cfg = OmegaConf.to_object(gen_cfg)

    gen(gen_cfg=gen_cfg, ckpt=ckpt)  # type: ignore


if __name__ == "__main__":
    main()
