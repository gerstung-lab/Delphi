import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from delphi.data.dataset import (
    M4Dataset,
    UKBDataConfig,
    eval_iter,
    load_sequences,
)
from delphi.log import GenLogConfig, GenLogger
from delphi.model.transformer import Delphi, load_model
from delphi.tokenizer import (
    Tokenizer,
    load_tokenizer_from_ckpt,
)


@dataclass
class FeedForwardConfig:
    name: str = "debug"
    device: str = "cpu"
    batch_size: int = 512
    subsample: Optional[int] = None
    use_val_data: bool = False
    data: UKBDataConfig = field(default_factory=UKBDataConfig)
    log: GenLogConfig = field(default_factory=GenLogConfig)


def forward(
    cfg: FeedForwardConfig,
    ckpt: str,
    model: Optional[Delphi] = None,
    tokenizer: Optional[Tokenizer] = None,
) -> None:

    ckpt = os.path.join(os.environ["DELPHI_CKPT_DIR"], ckpt)
    assert os.path.exists(ckpt), f"checkpoint {ckpt} does not exist."

    if model is None:
        model, train_cfg = load_model(ckpt)
    model.eval()
    model.to(cfg.device)

    if tokenizer is None:
        tokenizer = load_tokenizer_from_ckpt(ckpt)

    dump_dir = os.path.join(ckpt, cfg.name)
    os.makedirs(dump_dir, exist_ok=True)
    logger = GenLogger(cfg=cfg.log, dump_dir=dump_dir)
    with open(os.path.join(dump_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    if cfg.use_val_data:
        data_cfg = train_cfg.val_data
    else:
        data_cfg = cfg.data
    ds = M4Dataset(data_cfg)

    n_participants = len(ds) if cfg.subsample is None else cfg.subsample

    it = eval_iter(total_size=n_participants, batch_size=cfg.batch_size)
    loader = load_sequences(it=it, dataset=ds)

    logger.init_memmaps(
        # todo: fix quick & dirty estimate
        n_max_token=len(ds) * 50,
        n_vocab=tokenizer.vocab_size,
    )

    loader = tqdm(loader, total=math.ceil(n_participants / cfg.batch_size), leave=True)

    with torch.no_grad():
        for P, X, T, M, biomarker in loader:
            biomarker = {k: v.to(cfg.device) for k, v in biomarker.items()}
            batch_logits, _, _ = model(
                idx=X.to(cfg.device),
                age=T.to(cfg.device),
                modality=M.to(cfg.device),
                biomarker=biomarker,
            )
            logger.write_memmaps(
                participants=P.cpu().numpy(),
                tokens=X.cpu().numpy(),
                timesteps=T.cpu().numpy(),
                logits=batch_logits.cpu().numpy(),
            )

    logger.close()


def main():

    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config
    ckpt = cli_args.ckpt
    del cli_args.ckpt
    # remove 'config' and 'ckpt' attributes as the underlying dataclass does not have it

    default_cfg = OmegaConf.structured(FeedForwardConfig)
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    forward(cfg=cfg, ckpt=ckpt)  # type: ignore


if __name__ == "__main__":
    main()
