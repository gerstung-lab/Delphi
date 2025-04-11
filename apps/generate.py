import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from delphi import DAYS_PER_YEAR
from delphi.config import dataclass_from_dict
from delphi.eval import clock
from delphi.model.transformer import Delphi, DelphiConfig
from delphi.sample import (
    sample_comorbid_based_on_cutoff,
    sample_competing_exponentials,
    truncate_top_k,
)
from delphi.tokenizer import (
    Tokenizer,
    load_tokenizer_from_yaml,
)
from delphi.utils import get_batch, get_p2i


@dataclass
class PromptConfig:
    data_memmap: str = "data/ukb_simulated_data/train.bin"
    subsample: Optional[int] = None
    start_age_in_years: float = 60


@dataclass
class GeneratorConfig:
    seed: int = 1337
    no_repeat: bool = True
    top_k: Optional[int] = None
    temperature: float = 1.0
    max_age_in_years: float = 80
    max_new_tokens: int = 128
    termination_tokens: list[str] = field(default_factory=list)
    simulate_comorbid: bool = True
    comorbid_cutoff: float = 0.2
    always_single_tokens: list[str] = field(default_factory=list)
    device: str = "cpu"
    batch_size: int = 512


@dataclass
class GenConfig:
    name: str = "debug"
    ckpt_path: str = "checkpoints/Delphi-demo"
    prompt: PromptConfig = field(default_factory=PromptConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)


def validate_generator_config():
    pass


def pack_arrays(matrices: list[np.ndarray], pad_val: float = 0) -> np.ndarray:

    max_cols = max(matrix.shape[1] for matrix in matrices)

    padded_matrices = []

    for matrix in matrices:
        padding = max_cols - matrix.shape[1]
        if padding > 0:
            padded_matrix = np.pad(
                matrix, ((0, 0), (0, padding)), mode="constant", constant_values=pad_val
            )
        else:
            padded_matrix = matrix
        padded_matrices.append(padded_matrix)

    return np.vstack(padded_matrices)


class Generator:

    def __init__(self, cfg: GeneratorConfig, model: Delphi, tokenizer: Tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def next_token(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        logits, _, _ = self.model(idx, age)

        logits = logits[:, -1, :] / self.cfg.temperature

        logits[:, self.model.config.ignore_tokens] = -torch.inf

        if self.cfg.top_k is not None:
            logits = truncate_top_k(logits, self.cfg.top_k)

        if self.cfg.no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = logits.scatter_(1, fill, -torch.inf)

        if self.cfg.simulate_comorbid:
            always_single_tokens = [
                self.tokenizer[disease] for disease in self.cfg.always_single_tokens
            ]
            idx_next, time_til_next = sample_comorbid_based_on_cutoff(
                logits=logits,
                comorbid_cutoff=self.cfg.comorbid_cutoff,
                always_single_tokens=always_single_tokens,
            )
        else:
            idx_next, time_til_next = sample_competing_exponentials(logits)

        return idx_next, time_til_next

    @clock
    @torch.no_grad()
    def generate_one_batch(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        termination_tokens = torch.Tensor(
            [self.tokenizer[token] for token in self.cfg.termination_tokens]
        )

        max_age = self.cfg.max_age_in_years * DAYS_PER_YEAR
        n_new_tokens = 0
        while True:

            idx_next, time_til_next = self.next_token(
                idx=idx,
                age=age,
            )
            age_next = age[..., [-1]] + time_til_next
            age_next[time_til_next == -10000] = -10000

            n_new_tokens += idx_next.shape[1]
            if n_new_tokens >= self.cfg.max_new_tokens:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)

            sort_by_time = age.argsort(1)
            idx = idx.gather(1, sort_by_time)
            age = age.gather(1, sort_by_time)

            is_termination_token = torch.isin(idx, termination_tokens)
            if torch.logical_or(
                is_termination_token.any(-1), (age_next > max_age).any(-1)
            ).all():
                break

        exceed_max_age = age > max_age
        beyond_termination = (
            torch.cumsum(torch.cumsum(is_termination_token.int(), 1), 1) > 1
        )

        pad = exceed_max_age | beyond_termination
        idx[pad] = 0
        age[pad] = -10000

        logits, _, _ = self.model(idx, age)

        if self.cfg.no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = torch.stack(
                [
                    logits[:, j].scatter_(1, fill[:, : j + 1], float("NaN"))
                    for j in range(fill.shape[1])
                ]
            ).transpose(0, 1)

        return idx, age, logits

    def generate(
        self,
        d0: torch.Tensor,
        d1: torch.Tensor,
    ) -> tuple[list, list, list]:

        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)

        token_paths = []
        time_paths = []
        logits_paths = []

        self.model.to(self.cfg.device)
        batch = 0
        for dd in zip(*map(lambda x: torch.split(x, self.cfg.batch_size), (d0, d1))):

            print(f"generating batch {batch}...")
            idx, age, logits = self.generate_one_batch(
                dd[0].to(self.cfg.device),
                dd[1].to(self.cfg.device),
            )

            token_paths.append(idx.cpu().numpy())
            time_paths.append(age.cpu().numpy())
            logits_paths.append(logits.cpu().numpy())

            batch += 1

        return token_paths, time_paths, logits_paths


def load_tokenizer(
    ckpth_path,
) -> Tokenizer:

    tokenizer_path = Path(ckpth_path) / "tokenizer.yaml"
    tokenizer = load_tokenizer_from_yaml(tokenizer_path)

    return tokenizer


def load_model(
    ckpt_path,
    model_cls=Delphi,
    model_cfg_cls=DelphiConfig,
):

    ckpt_path = Path(ckpt_path)
    train_cfg = OmegaConf.load(ckpt_path / "config.yaml")
    ckpt_dict = torch.load(
        ckpt_path / "ckpt.pt",
        map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    )

    param_dtype = dict(
        float32=torch.float32,
        float64=torch.float64,
        float16=torch.float16,
        bfloat16=torch.bfloat16,
    )[train_cfg.dtype]
    model_cfg = dataclass_from_dict(model_cfg_cls, train_cfg.model, strict=False)
    model = model_cls(model_cfg)
    model.load_state_dict(ckpt_dict["model"])
    model = model.eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)

    return model, train_cfg


def load_prompt(
    cfg: PromptConfig,
) -> tuple[torch.Tensor, torch.Tensor]:

    val = np.fromfile(cfg.data_memmap, dtype=np.uint32).reshape(-1, 3)

    val_p2i = get_p2i(val)

    if cfg.subsample is not None:
        ix = range(0, cfg.subsample, 1)
    else:
        ix = range(0, val_p2i.shape[0] - 1, 1)

    d = get_batch(
        ix=ix,
        data=val,
        p2i=val_p2i,
        select="left",
        block_size=63,
        device="cpu",  # TODO: fix this with a proper device
        padding="random",
        cut_batch=True,
    )

    n_samples = 1024 * 1024  # TODO: find out the purpose of this

    start_age = cfg.start_age_in_years * DAYS_PER_YEAR

    w = np.where(
        (d[1].cpu().detach().numpy() <= start_age).any(1)
        * (d[3].cpu().detach().numpy() >= start_age).any(1)
    )
    # select everything
    # w = np.arange(d[0].shape[0])[None]
    u = np.unique(w[0])

    d0 = d[0][u[:n_samples]].clone().detach()
    d1 = d[1][u[:n_samples]].clone().detach()

    d0[d1 > start_age] = 0
    d1[d1 > start_age] = -10000.0

    if start_age > 0:
        d0 = torch.nn.functional.pad(d0, (0, 1), "constant", 1)
        d1 = torch.nn.functional.pad(d1, (0, 1), "constant", start_age)

    o = d1.argsort(1)
    d0 = d0.gather(1, o)
    d1 = d1.gather(1, o)

    return d0, d1


def gen(gen_cfg: GenConfig) -> None:
    """
    Generate data using the generator.
    """

    dump_dir = os.path.join(gen_cfg.ckpt_path, gen_cfg.name)
    os.makedirs(dump_dir, exist_ok=True)
    with open(os.path.join(dump_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=gen_cfg, f=f)

    model, _ = load_model(gen_cfg.ckpt_path)
    tokenizer = load_tokenizer(gen_cfg.ckpt_path)
    gen = Generator(cfg=gen_cfg.generator, model=model, tokenizer=tokenizer)

    d0, d1 = load_prompt(cfg=gen_cfg.prompt)

    tokens, timesteps, logits = gen.generate(d0=d0, d1=d1)

    tokens = pack_arrays(tokens, pad_val=0)
    timesteps = pack_arrays(timesteps, pad_val=-10000)
    logits = pack_arrays(logits, pad_val=-np.inf)

    np.save(arr=tokens, file=os.path.join(dump_dir, "token.npy"))
    np.save(arr=timesteps.astype(int), file=os.path.join(dump_dir, "timesteps.npy"))
    np.save(arr=logits, file=os.path.join(dump_dir, "logits.npy"))


def main():

    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(GenConfig)
    gen_cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    gen_cfg = OmegaConf.to_object(gen_cfg)

    gen(gen_cfg)


if __name__ == "__main__":
    main()
