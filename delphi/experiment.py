import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml
from omegaconf import OmegaConf

from delphi import distributed
from delphi.env import DELPHI_CKPT_DIR
from delphi.log import TrainLogConfig, TrainLogger
from delphi.model.multimodal import DelphiM4, DelphiM4Config
from delphi.model.transformer import Delphi2M, Delphi2MConfig
from delphi.optim import OptimConfig, configure_optimizers


def move_batch_to_device(args: Iterable, device: str):

    outputs = list()
    for arg in args:
        if isinstance(arg, torch.Tensor):
            outputs.append(arg.to(device))
        elif isinstance(arg, dict):
            outputs.append({k: v.to(device) for k, v in arg.items()})
        else:
            raise NotImplementedError
    return tuple(outputs)


def train_iter(
    seed: int,
    total_size: int,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    step: int = 0,
) -> Iterator[np.ndarray]:

    while True:
        seed_with_offset = seed + step * world_size + rank
        rng = np.random.default_rng(seed_with_offset)
        batch_idx = rng.integers(total_size, size=(batch_size,))
        step += 1

        yield batch_idx


def eval_iter(total_size: int, batch_size: int) -> Iterator[np.ndarray]:

    batch_start_pos = np.arange(0, total_size, batch_size)
    batch_end_pos = batch_start_pos + batch_size
    batch_end_pos[-1] = total_size

    for start, end in zip(batch_start_pos, batch_end_pos):
        yield np.arange(start, end)


@dataclass
class TrainBaseConfig:
    ckpt_dir: str = "."
    eval_interval: int = 2000
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    init_from: str = "scratch"

    seed: int = 42
    gradient_accumulation_steps: int = 1  # used to simulate larger batch sizes
    batch_size: int = 128
    # if gradient_accumulation_steps > 1, this is the micro-batch size

    # system
    device: str = "cuda"
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "float32"
    # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster

    distributed_backend: Optional[str] = None

    optim: OptimConfig = field(default_factory=OptimConfig)

    log: TrainLogConfig = field(default_factory=TrainLogConfig)


class BaseTrainer:

    def __init__(
        self,
        cfg: TrainBaseConfig,
        backend: distributed.backend.DistributedBackend,
        model: torch.nn.Module,
        train_ds: Any,
        val_ds: Any,
    ):
        self.backend = backend
        cfg = self.backend.get_adjusted_args_for_process(cfg)

        self.cfg = cfg
        self.device = cfg.device
        self.device_type = (
            "cuda" if "cuda" in cfg.device else "cpu"
        )  # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        self.ptdtype = {
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[cfg.dtype]
        if self.device_type == "cuda" and self.ptdtype == "float16":
            self.ctx = torch.autocast(device_type=self.device_type, dtype=self.ptdtype)
        else:
            self.ctx = nullcontext()

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model = model
        self.model.to(self.device)
        self.model = self.backend.transform_model(self.model)

        self.optimizer, self.scheduler = configure_optimizers(
            model=model, cfg=cfg.optim, device_type=self.device_type
        )
        self.scaler = torch.GradScaler(
            device=self.device_type, enabled=(cfg.dtype == "float16")
        )

        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            print(f"\tinitialized data loader for worker {rank}/{world_size}")
        else:
            world_size = 1
            rank = 0
        self.train_iter = train_iter(
            seed=cfg.seed,
            total_size=len(train_ds),
            batch_size=cfg.batch_size,
            world_size=world_size,
            rank=rank,
        )
        self.estimate_iters = {
            "train": train_iter(
                seed=cfg.seed,
                total_size=len(train_ds),
                batch_size=cfg.batch_size,
                world_size=world_size,
                rank=rank,
            ),
            "val": train_iter(
                seed=cfg.seed,
                total_size=len(val_ds),
                batch_size=cfg.batch_size,
                world_size=world_size,
                rank=rank,
            ),
        }

        run_dir = os.path.normpath(
            os.path.join(DELPHI_CKPT_DIR, cfg.ckpt_dir, cfg.log.run_name)
        )
        self.logger = TrainLogger(
            cfg=cfg.log,
            exp_cfg=asdict(cfg),
            dump_dir=run_dir,
            tokenizer=train_ds.tokenizer,
            model=self.model,  # type: ignore
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            backend=self.backend,
        )

        self.iter_num = 0

    def mini_step(
        self, batch_data: Iterable, *args, **kwargs
    ) -> dict[str, torch.Tensor]:

        batch_data = move_batch_to_device(args=batch_data, device=self.device)
        with self.ctx:
            _, loss, _ = self.model(*batch_data)

        return loss

    @torch.no_grad()
    def estimate_loss(self, *args, **kwargs) -> tuple[dict, dict]:
        self.model.eval()
        eval_loss = {}
        for split in ["train", "val"]:
            split_loss = defaultdict(float)
            for _ in range(self.cfg.eval_iters):

                batch_idx = next(self.estimate_iters[split])
                if split == "train":
                    batch_data = self.train_ds.get_batch(batch_idx)
                else:
                    batch_data = self.val_ds.get_batch(batch_idx)
                loss = self.mini_step(batch_data=batch_data)

                for key in loss.keys():
                    split_loss[key] += loss[key].item()
            split_loss = dict(split_loss)
            for key in split_loss.keys():
                split_loss[key] /= self.cfg.eval_iters
            eval_loss[split] = split_loss

        self.model.train()

        return eval_loss["train"], eval_loss["val"]

    def train(self):

        if self.cfg.compile:
            print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model)

        while True:

            # evaluate the loss on train/val sets and write checkpoints
            if self.iter_num % self.cfg.eval_interval == 0 and self.iter_num > 0:
                _, val_loss = self.estimate_loss()
                self.logger.eval_step(step=self.iter_num, loss=val_loss)

            if self.iter_num == 0 and self.cfg.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16``
            for i in range(self.cfg.gradient_accumulation_steps):
                with self.backend.get_context_for_microstep_forward(
                    model=self.model,
                    microstep_idx=i,
                    gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
                ):
                    batch_idx = next(self.train_iter)
                    batch_data = self.train_ds.get_batch(batch_idx)
                    loss = self.mini_step(batch_data=batch_data)

                # backward pass, with gradient scaling if training in fp16
                loss_agg = sum([loss[key] for key in loss.keys()])
                self.scaler.scale(loss_agg).backward()  # type: ignore

            # clip the gradient
            if self.cfg.optim.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.optim.grad_clip
                )

            self.logger.train_step(step=self.iter_num, loss=loss)
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            self.logger.ckpt_step(step=self.iter_num)

            self.iter_num += 1

            # termination conditions
            if self.iter_num > self.cfg.optim.max_iters:
                break


def load_ckpt(ckpt_path):

    ckpt_path = Path(ckpt_path)
    train_cfg = OmegaConf.load(ckpt_path / "config.yaml")
    ckpt_dict = torch.load(
        ckpt_path / "ckpt.pt",
        map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    )
    model_type = ckpt_dict["model_type"]
    if model_type == "delphi-2m":
        model_cfg_cls = Delphi2MConfig
        model_cls = Delphi2M
    elif model_type == "delphi-m4":
        model_cfg_cls = DelphiM4Config
        model_cls = DelphiM4
    else:
        raise ValueError

    model_cfg = model_cfg_cls(**ckpt_dict["model_args"])
    model = model_cls(model_cfg)  # type: ignore
    model.load_state_dict(ckpt_dict["model"])
    model = model.eval()

    with open(ckpt_path / "tokenizer.yaml", "r") as f:
        tokenizer = yaml.safe_load(f)

    return model, train_cfg, tokenizer
