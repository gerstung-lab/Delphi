import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch
from omegaconf import OmegaConf

from delphi import distributed
from delphi.baselines import ethos, motor
from delphi.config import dataclass_from_dict
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment.config import TrainBaseConfig
from delphi.log import TrainLogger
from delphi.model import delphi
from delphi.model.transformer import Delphi, DelphiConfig
from delphi.optim import configure_optimizers
from delphi.tokenizer import Tokenizer, load_tokenizer_from_yaml


class BaseTrainer:

    def __init__(
        self,
        cfg: TrainBaseConfig,
        model: torch.nn.Module,
        train_ds: Any,
        val_ds: Any,
        loader: Callable,
    ):
        self.backend = distributed.make_backend_from_args(cfg)
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

        self.model = model
        self.model.to(self.device)
        self.model = self.backend.transform_model(self.model)

        self.optimizer, self.scheduler = configure_optimizers(
            model=model, cfg=cfg.optim, device_type=self.device_type
        )
        self.scaler = torch.GradScaler(
            device=self.device_type, enabled=(cfg.dtype == "float16")
        )

        self.train_loader = loader(
            seed=cfg.seed, dataset=train_ds, batch_size=cfg.batch_size
        )
        self.estimate_loaders = {
            "train": loader(seed=cfg.seed, dataset=train_ds, batch_size=cfg.batch_size),
            "val": loader(seed=cfg.seed, dataset=val_ds, batch_size=cfg.batch_size),
        }

        run_dir = os.path.normpath(
            os.path.join(DELPHI_CKPT_DIR, cfg.ckpt_dir, cfg.log.run_name)
        )
        self.logger = TrainLogger(
            cfg=cfg.log,
            exp_cfg=asdict(cfg),
            dump_dir=run_dir,
            tokenizer=train_ds.tokenizer.to_dict(),
            model=model,  # type: ignore
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            backend=self.backend,
        )

        self.iter_num = 0

    def mini_step(self, loader: Iterator, *args, **kwargs) -> dict[str, torch.Tensor]:
        X, T = next(loader)
        X, T = X.to(self.device), T.to(self.device)
        X_t0, X_t1 = X[:, :-1], X[:, 1:]
        T_t0, T_t1 = T[:, :-1], T[:, 1:]

        with self.ctx:
            _, loss = self.model(idx=X_t0, targets=X_t1, age=T_t0, targets_age=T_t1)

        return loss

    @torch.no_grad()
    def estimate_loss(
        self, loaders: dict[str, Iterator], *args, **kwargs
    ) -> tuple[dict, dict]:
        self.model.eval()
        eval_loss = {}
        for split in ["train", "val"]:
            split_loss = defaultdict(float)
            for _ in range(self.cfg.eval_iters):
                loss = self.mini_step(loader=loaders[split])
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
                _, val_loss = self.estimate_loss(self.estimate_loaders)
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
                    loss = self.mini_step(loader=self.train_loader)

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
    if model_type == "delphi":
        model_cfg_cls = delphi.ModelConfig
        model_cls = delphi.Model
    elif model_type == "delphi-m4":
        model_cfg_cls = DelphiConfig
        model_cls = Delphi
    elif model_type == "ethos":
        model_cfg_cls = ethos.ModelConfig
        model_cls = ethos.Model
    elif model_type == "motor":
        model_cfg_cls = motor.ModelConfig
        model_cls = motor.Model
    else:
        raise ValueError

    model_cfg = dataclass_from_dict(
        model_cfg_cls, ckpt_dict["model_args"], strict=False
    )
    model = model_cls(model_cfg)  # type: ignore
    model.load_state_dict(ckpt_dict["model"])
    model = model.eval()

    tokenizer = load_tokenizer_from_yaml(ckpt_path / "tokenizer.yaml")

    return model, train_cfg, tokenizer
