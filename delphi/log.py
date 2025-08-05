import gc
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import wandb
import yaml
from omegaconf import OmegaConf


@dataclass
class GenLogConfig:
    save_tokens: bool = True
    save_logits: bool = False
    flush_interval: int = 10
    wandb_log: bool = True
    wandb_project: str = "delphi"
    wandb_run_name: str = "gen" + str(datetime.now().strftime("%Y-%m-%d-%H%M%S"))


class GenLogger:
    def __init__(
        self,
        cfg: GenLogConfig,
        dump_dir: str,
    ):
        self.cfg = cfg
        self.dump_dir = dump_dir

        self.wandb = self.cfg.wandb_log
        if self.wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
            )

        self.save_tokens = self.cfg.save_tokens
        self.token_path = os.path.join(self.dump_dir, "gen.bin")
        self.token_dtype = np.uint32
        self.save_logits = self.cfg.save_logits
        self.logit_path = os.path.join(self.dump_dir, "logits.bin")
        self.logit_dtype = np.float16
        if self.save_logits:
            assert self.save_tokens
        self.flush_freq = self.cfg.flush_interval

        self.step = 0
        self.offset = 0

    def init_memmaps(self, n_max_token: int, n_vocab: int):

        if self.save_tokens:
            self.n_max_token = n_max_token
            self.gen_bin = np.memmap(
                self.token_path,
                dtype=self.token_dtype,
                mode="w+",
                shape=(int(n_max_token), 3),
            )
            if self.save_logits:
                self.n_vocab = n_vocab
                self.logits_bin = np.memmap(
                    self.logit_path,
                    dtype=self.logit_dtype,
                    mode="w+",
                    shape=(int(n_max_token), n_vocab),
                )

    def flush_memmaps(self):
        if self.save_tokens:
            self.gen_bin.flush()
            if self.save_logits:
                self.logits_bin.flush()

            if self.step % (self.flush_freq * 10) == 0:
                self.delete_memmaps()
                gc.collect()
                self.reopen_memmaps()

    def delete_memmaps(self):
        if hasattr(self, "gen_bin"):
            del self.gen_bin
        if hasattr(self, "logits_bin"):
            del self.logits_bin

    def reopen_memmaps(self):
        if self.save_tokens:
            self.gen_bin = np.memmap(
                self.token_path,
                dtype=self.token_dtype,
                mode="r+",
                shape=(self.n_max_token, 3),
            )
            if self.save_logits:
                self.logits_bin = np.memmap(
                    self.logit_path,
                    dtype=self.logit_dtype,
                    mode="r+",
                    shape=(self.n_max_token, self.n_vocab),
                )

    def write_memmaps(
        self,
        participants: np.ndarray,
        tokens: np.ndarray,
        timesteps: np.ndarray,
        logits: np.ndarray,
    ):

        if not hasattr(self, "gen_bin"):
            raise ValueError("memmaps not initialized; call init_memmaps first")

        if self.save_tokens:
            sub_idx, pos_idx = np.nonzero(timesteps != -1e4)
            packed_batch = np.stack(
                (
                    participants[sub_idx],
                    timesteps[sub_idx, pos_idx],
                    tokens[sub_idx, pos_idx],
                ),
                axis=-1,
            )
            token_n = packed_batch.shape[0]
            self.gen_bin[self.offset : self.offset + token_n, :] = packed_batch
            if self.save_logits:
                packed_logits = logits[..., sub_idx, pos_idx, :]
                self.logits_bin[self.offset : self.offset + token_n, :] = packed_logits

        if self.step % self.flush_freq == 0:
            self.flush_memmaps()

        self.step += 1
        self.offset += token_n

    def close(self):
        if self.save_tokens:
            self.gen_bin.flush()
            del self.gen_bin
            with open(self.token_path, "r+b") as f:
                f.truncate(self.offset * 3 * np.dtype(self.token_dtype).itemsize)
            if self.save_logits:
                self.logits_bin.flush()
                del self.logits_bin
                with open(self.logit_path, "r+b") as f:
                    f.truncate(
                        self.offset * self.n_vocab * np.dtype(self.logit_dtype).itemsize
                    )


@dataclass
class TrainLogConfig:
    wandb_log: bool = True
    wandb_project: str = "delphi"
    run_name: str = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    always_ckpt_after_eval: bool = False
    ckpt_interval: Optional[int] = 1000
    log_interval: int = 1


class TrainLogger:
    def __init__(
        self,
        cfg: TrainLogConfig,
        exp_cfg: dict,
        dump_dir: str,
        model: torch.nn.Module,
        tokenizer: dict,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
    ):
        self.cfg = cfg
        self.wandb = self.cfg.wandb_log
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        if self.wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.run_name,
                config=exp_cfg,
            )

            wandb.define_metric("step")
            wandb.define_metric("lr", step_metric="step")
            wandb.define_metric("val/loss", step_metric="step")
            wandb.define_metric("train/loss", step_metric="step")
            self.addon_metrics = set()

            wandb.summary["model_params"] = sum(p.numel() for p in model.parameters())
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    wandb.define_metric(f"grad_norm/{name}", step_metric="step")

        self.exp_cfg = exp_cfg

        self.dump_dir = dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config=exp_cfg, f=f)

        with open(os.path.join(dump_dir, "tokenizer.yaml"), "w") as f:
            yaml.dump(
                tokenizer,
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        self.best_val_loss = float("inf")

    def save_ckpt(
        self,
        step: int,
        ckpt_fname: str = "ckpt.pt",
    ):

        checkpoint = {
            "model": self.model.state_dict(),
            "model_type": self.model.model_type,
            "optimizer": self.optimizer.state_dict(),
            "model_args": self.exp_cfg["model"],
            "iter_num": step,
            "best_val_loss": self.best_val_loss,
            "config": self.exp_cfg,
        }
        ckpt_path = os.path.join(self.dump_dir, ckpt_fname)
        print(f"saving checkpoint to {ckpt_path}")
        torch.save(checkpoint, ckpt_path)

    def eval_step(self, step: int, loss: dict[str, float]):

        lossf = 0.0
        log_dict = {"step": step}
        for loss_key, loss_pt in loss.items():
            metric = f"val/{loss_key}"
            log_dict[metric] = loss_pt  # type: ignore
            if metric not in self.addon_metrics:
                wandb.define_metric(metric, step_metric="step")
                self.addon_metrics.add(metric)
            lossf += loss_pt
        log_dict["val/loss"] = lossf  # type: ignore

        print(f"iter {step}: val loss {lossf:.4f}")
        if self.cfg.always_ckpt_after_eval or lossf < self.best_val_loss:
            self.save_ckpt(step, ckpt_fname="ckpt.pt")

        if self.wandb:
            wandb.log(log_dict)

        self.best_val_loss = min(lossf, self.best_val_loss)

    def log_grad(self):
        if self.cfg.wandb_log:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    wandb.log(
                        {
                            f"grad_norm/{name}": param.grad.norm().item(),
                        },
                        commit=False,
                    )

    def train_step(
        self,
        step: int,
        loss: dict[str, torch.Tensor],
    ):
        lossf = 0.0
        log_dict = {
            "step": step,
            "lr": self.scheduler.get_last_lr()[0],
        }
        if step % self.cfg.log_interval == 0:
            for loss_key, loss_pt in loss.items():
                metric = f"train/{loss_key}"
                log_dict[metric] = loss_pt.item()
                if metric not in self.addon_metrics:
                    wandb.define_metric(metric, step_metric="step")
                    self.addon_metrics.add(metric)
                lossf += loss_pt.item()
            log_dict["train/loss"] = lossf

            print(f"iter {step}: loss {lossf:.4f}")
            if self.wandb:
                wandb.log(log_dict)
                self.log_grad()

    def ckpt_step(
        self,
        step: int,
    ):
        if self.cfg.ckpt_interval is None:
            return
        if step % self.cfg.ckpt_interval == 0 and step > 0:
            self.save_ckpt(step, ckpt_fname=f"ckpt_{step}.pt")
