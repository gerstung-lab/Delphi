import gc
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import wandb
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
            sub_idx, pos_idx = np.nonzero(tokens > 0)
            packed_batch = np.stack(
                (
                    participants[sub_idx],
                    tokens[sub_idx, pos_idx],
                    timesteps[sub_idx, pos_idx],
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
    wandb_run_name: str = "gen" + str(datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    always_ckpt_after_eval: bool = False
    ckpt_interval: int = 1000
    log_interval: int = 1


class TrainLogger:
    def __init__(
        self,
        cfg: TrainLogConfig,
        exp_cfg: dict,
        dump_dir: str,
    ):
        self.cfg = cfg
        self.wandb = self.cfg.wandb_log
        if self.wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
                config=exp_cfg,
            )

        self.exp_cfg = exp_cfg

        self.dump_dir = dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config=exp_cfg, f=f)

        self.best_val_loss = float("inf")

    def save_ckpt(
        self,
        iter_num: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ckpt_fname: str = "ckpt.pt",
    ):

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_args": self.exp_cfg["model"],
            "iter_num": iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.exp_cfg,
        }
        ckpt_path = os.path.join(self.dump_dir, ckpt_fname)
        print(f"saving checkpoint to {ckpt_path}")
        torch.save(checkpoint, ckpt_path)

    def eval_step(
        self,
        iter_num: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        val_loss: float,
        addons: Optional[dict] = None,
    ):

        print(f"step {iter_num}: val loss {val_loss:.4f}")
        if self.cfg.always_ckpt_after_eval or val_loss < self.best_val_loss:
            self.save_ckpt(iter_num, model, optimizer, ckpt_fname="ckpt.pt")

        if self.wandb:
            log_dict = {"iter": iter_num, "val/loss": val_loss}
            if addons is not None:
                log_dict.update(addons)
            wandb.log(log_dict)

        self.best_val_loss = min(val_loss, self.best_val_loss)

    def train_step(
        self,
        iter_num: int,
        train_loss: torch.Tensor,
        addons: Optional[dict] = None,
    ):
        if iter_num % self.cfg.log_interval == 0:
            lossf = train_loss.item()
            print(f"iter {iter_num}: loss {lossf:.4f}")
            if self.wandb:
                log_dict = {"iter": iter_num, "train/loss": lossf}
                if addons is not None:
                    log_dict.update(addons)
                wandb.log(log_dict)

    def ckpt_step(
        self,
        model,
        optimizer,
        iter_num: int,
    ):
        if self.cfg.ckpt_interval is None:
            return
        if iter_num % self.cfg.ckpt_interval == 0:
            self.save_ckpt(iter_num, model, optimizer, ckpt_fname=f"ckpt_{iter_num}.pt")
