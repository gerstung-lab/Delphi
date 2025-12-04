import os
from dataclasses import dataclass
from datetime import datetime
from pprint import pprint

import torch
import wandb

from delphi import distributed


@dataclass
class TrainLogConfig:
    wandb_log: bool = True
    wandb_project: str = "delphi"
    run_name: None | str = None
    ckpt_interval: None | int = None
    log_interval: int = 250

    def __post_init__(self):
        # using post_init pattern for the following use cases:
        # python run.py config=config.yaml log.run_name=null
        # config.yaml defines run_name and is overridden with datetime
        if self.run_name == None:
            self.run_name = datetime.now().strftime("%Y-%m-%d-%H%M%S")


class TrainLogger:
    def __init__(
        self,
        cfg: TrainLogConfig,
        exp_cfg: dict,
        data_cfg: dict,
        dump_dir: str,
        model: torch.nn.Module,
        tokenizer: dict,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        backend: distributed.backend.DistributedBackend,
    ):
        self.cfg = cfg
        self.exp_cfg = exp_cfg
        self.data_cfg = data_cfg
        self.wandb = self.cfg.wandb_log
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_val_loss = float("inf")
        self.dump_dir = dump_dir
        self.backend = backend

        if backend.is_master_process():

            print("=== config ===")
            pprint(exp_cfg, indent=2, width=60)

            n_params = sum(p.numel() for p in model.parameters())
            print("number of model parameters: %.2fM" % (n_params / 1e6,))

            if self.wandb:
                wandb.init(
                    project=self.cfg.wandb_project,
                    name=self.cfg.run_name,
                    config=exp_cfg,
                )
                wandb.define_metric("step")
                wandb.define_metric("lr", step_metric="step")
                wandb.define_metric("val/*", step_metric="step")
                wandb.define_metric("train/*", step_metric="step")
                wandb.define_metric("grad_norm/*", step_metric="step")

                wandb.summary["model_params"] = n_params

            os.makedirs(dump_dir, exist_ok=True)

    def save_ckpt(
        self,
        step: int,
        ckpt_fname: str = "ckpt.pt",
    ):
        if self.backend.is_master_process():
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model = self.model.module
            else:
                model = self.model
            checkpoint = {
                "model": model.state_dict(),
                "model_type": model.model_type,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "model_args": self.exp_cfg["model"],
                "data_args": self.data_cfg,
                "iter_num": step,
                "best_val_loss": self.best_val_loss,
                "config": self.exp_cfg,
                "tokenizer": self.tokenizer,
            }
            ckpt_path = os.path.join(self.dump_dir, ckpt_fname)
            print(f"saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)

    def eval_step(self, step: int, loss: dict[str, float]):
        if self.backend.is_master_process():
            lossf = 0.0
            log_dict = {"step": step}
            for loss_key, loss_pt in loss.items():
                metric = f"val/{loss_key}"
                log_dict[metric] = loss_pt  # type: ignore
                lossf += loss_pt
            log_dict["val/loss"] = lossf  # type: ignore
            if self.wandb:
                wandb.log(log_dict)

            print(f"iter {step}: val loss {lossf:.4f}")
            self.save_ckpt(step, ckpt_fname="ckpt.pt")
            if lossf < self.best_val_loss:
                self.save_ckpt(step, ckpt_fname="ckpt_best.pt")
            self.best_val_loss = min(lossf, self.best_val_loss)

    def log_grad(self):
        if self.backend.is_master_process():
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
        if self.backend.is_master_process():
            if step % self.cfg.log_interval == 0:
                log_dict = {
                    "step": step,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                lossf = 0.0
                for loss_key, loss_pt in loss.items():
                    metric = f"train/{loss_key}"
                    log_dict[metric] = loss_pt.item()
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
        if self.backend.is_master_process():
            if self.cfg.ckpt_interval is None:
                return
            if step % self.cfg.ckpt_interval == 0 and step > 0:
                self.save_ckpt(step, ckpt_fname=f"ckpt_{step}.pt")
