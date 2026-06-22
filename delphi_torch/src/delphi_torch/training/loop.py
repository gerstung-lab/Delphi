"""Training loop for Delphi model.

This module provides a clean, modular training loop with:
- Separated concerns (setup, training step, evaluation, logging)
- Clear optimizer step sequence
- Optional wandb integration
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn

from delphi_torch.config import AppConfig, TrainConfig
from delphi_torch.data.dataloaders import build_dataloaders
from delphi_torch.models.delphi import DelphiModel
from delphi_torch.training.losses import compute_losses


# =============================================================================
# Setup helpers
# =============================================================================

def _set_tf32() -> None:
    """Enable TF32 for faster matmul on Ampere+ GPUs."""
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        elif hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        if hasattr(torch.backends.cudnn, "conv") and hasattr(
            torch.backends.cudnn.conv, "fp32_precision"
        ):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        elif hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True


def _init_wandb(cfg: TrainConfig):
    """Initialize wandb if enabled. Returns wandb run or None."""
    if not cfg.wandb_log:
        return None
    
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed. Install it with: pip install wandb"
        ) from exc
    
    import datetime as _dt
    
    run_name = cfg.wandb_run_name
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if "{timestamp}" in run_name:
        run_name = run_name.replace("{timestamp}", timestamp)
    elif run_name in {"", "run"}:
        run_name = f"run-{timestamp}"
    
    return wandb.init(project=cfg.wandb_project, name=run_name)


@dataclass
class TrainingContext:
    """Holds all training state and components."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    train_iter: Iterator
    val_loader: torch.utils.data.DataLoader
    device: torch.device
    autocast_ctx: nullcontext
    cfg: AppConfig
    wandb_run: object | None = None


def _setup_context(cfg: AppConfig) -> TrainingContext:
    """Create model, optimizer, dataloaders, and training context."""
    # Device and dtype
    device = torch.device(cfg.train.device)
    device_type = "cuda" if "cuda" in cfg.train.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.train.dtype]
    
    autocast_ctx = (
        nullcontext() if device_type == "cpu" 
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    
    # Model
    model = DelphiModel(cfg.model)
    model.to(device)
    if cfg.train.compile:
        model = torch.compile(model)
    
    # Optimizer and scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=(cfg.train.beta1, cfg.train.beta2),
        weight_decay=cfg.train.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.dtype == "float16"))
    
    # Data
    train_loader, val_loader = build_dataloaders(cfg.data)
    train_iter = cycle(train_loader)
    
    # Wandb
    wandb_run = _init_wandb(cfg.train)
    if wandb_run is not None:
        wandb_run.config.update(cfg.model_dump())
    
    return TrainingContext(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_iter=train_iter,
        val_loader=val_loader,
        device=device,
        autocast_ctx=autocast_ctx,
        cfg=cfg,
        wandb_run=wandb_run,
    )


# =============================================================================
# Learning rate schedule
# =============================================================================

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Compute learning rate with linear warmup and cosine decay."""
    # Linear warmup
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    
    # After decay period, return minimum
    if step > cfg.lr_decay_steps:
        return cfg.min_lr
    
    # Cosine decay
    decay_ratio = (step - cfg.warmup_steps) / (cfg.lr_decay_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Update learning rate in optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# =============================================================================
# Training step
# =============================================================================

def train_step(ctx: TrainingContext, step: int) -> dict[str, float]:
    """
    Execute one training step.
    
    Sequence:
    1. Get batch and move to device
    2. Update learning rate
    3. Forward pass (with autocast)
    4. Backward pass (with gradient scaling)
    5. Gradient clipping (optional)
    6. Optimizer step
    7. Zero gradients
    
    Returns loss dict with 'loss', 'loss_ce', 'loss_dt'.
    """
    cfg = ctx.cfg
    
    # 1. Get batch
    x, a, y, b = next(ctx.train_iter)
    x = x.to(ctx.device, non_blocking=True)
    a = a.to(ctx.device, non_blocking=True)
    y = y.to(ctx.device, non_blocking=True)
    b = b.to(ctx.device, non_blocking=True)
    
    # 2. Update learning rate
    lr = get_lr(step, cfg.train)
    set_lr(ctx.optimizer, lr)
    
    # 3. Forward pass
    with ctx.autocast_ctx:
        logits, attn_mask, _ = ctx.model(x, a, targets_age=b)
        loss_dict = compute_losses(
            logits,
            idx=x, age=a, targets=y, targets_age=b,
            attn_mask=attn_mask,
            cfg=cfg.model,
            validation=False,
        )
        loss = loss_dict["loss"]
    
    # 4. Backward pass
    ctx.scaler.scale(loss).backward()
    
    # 5. Gradient clipping
    if cfg.train.grad_clip > 0:
        ctx.scaler.unscale_(ctx.optimizer)
        torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), cfg.train.grad_clip)
    
    # 6. Optimizer step
    ctx.scaler.step(ctx.optimizer)
    ctx.scaler.update()
    
    # 7. Zero gradients
    ctx.optimizer.zero_grad(set_to_none=True)
    
    return {
        "loss": loss.item(),
        "loss_ce": loss_dict["loss_ce"].item(),
        "loss_dt": loss_dict["loss_dt"].item(),
        "lr": lr,
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(ctx: TrainingContext) -> dict[str, float]:
    """Evaluate model on validation set."""
    ctx.model.eval()
    losses = []
    cfg = ctx.cfg
    
    with torch.no_grad():
        for i, batch in enumerate(ctx.val_loader):
            x, a, y, b = [t.to(ctx.device, non_blocking=True) for t in batch]
            logits, attn_mask, _ = ctx.model(x, a, targets_age=b)
            loss_dict = compute_losses(
                logits,
                idx=x, age=a, targets=y, targets_age=b,
                attn_mask=attn_mask,
                cfg=cfg.model,
                validation=True,
            )
            losses.append((loss_dict["loss_ce"].item(), loss_dict["loss_dt"].item()))
            if i + 1 >= cfg.train.eval_iters:
                break
    
    ctx.model.train()
    
    if not losses:
        return {"loss_ce": float("nan"), "loss_dt": float("nan"), "loss": float("nan")}
    
    loss_ce = sum(x[0] for x in losses) / len(losses)
    loss_dt = sum(x[1] for x in losses) / len(losses)
    return {"loss_ce": loss_ce, "loss_dt": loss_dt, "loss": loss_ce + loss_dt}


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(path: Path, ctx: TrainingContext, step: int) -> None:
    """Save model and optimizer state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": ctx.model.state_dict(),
            "optimizer": ctx.optimizer.state_dict(),
            "step": step,
            "config": ctx.cfg.model_dump(),
        },
        path,
    )


# =============================================================================
# Logging
# =============================================================================

def log_train(step: int, metrics: dict[str, float], wandb_run, iter_time_ms: float | None = None) -> None:
    """Log training metrics."""
    time_str = f", time {iter_time_ms:.2f}ms" if iter_time_ms is not None else ""
    print(
        f"iter {step}: loss {metrics['loss']:.4f} "
        f"(ce {metrics['loss_ce']:.4f}, dt {metrics['loss_dt']:.4f}){time_str}"
    )
    if wandb_run is not None:
        log_dict = {
            "step": step,
            "train/loss": metrics["loss"],
            "train/loss_ce": metrics["loss_ce"],
            "train/loss_dt": metrics["loss_dt"],
            "train/lr": metrics["lr"],
        }
        if iter_time_ms is not None:
            log_dict["train/iter_time_ms"] = iter_time_ms
        wandb_run.log(log_dict)


def log_eval(step: int, metrics: dict[str, float], lr: float, wandb_run) -> None:
    """Log evaluation metrics."""
    print(
        f"step {step}: val loss {metrics['loss']:.4f} "
        f"(ce {metrics['loss_ce']:.4f}, dt {metrics['loss_dt']:.4f}) "
        f"lr {lr:.6f}"
    )
    if wandb_run is not None:
        wandb_run.log({
            "step": step,
            "val/loss": metrics["loss"],
            "val/loss_ce": metrics["loss_ce"],
            "val/loss_dt": metrics["loss_dt"],
            "lr": lr,
        })


# =============================================================================
# Main training loop
# =============================================================================

def fit(cfg: AppConfig) -> None:
    """
    Train the Delphi model.
    
    Training loop structure:
    1. Setup: model, optimizer, data, wandb
    2. For each step:
       a. Evaluate (if at eval_interval)
       b. Check early stopping
       c. Train step
       d. Log (if at log_interval)
    """
    import time
    
    torch.manual_seed(cfg.train.seed)
    _set_tf32()
    
    # Setup
    ctx = _setup_context(cfg)
    
    # Training state
    best_val_loss = float("inf")
    no_improve_count = 0
    t0 = time.time()
    
    # Training loop
    for step in range(cfg.train.max_steps + 1):
        lr = get_lr(step, cfg.train)
        
        # --- Evaluation ---
        if step > 0 and step % cfg.train.eval_interval == 0:
            val_metrics = evaluate(ctx)
            log_eval(step, val_metrics, lr, ctx.wandb_run)
            
            # Check for improvement
            improved = val_metrics["loss"] < (best_val_loss - cfg.train.early_stop_min_delta)
            if improved:
                best_val_loss = val_metrics["loss"]
                no_improve_count = 0
                save_checkpoint(cfg.train.out_dir / "ckpt.pt", ctx, step)
            else:
                no_improve_count += 1
            
            # Early stopping
            if (
                cfg.train.early_stop
                and step >= cfg.train.early_stop_min_steps
                and no_improve_count >= cfg.train.early_stop_patience
            ):
                print(f"Early stopping at step {step} (no improvement for {no_improve_count} evals)")
                break
        
        # --- Training step ---
        train_metrics = train_step(ctx, step)
        
        # --- Timing ---
        t1 = time.time()
        iter_time_ms = (t1 - t0) * 1000
        t0 = t1
        
        # --- Logging ---
        if step % cfg.train.log_interval == 0:
            log_train(step, train_metrics, ctx.wandb_run, iter_time_ms)
