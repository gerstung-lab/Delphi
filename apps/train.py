import math
import os
import time
from datetime import datetime
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from omegaconf import OmegaConf

from delphi.model.transformer import Delphi, DelphiConfig
from delphi.utils import get_batch, get_p2i


@dataclass
class TrainConfig:
    run_name: str = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    ckpt_dir: str = "./checkpoints"
    
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = False
    # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
    seed: int = 42

    # wandb logging
    wandb_log: bool = False  # disabled by default
    wandb_project: str = "delphi"
    wandb_run_name: str = "run" + str(time.time())

    # data
    data_dir: str = "./data"
    dataset: str = "ukb_data"
    gradient_accumulation_steps: int = 1  # used to simulate larger batch sizes
    batch_size: int = 128
    # if gradient_accumulation_steps > 1, this is the micro-batch size

    # model
    model: DelphiConfig = field(default_factory=DelphiConfig)

    # adamw optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 10000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 10000  # should be ~= max_iters per Chinchilla
    min_lr: float = (
        6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    )

    # system
    device: str = "cpu"
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "float32"
    # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster

    # delphi training
    data_fraction: float = 1.0
    no_event_token_rate: int = 5


def train(cfg: TrainConfig):

    run_dir = os.path.join(cfg.ckpt_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in cfg.device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    torch.set_default_dtype(ptdtype)

    dataset_dir = os.path.join(cfg.data_dir, cfg.dataset)
    assert os.path.exists(dataset_dir), f"dataset_dir {dataset_dir} does not exist"
    train_data = np.memmap(
        os.path.join(dataset_dir, "train.bin"), dtype=np.uint32, mode="r"
    ).reshape(-1, 3)
    val_data = np.memmap(
        os.path.join(dataset_dir, "val.bin"), dtype=np.uint32, mode="r"
    ).reshape(-1, 3)

    train_p2i = get_p2i(train_data)
    val_p2i = get_p2i(val_data)

    # downsample the data to requested fraction
    if cfg.data_fraction < 1.0:
        train_p2i = train_p2i[: int(cfg.data_fraction * len(train_p2i))]

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    print(f"found vocab_size = {cfg.model.vocab_size}")

    if cfg.init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        model = Delphi(cfg.model)
    elif cfg.init_from == "resume":
        print(f"Resuming training from {run_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(run_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        model_args = asdict(cfg.model)
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        cfg.model = DelphiConfig(**model_args)
        model = Delphi(cfg.model)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    model.to(cfg.device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type
    )
    if cfg.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])

    # compile the model
    if cfg.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # helps estimate an arbitrarily accurate loss over either split using many batches

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.eval_iters, 2)
            data = train_data if split == "train" else val_data
            p2i = train_p2i if split == "train" else val_p2i
            for k in range(cfg.eval_iters):
                ix = torch.randint(len(p2i), (cfg.batch_size,))
                X, A, Y, B = get_batch(
                    ix,
                    data,
                    p2i,
                    block_size=cfg.model.block_size,
                    device=cfg.device,
                    select="left",
                    no_event_token_rate=cfg.no_event_token_rate,
                    cut_batch=True,
                )
                with ctx:
                    logits, loss, _ = model(X, A, Y, B, validation_loss_mode=True)
                losses[k] = torch.stack([loss["loss_ce"], loss["loss_dt"]])
            out[split] = losses.mean(0)
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < cfg.warmup_iters:
            return cfg.learning_rate * it / cfg.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > cfg.lr_decay_iters:
            return cfg.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    # logging
    if cfg.wandb_log:
        import wandb

        wandb.init(
            project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg)
        )

    # training loop
    ix = torch.randint(len(train_p2i), (cfg.batch_size,))
    X, A, Y, B = get_batch(
        ix,
        train_data,
        train_p2i,
        block_size=cfg.model.block_size,
        device=cfg.device,
        padding="random",
        lifestyle_augmentations=True,
        select="left",
        no_event_token_rate=cfg.no_event_token_rate,
    )
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process

    val_loss = None
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss()
            if val_loss is None:
                val_loss_unpooled = losses["val"]
            val_loss_unpooled = (
                0.1 * losses["val"] + 0.9 * val_loss_unpooled
            )  # ie exponential decay
            val_loss = val_loss_unpooled.sum().item()
            print(
                f"step {iter_num}: train loss {losses['train'].sum().item():.4f}, val loss {losses['val'].sum().item():.4f} ({val_loss:.4f})"
            )
            if cfg.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/agg_loss": losses["train"].sum().item(),
                        "val/loss": val_loss,
                        "val/loss_ce": val_loss_unpooled[0].item(),
                        "val/loss_dt": val_loss_unpooled[1].item(),
                    }
                )

            if cfg.always_save_checkpoint or val_loss < best_val_loss:
                best_val_loss = val_loss
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": asdict(cfg.model),
                        "iter_num": iter_num,
                        "best_val_loss": val_loss,
                        "config": asdict(cfg),
                    }
                    print(f"saving checkpoint to {run_dir}")
                    torch.save(checkpoint, os.path.join(run_dir, "ckpt.pt"))

            if iter_num % 10_000 == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": asdict(cfg.model),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": asdict(cfg),
                }
                print(f"saving checkpoint to {run_dir}")
                torch.save(checkpoint, os.path.join(run_dir, f"ckpt_{iter_num}.pt"))

        if iter_num == 0 and cfg.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(cfg.gradient_accumulation_steps):
            with ctx:
                logits, loss, att = model(X, A, Y, B)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            ix = torch.randint(len(train_p2i), (cfg.batch_size,))
            # print(ix)
            X, A, Y, B = get_batch(
                ix,
                train_data,
                train_p2i,
                block_size=cfg.model.block_size,
                device=cfg.device,
                padding="random",
                lifestyle_augmentations=True,
                select="left",
                no_event_token_rate=cfg.no_event_token_rate,
                cut_batch=True,
            )

            # backward pass, with gradient scaling if training in fp16
            loss = loss["loss_ce"] + loss["loss_dt"]
            scaler.scale(loss).backward()
        # clip the gradient
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

            if cfg.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": loss,
                        "lr": lr,
                        "weights": wandb.Histogram(
                            model.transformer.embed.token_embedding.weight.cpu()
                            .detach()
                            .numpy()
                        ),
                        "logits": wandb.Histogram(logits.cpu().detach().numpy()),
                    }
                )

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > cfg.max_iters:
            break


def main():

    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainConfig())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    # convert structured config back to underlying dataclass
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
