import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime

import torch
from omegaconf import OmegaConf

from delphi.data.dataset import Dataset, UKBDataConfig, load_sequences, train_iter
from delphi.log import TrainLogConfig, TrainLogger
from delphi.model.transformer import Delphi, DelphiConfig


@dataclass
class TrainConfig:
    run_name: str = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    ckpt_dir: str = "./checkpoints"

    eval_interval: int = 2000
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
    seed: int = 42

    gradient_accumulation_steps: int = 1  # used to simulate larger batch sizes
    batch_size: int = 128
    # if gradient_accumulation_steps > 1, this is the micro-batch size

    # system
    device: str = "cpu"
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "float32"
    # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster

    # data
    data_fraction: float = 1.0
    train_data: UKBDataConfig = field(default_factory=UKBDataConfig)
    val_data: UKBDataConfig = field(default_factory=UKBDataConfig)

    model: DelphiConfig = field(default_factory=DelphiConfig)

    log: TrainLogConfig = field(default_factory=TrainLogConfig)

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


def train(cfg: TrainConfig):

    run_dir = os.path.join(cfg.ckpt_dir, cfg.run_name)
    logger = TrainLogger(cfg=cfg.log, exp_cfg=asdict(cfg), dump_dir=run_dir)

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

    train_ds = Dataset(cfg.train_data)
    total_train_size = len(train_ds)
    if cfg.data_fraction < 1.0:
        total_train_size = int(cfg.data_fraction * len(train_ds))
    train_it = train_iter(total_size=total_train_size, batch_size=cfg.batch_size)
    train_loader = load_sequences(it=train_it, dataset=train_ds)

    val_ds = Dataset(cfg.val_data)

    iter_num = 0

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
        # best_val_loss = checkpoint["best_val_loss"]

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
        model = torch.compile(model)  # requires PyTorch 2.0

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.eval_iters, 2)
            ds = train_ds if split == "train" else val_ds
            it = train_iter(
                total_size=len(ds),
                batch_size=cfg.batch_size,
            )
            loader = load_sequences(it=it, dataset=ds)
            for k in range(cfg.eval_iters):

                _, X, T = next(loader)
                X.to(cfg.device)
                T.to(cfg.device)

                X_t0 = X[:, :-1]
                T_t0 = T[:, :-1]
                X_t1 = X[:, 1:]
                T_t1 = T[:, 1:]

                with ctx:
                    logits, loss, _ = model(
                        X_t0, T_t0, X_t1, T_t1, validation_loss_mode=True
                    )
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

    val_loss = None
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss()
            val_loss = losses["val"].sum().item()

            logger.eval_step(
                iter_num=iter_num,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
                addons={
                    "train/loss": losses["train"].sum().item(),
                    "train/loss_ce": losses["train"][0].item(),
                    "train/loss_dt": losses["train"][1].item(),
                },
            )

        if iter_num == 0 and cfg.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16``
        for micro_step in range(cfg.gradient_accumulation_steps):

            _, X, T = next(train_loader)
            X.to(cfg.device)
            T.to(cfg.device)

            X_t0 = X[:, :-1]
            T_t0 = T[:, :-1]
            X_t1 = X[:, 1:]
            T_t1 = T[:, 1:]

            with ctx:
                logits, loss, att = model(X_t0, T_t0, X_t1, T_t1)

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

        logger.train_step(iter_num=iter_num, train_loss=loss, addons={"lr": lr})
        logger.ckpt_step(iter_num=iter_num, model=model, optimizer=optimizer)

        iter_num += 1

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

    train(cfg)  # type: ignore


if __name__ == "__main__":
    main()
