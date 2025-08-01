import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from delphi.data.core import train_iter
from delphi.data.multimodal import (
    M4Dataset,
    UKBDataConfig,
    load_sequences,
    pad_trailing_biomarkers,
)
from delphi.env import DELPHI_CKPT_DIR
from delphi.log import TrainLogConfig, TrainLogger
from delphi.model.config import (
    DelphiConfig,
    parse_token_list,
    validate_model_config,
    validate_model_config_for_finetuning,
)
from delphi.model.transformer import Delphi
from delphi.optim import OptimConfig, configure_optimizers


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
    device: str = "cpu"
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "float32"
    # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster

    train_data: dict = field(default_factory=dict)
    val_data: dict = field(default_factory=dict)

    model: dict = field(default_factory=dict)

    optim: OptimConfig = field(default_factory=OptimConfig)

    log: TrainLogConfig = field(default_factory=TrainLogConfig)


@dataclass
class TrainConfig(TrainBaseConfig):
    # finetune
    resume_from: Optional[str] = None

    # data
    data_fraction: float = 1.0
    memmap: bool = False
    train_data: UKBDataConfig = field(default_factory=UKBDataConfig)
    infer_train_biomarkers: bool = True
    val_data: UKBDataConfig = field(default_factory=UKBDataConfig)
    infer_val_biomarkers: bool = True
    infer_val_expansion_packs: bool = True
    infer_val_transforms: bool = True
    infer_val_subject_filters: bool = True

    model: DelphiConfig = field(default_factory=DelphiConfig)
    ignore_expansion_tokens: bool = True


def train(cfg: TrainConfig):

    validate_model_config(cfg.model)

    run_dir = os.path.normpath(
        os.path.join(DELPHI_CKPT_DIR, cfg.ckpt_dir, cfg.log.run_name)
    )
    os.makedirs(run_dir, exist_ok=True)

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
        else torch.autocast(device_type=device_type, dtype=ptdtype)
    )

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html
    torch.set_default_dtype(ptdtype)

    if cfg.infer_train_biomarkers:
        cfg.train_data.biomarkers = list(cfg.model.biomarkers.keys())
    assert set(cfg.train_data.biomarkers).issubset(set(cfg.model.biomarkers.keys()))
    print(f"using memmap mode: {cfg.memmap}")
    print("training dataset")
    train_ds = M4Dataset(cfg=cfg.train_data, memmap=cfg.memmap)
    tokenizer = train_ds.tokenizer
    tokenizer.save_to_yaml(os.path.join(run_dir, "tokenizer.yaml"))
    train_it = train_iter(rng=rng, total_size=len(train_ds), batch_size=cfg.batch_size)
    train_loader = load_sequences(it=train_it, dataset=train_ds)

    if cfg.infer_val_biomarkers:
        cfg.val_data.biomarkers = cfg.train_data.biomarkers
    assert set(cfg.val_data.biomarkers).issubset(set(cfg.train_data.biomarkers))
    if cfg.infer_val_expansion_packs:
        cfg.val_data.expansion_packs = cfg.train_data.expansion_packs
    assert set(cfg.val_data.expansion_packs).issubset(
        set(cfg.train_data.expansion_packs)
    )
    if cfg.infer_val_subject_filters:
        cfg.val_data.must_have_biomarkers = cfg.train_data.must_have_biomarkers
    if cfg.infer_val_transforms:
        cfg.val_data.transforms = cfg.train_data.transforms
    print("validation dataset")
    val_ds = M4Dataset(cfg=cfg.val_data, memmap=cfg.memmap)

    loaders_for_loss_estimates = {
        "train": load_sequences(
            it=train_iter(
                rng=np.random.default_rng(cfg.seed),
                total_size=len(train_ds),
                batch_size=cfg.batch_size,
            ),
            dataset=train_ds,
        ),
        "val": load_sequences(
            it=train_iter(
                rng=np.random.default_rng(cfg.seed),
                total_size=len(val_ds),
                batch_size=cfg.batch_size,
            ),
            dataset=val_ds,
        ),
    }

    if cfg.model.vocab_size is None:
        cfg.model.vocab_size = train_ds.vocab_size
        print(
            f"\nvocab_size not set, using vocab size of training dataset's tokenizer: {cfg.model.vocab_size}"
        )
    else:
        print(f"vocab_size: {cfg.model.vocab_size}")
        assert (
            cfg.model.vocab_size == train_ds.vocab_size
        ), f"inconsistent vocab size between tokenizer ({train_ds.vocab_size}) and model"

    print(f"ignored tokens:")
    print(f"\t- {cfg.model.ignore_tokens}")
    ignore_tokens = parse_token_list(cfg.model.ignore_tokens)
    if cfg.ignore_expansion_tokens:
        print(f"\t- all expansion pack tokens")
        ignore_tokens = set(ignore_tokens).union(set(train_ds.expansion_tokens))
        ignore_tokens = list(ignore_tokens)
    cfg.model.ignore_tokens = train_ds.tokenizer.encode(ignore_tokens)  # type: ignore

    if cfg.model.loss.motor:
        motor_task_tokens = parse_token_list(cfg.model.loss.motor_task_tokens)
        cfg.model.loss.motor_task_tokens = train_ds.tokenizer.encode(motor_task_tokens)  # type: ignore

    iter_num = 0

    if cfg.init_from == "scratch":
        print("Initializing a new model from scratch")
        model = Delphi(cfg.model)
    elif cfg.init_from == "finetune":
        assert cfg.resume_from is not None
        print(f"Finetuning pretrained model at {cfg.resume_from}")
        checkpoint = torch.load(
            os.path.join(DELPHI_CKPT_DIR, cfg.resume_from, "ckpt.pt"),
            map_location=cfg.device,
        )
        validate_model_config_for_finetuning(
            finetune_config=cfg.model,
            pretrain_config=DelphiConfig(**checkpoint["model_args"]),
        )
        model = Delphi(cfg.model)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        existing_keys = set(state_dict.keys())
        model_keys = set(model.state_dict().keys())
        new_keys = model_keys - existing_keys
        print(f"new layers: {new_keys}")
        model.load_state_dict(state_dict, strict=False)
        trainable_layers = []
        for name, param in model.named_parameters():
            if name in existing_keys:
                param.requires_grad = False
            else:
                trainable_layers.append(name)
                param.requires_grad = True
        print(f"trainable layers: {trainable_layers}")

    model.to(cfg.device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.GradScaler(device=device_type, enabled=(cfg.dtype == "float16"))

    optimizer, scheduler = configure_optimizers(
        model=model, cfg=cfg.optim, device_type=device_type
    )

    if cfg.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    logger = TrainLogger(
        cfg=cfg.log,
        exp_cfg=asdict(cfg),
        dump_dir=run_dir,
        model=model,  # type: ignore
        optimizer=optimizer,
        scheduler=scheduler,
    )

    def mini_step(
        loader: Iterator, validation_loss_mode: bool = False
    ) -> dict[str, torch.Tensor]:

        _, X, T, M, biomarker_X = next(loader)
        X, T, M = pad_trailing_biomarkers(X, T, M)
        X, T, M = X.to(cfg.device), T.to(cfg.device), M.to(cfg.device)
        biomarker_X = {k: v.to(cfg.device) for k, v in biomarker_X.items()}

        X_t0, X_t1 = X[:, :-1], X[:, 1:]
        T_t0, T_t1 = T[:, :-1], T[:, 1:]
        M_t0, _ = M[:, :-1], M[:, 1:]

        with ctx:
            _, loss, _ = model(
                idx=X_t0,
                targets=X_t1,
                modality=M_t0,
                age=T_t0,
                targets_age=T_t1,
                biomarker=biomarker_X,
                validation_loss_mode=validation_loss_mode,
            )

        return loss

    @torch.no_grad()
    def estimate_loss(loaders: dict[str, Iterator]):
        model.eval()
        eval_loss = {}
        for split in ["train", "val"]:
            split_loss = defaultdict(float)
            for _ in range(cfg.eval_iters):
                loss = mini_step(loader=loaders[split], validation_loss_mode=True)
                for key in loss.keys():
                    split_loss[key] += loss[key].item()
            split_loss = dict(split_loss)
            for key in split_loss.keys():
                split_loss[key] /= cfg.eval_iters
            eval_loss[split] = split_loss

        model.train()

        return eval_loss["train"], eval_loss["val"]

    while True:

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.eval_interval == 0 and iter_num > 0:
            _, val_loss = estimate_loss(loaders_for_loss_estimates)
            logger.eval_step(step=iter_num, loss=val_loss)

        if iter_num == 0 and cfg.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16``
        for _ in range(cfg.gradient_accumulation_steps):
            loss = mini_step(loader=train_loader)

            # backward pass, with gradient scaling if training in fp16
            loss_agg = sum([loss[key] for key in loss.keys()])
            scaler.scale(loss_agg).backward()  # type: ignore

        # clip the gradient
        if cfg.optim.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)

        logger.train_step(step=iter_num, loss=loss)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        logger.ckpt_step(step=iter_num)

        iter_num += 1

        # termination conditions
        if iter_num > cfg.optim.max_iters:
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
