import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from delphi.baselines import ethos
from delphi.data.ukb import UKBDataset
from delphi.data.utils import eval_iter, move_batch_to_device
from delphi.model.components import target_mask
from delphi.train import load_ckpt


def estimate_loss(
    ckpt,
    device: str = "cuda",
    batch_size: int = 1024,
    data_dir: str = "ukb_real_data",
    subject_list: str = "participants/val_fold.bin",
):

    model, cfg, tokenizer = load_ckpt(ckpt)
    model.to(device)
    model.eval()

    common_args = {
        "data_dir": data_dir,
        "subject_list": subject_list,
        "block_size": model.config.block_size,
        "no_event_interval": cfg.data["no_event_interval"],
        "memmap": False,
    }
    if model.model_type == "ethos":
        with open(cfg.data["time_bins"], "r") as f:
            time_bins = yaml.safe_load(f)
        ds = ethos.UKBDataset(
            **common_args,
            time_bins=time_bins,
        )
    else:
        ds = UKBDataset(**common_args)

    it = tqdm(
        eval_iter(total_size=len(ds), batch_size=batch_size),
        total=math.ceil(len(ds) / batch_size),
        leave=False,
    )

    loss = defaultdict(float)
    with torch.no_grad():
        for batch_idx in it:
            batch_input = ds.get_batch(batch_idx)
            batch_input = move_batch_to_device(batch_input, device=device)
            targets = batch_input[2]

            batch_size = batch_idx.shape[0]
            batch_logits, batch_loss, _ = model(*batch_input)

            if hasattr(ds, "time_tokens"):
                batch_logits[:, :, ds.time_tokens] = -torch.inf
                batch_logits = batch_logits.permute(0, 2, 1)
                loss_ce = F.cross_entropy(batch_logits, targets, reduction="none")
                timeless_ce = torch.mean(
                    loss_ce[target_mask(targets, ignore_tokens=ds.time_tokens)]
                )
                loss["timeless_ce"] += timeless_ce.item() * batch_size
            for key in batch_loss.keys():
                loss[key] += batch_loss[key].item() * batch_size

    loss = {key: value / len(ds) for key, value in loss.items()}
    print(loss)
