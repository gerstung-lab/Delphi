import math
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from delphi.data.ukb import UKBDataset
from delphi.data.utils import eval_iter, move_batch_to_device
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment.train import load_ckpt


def estimate_loss(ckpt, device: str = "cuda", batch_size: int = 1024):

    model, _, tokenizer = load_ckpt(ckpt)
    model.to(device)
    model.eval()

    ds = UKBDataset(
        data_dir="ukb_real_data",
        subject_list="participants/val_fold.bin",
        no_event_interval=365.25 * 5,
        block_size=model.config.block_size,
        memmap=False,
    )

    it = tqdm(
        eval_iter(total_size=len(ds), batch_size=batch_size),
        total=math.ceil(len(ds) / batch_size),
        leave=False,
    )

    loss = defaultdict(float)
    with torch.no_grad():
        for batch_idx in it:
            for batch_idx in it:
                batch_input = ds.get_batch(batch_idx)
                batch_input = move_batch_to_device(batch_input, device=device)

                batch_size = batch_idx.shape[0]
                _, batch_loss = model(*batch_input)
                for key in batch_loss.keys():
                    loss[key] += batch_loss[key].item() * batch_size

    loss = {key: value / len(ds) for key, value in loss.items()}
    print(loss)
