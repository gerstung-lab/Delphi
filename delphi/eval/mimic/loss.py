import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from delphi.data.mimic import MIMICDataset
from delphi.data.utils import eval_iter, move_batch_to_device
from delphi.env import DELPHI_CKPT_DIR, DELPHI_DATA_DIR
from delphi.model.components import target_mask
from delphi.train import load_ckpt


def estimate_loss(
    ckpt,
    device: str = "cuda",
    batch_size: int = 256,
    subsample: Optional[int] = None,
    mask_logits: bool = False,
):

    model, _, _ = load_ckpt(ckpt)
    model.to(device)
    model.eval()

    ds = MIMICDataset(
        input_dir=Path(DELPHI_DATA_DIR) / "mimic" / "test",
        n_positions=model.config.block_size,
        sep_time_tokens=model.model_type != "ethos",
    )

    total_size = subsample if subsample is not None else len(ds)
    it = tqdm(
        eval_iter(total_size=total_size, batch_size=batch_size),
        total=math.ceil(total_size / batch_size),
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

            if mask_logits:
                batch_logits[:, :, ds.time_tokens] = -torch.inf
            batch_logits = batch_logits.permute(0, 2, 1)
            loss_ce = F.cross_entropy(batch_logits, targets, reduction="none")
            timeless_ce = torch.mean(
                loss_ce[target_mask(targets, ignore_tokens=ds.time_tokens)]
            )
            loss["timeless_ce"] += timeless_ce.item() * batch_size
            for key in batch_loss.keys():
                loss[key] += batch_loss[key].item() * batch_size

    loss = {key: value / total_size for key, value in loss.items()}
    print(loss)
    logbook_path = Path(ckpt) / "mimic_loss.json"
    with open(logbook_path, "w") as f:
        json.dump(loss, f, indent=4)
