from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

from delphi.data.mimic import DrgPredictionDataset
from delphi.data.utils import eval_iter, move_batch_to_device
from delphi.env import DELPHI_DATA_DIR
from delphi.eval import eval_task
from delphi.train import load_ckpt


@dataclass
class DRGClassificationArgs:
    subsample: Optional[int] = None
    batch_size: int = 128
    device: str = "cuda"


@eval_task.register
def drg_classification(
    task_args: DRGClassificationArgs, task_name: str, ckpt: str
) -> None:

    device = task_args.device
    model, _, _ = load_ckpt(ckpt)
    model.to(device)
    model.eval()

    eval_ds = DrgPredictionDataset(
        input_dir=Path(DELPHI_DATA_DIR) / "mimic" / "test",
        n_positions=model.config.block_size,
        sep_time_tokens=(model.model_type != "ethos"),
    )
    non_drg_tokens = [0, 1]
    drg_tokens = list()
    for token_name, token in eval_ds.vocab.stoi.items():
        if not "DRG//" in token_name:
            non_drg_tokens.append(token)
        else:
            drg_tokens.append(token)

    n_participants = (
        len(eval_ds) if task_args.subsample is None else task_args.subsample
    )
    it = tqdm(
        eval_iter(total_size=n_participants, batch_size=task_args.batch_size),
        total=n_participants,
        leave=True,
    )

    y_prob = list()
    y_true = list()
    with torch.no_grad():
        for batch_idx in it:
            *batch_input, batch_label = eval_ds.get_batch(
                batch_idx, include_time=(model.model_type != "ethos")
            )
            batch_input = move_batch_to_device(batch_input, device=device)

            batch_logits, _ = model(*batch_input)
            drg_logits = batch_logits[:, -1, :]
            drg_logits[:, non_drg_tokens] = -torch.inf
            y_prob.append(F.softmax(drg_logits, dim=-1))

            for label in batch_label:
                y_true.append(eval_ds.vocab.encode(label["expected"]))

    y_prob = torch.cat(y_prob, dim=0).detach().cpu().numpy()
    y_prob = y_prob[:, drg_tokens]
    y_true = np.array(y_true)
    print(top_k_accuracy_score(y_true=y_true, y_score=y_prob, k=5, labels=drg_tokens))
