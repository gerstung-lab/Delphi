from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

from delphi.data.mimic import SofaPredictionDataset
from delphi.data.utils import eval_iter, move_batch_to_device
from delphi.env import DELPHI_DATA_DIR
from delphi.train import load_ckpt


def drg_classification(
    ckpt: str,
    device: str = "cuda",
    batch_size: int = 128,
    subsample: Optional[int] = None,
) -> None:

    model, _, _ = load_ckpt(ckpt)
    model.to(device)
    model.eval()

    eval_ds = SofaPredictionDataset(
        input_dir=Path(DELPHI_DATA_DIR) / "mimic" / "test",
        n_positions=model.config.block_size,
        sep_time_tokens=(model.model_type != "ethos"),
    )
    quantile_tokens = eval_ds.vocab.encode(eval_ds.vocab.quantile_stokens)

    n_participants = len(eval_ds) if subsample is None else subsample
    it = tqdm(
        eval_iter(total_size=n_participants, batch_size=batch_size),
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
            quantile_logits = batch_logits[:, -1, quantile_tokens]
            y_prob.append(F.softmax(quantile_logits, dim=-1))

            for label in batch_label:
                y_true.append(eval_ds.vocab.encode(label["expected"]))

    y_prob = torch.cat(y_prob, dim=0).detach().cpu().numpy()
    y_true = np.array(y_true)
    print(
        top_k_accuracy_score(y_true=y_true, y_score=y_prob, k=5, labels=quantile_tokens)
    )
