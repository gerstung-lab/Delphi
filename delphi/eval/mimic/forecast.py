import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sklearn
import sklearn.metrics
import torch
from tqdm import tqdm

from delphi.data.mimic import (
    HospitalMortalityDataset,
    ICUMortalityDataset,
    ICUReadmissionDataset,
    SpecialToken,
)
from delphi.data.utils import duplicate_participants, eval_iter, move_batch_to_device
from delphi.env import DELPHI_DATA_DIR
from delphi.eval import eval_task
from delphi.experiment.train import load_ckpt
from delphi.sampler import generate


@dataclass
class ForecastArgs:
    task: str = "hospital_mortality"
    n_samples: int = 30
    subsample: Optional[int] = None
    batch_size: int = 128
    device: str = "cuda"
    seed: int = 42
    no_repeat: bool = False
    top_k: Optional[int] = None
    temperature: float = 1.0
    termination_tokens: list[str] = field(default_factory=list)


@eval_task.register
def sample_future(task_args: ForecastArgs, task_name: str, ckpt: str) -> None:

    device = task_args.device
    model, _, _ = load_ckpt(ckpt)
    model.to(device)
    model.eval()

    data_dir = Path(DELPHI_DATA_DIR) / "mimic" / "test"
    n_positions = model.config.block_size
    if task_args.task == "hospital_mortality":
        eval_ds = HospitalMortalityDataset(
            input_dir=data_dir,
            n_positions=n_positions,
            sep_time_tokens=(model.model_type != "ethos"),
        )
        termination_events = [SpecialToken.TIMELINE_END, SpecialToken.DISCHARGE]
        outcome = SpecialToken.DEATH
        max_time = 30 * 24 * 60
    elif task_args.task == "icu_mortality":
        eval_ds = ICUMortalityDataset(
            input_dir=data_dir,
            n_positions=n_positions,
            sep_time_tokens=(model.model_type != "ethos"),
        )
        termination_events = ["MEDS_DEATH", "ICU_DISCHARGE"]
        outcome = "MEDS_DEATH"
        max_time = 30 * 24 * 60
    else:
        raise NotImplementedError
    termination_tokens = torch.tensor(
        eval_ds.vocab.encode(termination_events), device=device
    )
    target_token = eval_ds.vocab.encode(outcome)

    if model.model_type == "ethos":
        token_to_time = torch.zeros((model.config.vocab_size,))
        time_token_to_mean = eval_ds.interval_estimates["mean"]
        for time_token, mean_time in time_token_to_mean.items():
            time_token = eval_ds.vocab.encode(time_token)
            token_to_time[time_token] = mean_time / 1e6 / 60
        model.set_time(token_to_time)
        model.to(device)

    assert task_args.batch_size >= task_args.n_samples
    assert task_args.batch_size % task_args.n_samples == 0
    n_persons_per_batch = int(task_args.batch_size / task_args.n_samples)
    n_participants = (
        len(eval_ds) if task_args.subsample is None else task_args.subsample
    )
    it = tqdm(
        eval_iter(total_size=n_participants, batch_size=n_persons_per_batch),
        total=math.ceil(n_participants / n_persons_per_batch),
        leave=True,
    )

    y_prob = list()
    y_true = list()
    with torch.no_grad():
        for batch_idx in it:

            *batch_input, batch_label = eval_ds.get_batch(batch_idx)
            batch_input = move_batch_to_device(batch_input, device=device)
            batch_input = duplicate_participants(
                batch_input, n_repeat=task_args.n_samples
            )
            prompt_idx, prompt_age = batch_input
            gen_idx, _, gen_logits = generate(
                model=model,
                idx=prompt_idx,
                age=prompt_age,
                seed=task_args.seed,
                no_repeat=False,
                top_k=task_args.top_k,
                temperature=task_args.temperature,
                max_time=max_time,
                termination_tokens=termination_tokens,
                stop_at_block_size=False,
            )

            B, L, V = gen_logits.shape
            n_sample = task_args.n_samples
            n_person = int(B / n_sample)

            gen_idx = gen_idx.reshape((n_person, n_sample, L))
            occur = torch.zeros((n_person, n_sample, V), device=gen_idx.device).long()
            occur = occur.scatter_(
                dim=-1,
                index=gen_idx,
                src=torch.ones_like(gen_idx),
            )
            batch_y_prob = torch.mean(occur.float(), dim=-2, keepdim=False)
            batch_y_prob = batch_y_prob[:, target_token]
            y_prob.append(batch_y_prob)

            batch_label = torch.Tensor(
                [label["expected"] == outcome for label in batch_label]
            )
            y_true.append(batch_label)

    y_prob = torch.cat(y_prob, dim=0).detach().cpu().numpy()
    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()

    logbook = {"auc": sklearn.metrics.roc_auc_score(y_true, y_prob)}
    with open(Path(ckpt) / f"{task_name}.json", "w") as f:
        json.dump(logbook, f, indent=4)
