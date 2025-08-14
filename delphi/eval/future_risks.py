import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from delphi import DAYS_PER_YEAR
from delphi.data import core
from delphi.eval import eval_task
from delphi.eval.auc import mann_whitney_auc
from delphi.experiment.train import load_ckpt
from delphi.model.config import parse_token_list
from delphi.model.delphi import integrate_risk
from delphi.sampler import generate


@dataclass
class FutureArgs:
    data: dict = field(default_factory=dict)
    n_samples: int = 30
    start_age_years: int = 50
    end_age_years: float = 80
    subsample: Optional[int] = None
    batch_size: int = 128
    device: str = "cuda"
    seed: int = 42
    no_repeat: bool = True
    top_k: Optional[int] = None
    temperature: float = 1.0
    termination_tokens: list[str] = field(default_factory=list)


@eval_task.register
def sample_future(task_args: FutureArgs, task_name: str, ckpt: str) -> None:

    assert task_args.batch_size >= task_args.n_samples
    assert task_args.batch_size % task_args.n_samples == 0
    n_persons_per_batch = int(task_args.batch_size / task_args.n_samples)

    device = task_args.device
    model, _, tokenizer = load_ckpt(ckpt)
    model.to(device)
    model.eval()

    if model.model_type == "delphi-m4":
        raise NotImplementedError
    else:
        ds = core.build_dataset(task_args.data)
        duplicate_participants = core.duplicate_participants
        prompt_loader = core.load_prompt_sequences

    start_age = task_args.start_age_years * DAYS_PER_YEAR
    n_participants = len(ds) if task_args.subsample is None else task_args.subsample
    it = core.eval_iter(total_size=n_participants, batch_size=n_persons_per_batch)
    data_loader = prompt_loader(it=it, dataset=ds, start_age=start_age)
    data_loader = tqdm(
        data_loader, total=math.ceil(n_participants / n_persons_per_batch), leave=True
    )

    termination_tokens = parse_token_list(task_args.termination_tokens)
    termination_tokens = torch.Tensor(tokenizer.encode(termination_tokens)).to(device)

    future_risks = list()
    labels = list()

    with torch.no_grad():
        for batch_input in data_loader:

            batch_input = core.move_batch_to_device(batch_input, device=device)
            prompt_idx, prompt_age, target_idx, target_age = batch_input

            prompt_idx, prompt_age = duplicate_participants(
                prompt_idx, prompt_age, n_repeat=task_args.n_samples
            )
            idx, age = generate(
                model=model,
                idx=prompt_idx,
                age=prompt_age,
                seed=task_args.seed,
                no_repeat=task_args.no_repeat,
                top_k=task_args.top_k,
                temperature=task_args.temperature,
                max_age=task_args.end_age_years * DAYS_PER_YEAR,
                termination_tokens=termination_tokens,
            )
            idx = torch.cat((prompt_idx, idx), dim=1)
            age = torch.cat((prompt_age, age), dim=1)
            sort_by_age = torch.argsort(age, dim=1)
            idx = torch.take_along_dim(input=idx, indices=sort_by_age, dim=1)
            age = torch.take_along_dim(input=age, indices=sort_by_age, dim=1)

            logits, _ = model(idx=idx, age=age)
            B, L, V = logits.shape
            n_sample = task_args.n_samples
            n_person = int(B / n_sample)

            if model.model_type == "delphi":
                batch_risks = integrate_risk(
                    log_lambda=logits,
                    age=age,
                    start=task_args.start_age_years * DAYS_PER_YEAR,
                    end=task_args.end_age_years * DAYS_PER_YEAR,
                )
                batch_risks = batch_risks.reshape(-1, n_sample, V)
                batch_risks = torch.nanmean(batch_risks, dim=-2, keepdim=False)
                future_risks.append(batch_risks.detach().cpu().numpy())
            elif model.model_type == "ethos":
                tokens_aft_prompt = idx.clone()
                tokens_aft_prompt[age > prompt_age] = 0
                tokens_aft_prompt = tokens_aft_prompt.reshape((n_person, n_sample, L))
                occur = torch.zeros(
                    (n_person, n_sample, V), device=logits.device
                ).long()
                occur = occur.scatter_(
                    dim=-1,
                    index=tokens_aft_prompt,
                    src=torch.ones_like(tokens_aft_prompt),
                )
                batch_risks = torch.mean(occur.float(), dim=-2, keepdim=False)
                future_risks.append(batch_risks.detach().cpu().numpy())
            elif model.model_type == "motor":
                raise NotImplementedError
            else:
                raise ValueError

            tokens_after_prompt = target_idx.clone()
            tokens_after_prompt[target_age <= start_age] = 0
            occur_after = torch.zeros_like(batch_risks).long()
            occur_after = occur_after.scatter_(
                index=tokens_after_prompt, src=torch.ones_like(target_idx), dim=1
            )

            occur_after = occur_after.float()
            labels.append(occur_after.detach().cpu().numpy())

    labels = np.vstack(labels)
    future_risks = np.vstack(future_risks)

    logbook = {}
    for i in range(2, labels.shape[1]):

        is_ctl = labels[:, i] == 0
        is_dis = labels[:, i] == 1
        ctl = future_risks[is_ctl, i]
        dis = future_risks[is_dis, i]

        disease = tokenizer.decode(i)
        logbook[disease] = mann_whitney_auc(x1=ctl, x2=dis)

    with open(Path(ckpt) / f"{task_name}.json", "w") as f:
        json.dump(logbook, f, indent=4)
