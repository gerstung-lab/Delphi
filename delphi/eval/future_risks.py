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
from delphi.eval.auc import mann_whitney_auc, move_batch_to_device
from delphi.experiment import load_ckpt
from delphi.model.config import parse_token_list
from delphi.sampler import generate


def integrate_risk(x: torch.Tensor, t: torch.Tensor, start: float, end: float):
    r"""
    Aggregate values x over time intervals t within a specified time window [start, end].
    As per the theory of non-homogeneous exponential distribution, the probability
    an event occurs in the time window [start, end] is given by:
    P(event in [start, end]) = 1 - exp(- \int_{start}^{end} \lambda(t) dt)
    where \lambda(t) is the disease rate at time t.
    This this function calculates the integral of the disease rate over the time window
    under that piecewise constant disease rate assumption, using the tokens that
    fall in the time window.

    Args:
        x: Disease rate to integrate, lambda_0, ...., lambda_n, [batch, block_size, disease]
        t: Time points, days since birth, t_0, ...., t_n, t_(n+1) [batch, block_size]
            (the last time point is needed to calculate the duration of the last event)
        start: Start of time window
        end: End of time window

    Returns:
        Aggregated risk values, normalized by time exposure
    """

    # Clamp time values to the end of the window
    t_clamped = t.clamp(None, end)

    # Create usage mask for each time interval
    # If there are no time points in the window, the use mask will be all zeros
    # and the risk will be NaN
    use = ((t_clamped >= start) * (t_clamped < end)) + 0.0
    dt = t_clamped.diff(1)

    # Apply masks to get effective time exposure within the window
    dt_masked = dt * use[:, :-1]

    # Normalize time weights to sum to the length of the window
    if end == float("inf"):
        end_t, _ = torch.max(t, dim=1)
    else:
        end_t = end
    dt_norm = dt_masked / ((dt_masked.sum(1) + 1e-6) * (end_t - start)).unsqueeze(-1)

    # Calculate risk by weighting x values with normalized time exposure
    # print(x.shape, dt_norm.shape)
    risk = x * dt_norm.unsqueeze(-1)
    risk = risk.sum(-2)  # Sum over the time dimension

    # Set zero risks to NaN (indicates no exposure in the time window)
    risk[risk == 0] = torch.nan

    return risk


@dataclass
class FutureArgs:
    data: dict = field(default_factory=dict)
    subsample: Optional[int] = None
    disease_lst: list = field(default_factory=list)
    batch_size: int = 128
    n_samples_per_person: int = 32
    age_at_prompt: int = 40
    device: str = "cuda"
    seed: int = 42
    no_repeat: bool = True
    top_k: Optional[int] = None
    temperature: float = 1.0
    max_age_in_years: float = 80
    termination_tokens: list[str] = field(default_factory=list)


@eval_task.register
def sample_future(task_args: FutureArgs, task_name: str, ckpt: str) -> None:

    assert task_args.batch_size >= task_args.n_samples_per_person
    assert task_args.batch_size % task_args.n_samples_per_person == 0
    n_persons_per_batch = int(task_args.batch_size / task_args.n_samples_per_person)

    device = task_args.device
    model, _, tokenizer = load_ckpt(ckpt)
    model.to(device)
    model.eval()

    if model.model_type == "delphi-m4":
        raise NotImplementedError
    else:
        ds = core.build_dataset(task_args.data)
        duplicate_participants = core.duplicate_participants
        loader = core.load_sequences
        prompt_loader = core.load_prompt_sequences

    prompt_age = task_args.age_at_prompt * DAYS_PER_YEAR
    n_participants = len(ds) if task_args.subsample is None else task_args.subsample
    it = core.eval_iter(total_size=n_participants, batch_size=n_persons_per_batch)
    data_loader = prompt_loader(it=it, dataset=ds, start_age=prompt_age)
    data_loader = tqdm(
        data_loader, total=math.ceil(n_participants / task_args.batch_size), leave=True
    )
    gt_it = core.eval_iter(total_size=n_participants, batch_size=n_persons_per_batch)
    gt_loader = loader(it=gt_it, dataset=ds)

    termination_tokens = parse_token_list(task_args.termination_tokens)
    termination_tokens = torch.Tensor(tokenizer.encode(termination_tokens)).to(device)

    future_risks = list()
    labels = list()

    with torch.no_grad():
        for batch_input in data_loader:

            batch_input = move_batch_to_device(batch_input, device=device)
            batch_input = duplicate_participants(
                *batch_input, n_repeat=task_args.n_samples_per_person
            )
            _, age, logits = generate(
                model=model,
                model_input=batch_input,
                seed=task_args.seed,
                no_repeat=task_args.no_repeat,
                top_k=task_args.top_k,
                temperature=task_args.temperature,
                max_age=task_args.max_age_in_years * DAYS_PER_YEAR,
                termination_tokens=termination_tokens,
            )
            batch_risks = integrate_risk(
                x=logits,
                t=age,
                start=task_args.age_at_prompt * DAYS_PER_YEAR,
                end=float("inf"),
            )
            batch_risks = batch_risks.reshape(
                -1, task_args.n_samples_per_person, batch_risks.shape[-1]
            )
            batch_risks = torch.nanmean(batch_risks, dim=-2, keepdim=False)
            future_risks.append(batch_risks.detach().cpu().numpy())

            batch_truth = next(gt_loader)
            batch_truth = move_batch_to_device(batch_truth, device=device)
            targets, age = batch_truth[0], batch_truth[1]
            tokens_before_prompt = targets.clone()
            tokens_before_prompt[age > prompt_age] = 0
            occur_before = torch.zeros_like(batch_risks).long()
            occur_before = occur_before.scatter_(
                index=tokens_before_prompt, src=torch.ones_like(targets), dim=1
            )

            tokens_after_prompt = targets.clone()
            tokens_after_prompt[age <= prompt_age] = 0
            occur_after = torch.zeros_like(batch_risks).long()
            occur_after = occur_after.scatter_(
                index=tokens_after_prompt, src=torch.ones_like(targets), dim=1
            )

            occur_after = occur_after.float()
            occur_after[occur_before.bool()] = torch.nan
            labels.append(occur_after.detach().cpu().numpy())

    labels = np.vstack(labels)
    future_risks = np.vstack(future_risks)
    print(future_risks)

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
