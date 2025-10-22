import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from tqdm import tqdm

from delphi import DAYS_PER_YEAR
from delphi.data.ukb import UKBDataConfig, UKBDataset
from delphi.data.utils import eval_iter
from delphi.eval.ukb.auc import mann_whitney_auc
from delphi.exponential import integrate_risk
from delphi.generate import generate
from delphi.train import load_ckpt


@dataclass
class ForecastArgs:
    data: UKBDataConfig = field(default_factory=UKBDataConfig)
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
    token_budget: Optional[int] = None
    log_name: str = "forecast.yaml"


def cleanse_nan(obj):
    """Recursively handle NaN values in nested data structures"""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {k: cleanse_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [cleanse_nan(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(cleanse_nan(item) for item in obj)
    else:
        return obj


def duplicate_participants(args: Iterable[torch.Tensor], n_repeat: int):

    return tuple(
        [torch.repeat_interleave(arg, repeats=n_repeat, dim=0) for arg in args]
    )


def sample_future(task_args: ForecastArgs, ckpt: str) -> None:
    assert task_args.batch_size >= task_args.n_samples
    assert task_args.batch_size % task_args.n_samples == 0
    n_persons_per_batch = int(task_args.batch_size / task_args.n_samples)

    device = task_args.device
    model, _, tokenizer = load_ckpt(ckpt)
    if model.model_type == "ethos":
        time_tokens = list()
        time_intervals = list()
        for token_name, token in tokenizer.items():
            if token_name.startswith("time"):
                time_tokens.append(token)
                _, start, end = token_name.split("-")
                time_intervals.append(float(start))
        model.set_time(time_tokens=time_tokens, time_intervals=time_intervals)
    model.to(device)
    model.eval()

    ds = UKBDataset(**asdict(task_args.data))

    start_age = task_args.start_age_years * DAYS_PER_YEAR
    end_age = task_args.end_age_years * DAYS_PER_YEAR
    n_participants = len(ds) if task_args.subsample is None else task_args.subsample
    it = eval_iter(total_size=n_participants, batch_size=n_persons_per_batch)
    it = tqdm(it, total=math.ceil(n_participants / n_persons_per_batch), leave=True)

    batch_occur = list()
    batch_person_time = list()
    batch_estimates = defaultdict(list)
    is_gender = {"male": list(), "female": list()}

    with torch.no_grad():
        for batch_idx in it:
            prompt_idx, prompt_age = ds.get_prompt_batch(batch_idx, start_age=start_age)
            _, target_begin_age, target_idx, target_age = ds.get_batch(batch_idx)

            prompt_idx, prompt_age = prompt_idx.to(device), prompt_age.to(device)
            target_idx, target_age = target_idx.to(device), target_age.to(device)
            is_gender["male"].append((prompt_idx == tokenizer["male"]).any(dim=1))
            is_gender["female"].append((prompt_idx == tokenizer["female"]).any(dim=1))

            n_sample = task_args.n_samples
            n_person = prompt_idx.shape[0]

            if model.model_type == "delphi" or model.model_type == "ethos":
                prompt_idx, prompt_age = duplicate_participants(
                    [prompt_idx, prompt_age], n_repeat=task_args.n_samples
                )
                prompt_logits, _, _ = model(prompt_idx, prompt_age)
                gen_idx, gen_age, gen_logits = generate(
                    model=model,
                    idx=prompt_idx,
                    age=prompt_age,
                    seed=task_args.seed,
                    no_repeat=task_args.no_repeat,
                    top_k=task_args.top_k,
                    temperature=task_args.temperature,
                    max_age=end_age,
                    termination_tokens=torch.tensor(
                        [tokenizer["death"]], device=device
                    ),
                    token_budget=task_args.token_budget,
                )
                pred_age = torch.cat((prompt_age[:, [-1]], gen_age[:, 1:]), dim=1)
                sort_by_age = torch.argsort(pred_age, dim=1)
                gen_idx = torch.take_along_dim(
                    input=gen_idx, indices=sort_by_age, dim=1
                )
                pred_age = torch.take_along_dim(
                    input=pred_age, indices=sort_by_age, dim=1
                )
                gen_age = torch.take_along_dim(
                    input=gen_age, indices=sort_by_age, dim=1
                )
                gen_logits = torch.take_along_dim(
                    input=gen_logits, indices=sort_by_age.unsqueeze(-1), dim=1
                )

                baseline_logits = prompt_logits[:, -1, :]
                baseline_logits = baseline_logits.reshape((n_person, n_sample, -1))
                baseline_logits = torch.mean(baseline_logits, dim=-2, keepdim=False)
                batch_estimates["baseline"].append(
                    baseline_logits.detach().cpu().numpy()
                )

                if model.model_type == "delphi":
                    int_risk = integrate_risk(
                        log_lambda=gen_logits,
                        age=pred_age,
                        start=start_age,
                        end=end_age,
                    )
                    int_lambda = integrate_incidence_rate(
                        log_lambda=gen_logits,
                        age=pred_age,
                        start=start_age,
                        end=end_age,
                    )
                    int_risk = int_risk.reshape(n_person, n_sample, -1)
                    int_lambda = int_lambda.reshape(n_person, n_sample, -1)
                    dis_prob = 1 - torch.exp(-int_risk)
                    dis_prob = torch.nanmean(dis_prob, dim=-2, keepdim=False)
                    int_lambda = torch.nanmean(int_lambda, dim=-2, keepdim=False)
                    batch_estimates["forecast"].append(dis_prob.detach().cpu().numpy())
                    batch_estimates["forecast_lambda"].append(
                        int_lambda.detach().cpu().numpy()
                    )

                gen_idx = gen_idx.reshape((n_person, n_sample, -1))
                occur = torch.zeros(
                    (n_person, n_sample, gen_logits.shape[-1]),
                    device=gen_logits.device,
                ).long()
                occur = occur.scatter_(
                    dim=-1,
                    index=gen_idx,
                    src=torch.ones_like(gen_idx),
                )
                frac_occur = torch.mean(occur.float(), dim=-2, keepdim=False)
                batch_estimates["sampling"].append(frac_occur.detach().cpu().numpy())

            end_age = target_age.clamp(min=0)
            end_age = end_age[:, [-1]].broadcast_to(
                end_age.shape[0], model.config.vocab_size
            )
            end_age = end_age.scatter_(index=target_idx, src=target_age, dim=1)

            person_time = (end_age - target_begin_age.clamp(min=0)).nansum(dim=1)
            batch_person_time.append(person_time)

            tokens_during = target_idx.clone()
            tokens_during[target_age <= start_age] = 0
            tokens_during[target_age > end_age] = 0
            occur = torch.zeros(
                size=(target_idx.shape[0], model.config.vocab_size)
            ).long()
            occur = occur.scatter_(index=tokens_during, value=1, dim=1)
            occur = occur.float()

            tokens_bef = target_idx.clone()
            tokens_bef[target_age > start_age] = 0
            occur_bef = torch.zeros_like(occur).long()
            occur_bef = occur_bef.scatter_(index=tokens_bef, value=1, dim=1)
            occur[occur_bef.bool()] = torch.nan

            last_age = target_age.max(dim=1).values
            not_enough_data = last_age < end_age
            occur[not_enough_data] = torch.nan

            batch_occur.append(occur.detach().cpu().numpy())

    occur = np.vstack(batch_occur)
    person_time = np.vstack(batch_person_time)
    estimates = dict()
    for key in batch_estimates.keys():
        estimates[key] = np.vstack(batch_estimates[key])

    for gender in ["male", "female"]:
        is_gender[gender] = torch.cat(is_gender[gender], dim=0).detach().cpu().numpy()
    is_gender["either"] = np.ones_like(is_gender["male"]).astype(bool)  # type: ignore
    logbook = defaultdict(dict)
    reverse_tokenizer = {v: k for k, v in tokenizer.items()}
    for i in range(2, occur.shape[1]):
        for gender in ["male", "female", "either"]:
            include = (~np.isnan(occur[:, i])) & is_gender[gender]
            is_ctl = (occur[i, :] == 0) & is_gender[gender]
            is_dis = (occur[i, :] == 1) & is_gender[gender]

            disease = reverse_tokenizer[i]
            logbook[disease][gender] = {
                "n_ctl": int(is_ctl.sum()),
                "n_dis": int(is_dis.sum()),
                "person_time": int(person_time[is_gender[gender]].sum()),
            }
            for key in batch_estimates.keys():
                mean_rate = (
                    np.nanmean(estimates[key][include, i])
                    if include.sum() > 0
                    else np.nan
                )
                logbook[disease][gender][key] = {
                    "auc": mann_whitney_auc(
                        x1=estimates[key][is_ctl, i], x2=estimates[key][is_dis, i]
                    ),
                    "mean": float(f"{mean_rate:.3e}"),
                }
                logbook[disease][gender][key]["mean_all"] = float(
                    f"{np.nanmean(estimates[key][:, i])}.3e"
                )

    logbook = cleanse_nan(logbook)
    with open(Path(ckpt) / f"{task_args.log_name}.json", "w") as f:
        json.dump(logbook, f, indent=2, separators=(",", ": "))
