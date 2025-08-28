import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from delphi import DAYS_PER_YEAR
from delphi.baselines import ethos
from delphi.data.ukb import UKBDataConfig, UKBDataset
from delphi.data.utils import duplicate_participants, eval_iter
from delphi.eval import eval_task
from delphi.eval.ukb.auc import mann_whitney_auc
from delphi.generate import generate
from delphi.model.delphi import integrate_risk
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


def calibrate(rates: np.ndarray, occur: np.ndarray, bins: list):
    n = occur.shape[0]
    n_bins = len(bins) - 1

    rate_per_bin, incidence_per_bin, n_per_bin = list(), list(), list()
    for i in range(n_bins):
        in_bin = np.logical_and(rates >= bins[i], rates < bins[i + 1])
        mean_rate = float(np.nanmean(rates[in_bin])) if in_bin.sum() > 0 else np.nan
        rate_per_bin.append(float(f"{mean_rate:.3e}"))
        incidence = float(occur[in_bin].sum() / n) if n > 0 else np.nan
        incidence_per_bin.append(float(f"{incidence:.3e}"))
        n_per_bin.append(int(in_bin.sum()))

    return {
        "rate_per_bin": rate_per_bin,
        "incidence_per_bin": incidence_per_bin,
        "n_per_bin": n_per_bin,
    }


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


@eval_task.register
def sample_future(task_args: ForecastArgs, task_name: str, ckpt: str) -> None:
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

    if model.model_type == "ethos":
        ds = ethos.UKBDataset(**asdict(task_args.data), time_bins=time_intervals)
    elif model.model_type == "delphi-m4":
        raise NotImplementedError
    else:
        ds = UKBDataset(**asdict(task_args.data))

    start_age = task_args.start_age_years * DAYS_PER_YEAR
    end_age = task_args.end_age_years * DAYS_PER_YEAR
    n_participants = len(ds) if task_args.subsample is None else task_args.subsample
    it = eval_iter(total_size=n_participants, batch_size=n_persons_per_batch)
    it = tqdm(it, total=math.ceil(n_participants / n_persons_per_batch), leave=True)

    labels = list()
    risks = defaultdict(list)
    is_gender = {"male": list(), "female": list()}

    with torch.no_grad():
        for batch_idx in it:
            prompt_idx, prompt_age = ds.get_prompt_batch(batch_idx, start_age=start_age)
            _, _, target_idx, target_age = ds.get_batch(batch_idx)

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
                baseline_prob = F.softmax(baseline_logits, dim=-1)
                baseline_prob = torch.mean(baseline_prob, dim=-2, keepdim=False)
                risks["baseline"].append(baseline_prob.detach().cpu().numpy())

                if model.model_type == "delphi":
                    pos = torch.arange(gen_idx.shape[1], device=gen_logits.device)
                    occur_pos = torch.full(
                        (gen_logits.shape[0], gen_logits.shape[-1]),
                        fill_value=gen_idx.shape[1] - 1,
                    ).to(gen_logits.device)
                    occur_pos = occur_pos.scatter_(
                        src=pos.broadcast_to(gen_idx.shape),
                        index=gen_idx,
                        dim=1,
                    ).unsqueeze(1)
                    occur_logits = torch.gather(
                        input=gen_logits, index=occur_pos, dim=1
                    )
                    gen_logits = torch.where(
                        condition=pos.view(1, -1, 1) >= occur_pos,
                        input=occur_logits,
                        other=gen_logits,
                    )
                    integrated_risks = integrate_risk(
                        log_lambda=gen_logits,
                        age=pred_age,
                        start=start_age,
                        end=end_age,
                    )
                    integrated_risks = integrated_risks.reshape(n_person, n_sample, -1)
                    batch_risks = 1 - torch.exp(-integrated_risks)
                    batch_risks = torch.nanmean(batch_risks, dim=-2, keepdim=False)
                    risks["forecast"].append(batch_risks.detach().cpu().numpy())

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
                risks["sampling"].append(frac_occur.detach().cpu().numpy())

            if model.model_type == "motor":

                # log_lambda, _, output_dict = model(prompt_idx, prompt_age)

                raise NotImplementedError

            tokens_in_range = target_idx.clone()
            tokens_in_range[target_age <= start_age] = 0
            tokens_in_range[target_age > end_age] = 0
            occur_during = torch.zeros_like(baseline_prob).long()
            occur_during = occur_during.scatter_(index=tokens_in_range, value=1, dim=1)
            occur_during = occur_during.float()

            tokens_bef = target_idx.clone()
            tokens_bef[target_age > start_age] = 0
            occur_bef = torch.zeros_like(baseline_prob).long()
            occur_bef = occur_bef.scatter_(index=tokens_bef, value=1, dim=1)
            occur_during[occur_bef.bool()] = torch.nan

            tokens_aft = target_idx.clone()
            tokens_aft[target_age <= end_age] = 0
            occur_aft = torch.zeros_like(baseline_prob).long()
            occur_aft = occur_aft.scatter_(index=tokens_aft, value=1, dim=1)
            occur_during[occur_aft.bool()] = torch.nan

            last_age = target_age.max(dim=1).values
            not_enough_data = last_age < end_age
            occur_during[not_enough_data] = torch.nan

            labels.append(occur_during.detach().cpu().numpy())

    labels = np.vstack(labels)
    for risk_type in risks.keys():
        risks[risk_type] = np.vstack(risks[risk_type])

    bins = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    for gender in ["male", "female"]:
        is_gender[gender] = torch.cat(is_gender[gender], dim=0).detach().cpu().numpy()
    is_gender["either"] = np.ones_like(is_gender["male"]).astype(bool)  # type: ignore
    logbook = defaultdict(dict)
    logbook["rate_bins"] = [float(f"{rate:.3e}") for rate in bins]  # type: ignore
    reverse_tokenizer = {v: k for k, v in tokenizer.items()}
    for i in range(2, labels.shape[1]):
        for gender in ["male", "female", "either"]:
            disease_labels = labels[:, i]
            include = (~np.isnan(disease_labels)) & is_gender[gender]
            is_ctl = (disease_labels == 0) & is_gender[gender]
            is_dis = (disease_labels == 1) & is_gender[gender]

            disease = reverse_tokenizer[i]
            logbook[disease][gender] = {
                "n_ctl": int(is_ctl.sum()),
                "n_dis": int(is_dis.sum()),
            }
            for risk_type in risks.keys():
                mean_rate = (
                    np.nanmean(risks[risk_type][include, i])
                    if include.sum() > 0
                    else np.nan
                )
                logbook[disease][gender][risk_type] = {
                    "auc": mann_whitney_auc(
                        x1=risks[risk_type][is_ctl, i], x2=risks[risk_type][is_dis, i]
                    ),
                    "mean": float(f"{mean_rate:.3e}"),
                    **calibrate(
                        risks[risk_type][include, i],
                        disease_labels[include],
                        bins=bins,
                    ),
                }

    logbook = cleanse_nan(logbook)
    with open(Path(ckpt) / f"{task_name}.json", "w") as f:
        json.dump(logbook, f, indent=2, separators=(",", ": "))
