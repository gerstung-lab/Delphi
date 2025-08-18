import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from delphi import DAYS_PER_YEAR
from delphi.data import ukb
from delphi.data.utils import duplicate_participants, eval_iter, move_batch_to_device
from delphi.eval import eval_task
from delphi.eval.ukb.auc import mann_whitney_auc
from delphi.experiment.train import load_ckpt
from delphi.model.delphi import integrate_risk
from delphi.sampler import generate


@dataclass
class ForecastArgs:
    data: dict = field(default_factory=dict)
    n_samples: int = 30
    start_age_years: int = 50
    end_age_years: float = 80
    resolution_years: int = 1
    n_calibration_bins: int = 5
    subsample: Optional[int] = None
    batch_size: int = 128
    device: str = "cuda"
    seed: int = 42
    no_repeat: bool = True
    top_k: Optional[int] = None
    temperature: float = 1.0


def calibrate(rates: np.ndarray, occur: np.ndarray, n_bins: int):

    n = occur.shape[0]
    q = np.linspace(0, 1, n_bins + 1)
    rate_bins = np.quantile(rates, q=q).tolist()

    rate_per_bin, incidence_per_bin, n_per_bin = list(), list(), list()
    for i in range(n_bins):
        in_bin = np.logical_and(rates >= rate_bins[i], rates < rate_bins[i + 1])
        rate_per_bin.append(float(np.nanmean(rates[in_bin])))
        incidence_per_bin.append(float(occur[in_bin].sum() / n))
        n_per_bin.append(float(in_bin.sum()))

    return {
        "rate_bins": rate_bins,
        "rate_per_bin": rate_per_bin,
        "incidence_per_bin": incidence_per_bin,
        "n_per_bin": n_per_bin,
    }


@eval_task.register
def sample_future(task_args: ForecastArgs, task_name: str, ckpt: str) -> None:

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
        ds = ukb.build_dataset(task_args.data)
        prompt_loader = ukb.load_prompt_sequences

    start_age = task_args.start_age_years * DAYS_PER_YEAR
    end_age = task_args.end_age_years * DAYS_PER_YEAR
    n_participants = len(ds) if task_args.subsample is None else task_args.subsample
    it = eval_iter(total_size=n_participants, batch_size=n_persons_per_batch)
    data_loader = prompt_loader(it=it, dataset=ds, start_age=start_age)
    data_loader = tqdm(
        data_loader, total=math.ceil(n_participants / n_persons_per_batch), leave=True
    )

    forecast_risks = list()
    sampling_risks = list()
    baseline_risks = list()
    labels = list()

    with torch.no_grad():
        for batch_input in data_loader:

            batch_input = move_batch_to_device(batch_input, device=device)
            prompt_idx, prompt_age, target_idx, target_age = batch_input
            n_sample = task_args.n_samples
            n_person = prompt_idx.shape[0]

            prompt_idx, prompt_age = duplicate_participants(
                [prompt_idx, prompt_age], n_repeat=task_args.n_samples
            )
            prompt_logits, _ = model(prompt_idx, prompt_age)
            baseline_logits = prompt_logits[:, -1, :]
            baseline_logits = baseline_logits.reshape((n_person, n_sample, -1))
            baseline_prob = F.softmax(baseline_logits, dim=-1)
            baseline_prob = torch.mean(baseline_prob, dim=-2, keepdim=False)
            baseline_risks.append(baseline_prob.detach().cpu().numpy())

            gen_idx, gen_age, gen_logits = generate(
                model=model,
                idx=prompt_idx,
                age=prompt_age,
                seed=task_args.seed,
                no_repeat=task_args.no_repeat,
                top_k=task_args.top_k,
                temperature=task_args.temperature,
                max_age=end_age,
                termination_tokens=torch.tensor([tokenizer["death"]], device=device),
            )
            pred_age = torch.cat((prompt_age[:, [-1]], gen_age[:, 1:]), dim=1)
            sort_by_age = torch.argsort(pred_age, dim=1)
            gen_idx = torch.take_along_dim(input=gen_idx, indices=sort_by_age, dim=1)
            pred_age = torch.take_along_dim(input=pred_age, indices=sort_by_age, dim=1)
            gen_age = torch.take_along_dim(input=gen_age, indices=sort_by_age, dim=1)
            gen_logits = torch.take_along_dim(
                input=gen_logits, indices=sort_by_age.unsqueeze(-1), dim=1
            )

            if model.model_type == "delphi" or model.model_type == "ethos":
                end_age_until_occur = torch.full(
                    (gen_logits.shape[0], gen_logits.shape[-1]), fill_value=end_age
                ).to(gen_logits.device)
                end_age_until_occur = end_age_until_occur.scatter_(
                    src=gen_age, index=gen_idx, dim=1
                )
                integrated_lambda = integrate_risk(
                    log_lambda=gen_logits,
                    age=pred_age,
                    start=start_age,
                    end=end_age_until_occur,
                )
                integrated_lambda = integrated_lambda.reshape(n_person, n_sample, -1)
                batch_risks = 1 - torch.exp(-integrated_lambda * (end_age - start_age))
                batch_risks = torch.nanmean(batch_risks, dim=-2, keepdim=False)
                forecast_risks.append(batch_risks.detach().cpu().numpy())

                gen_idx = gen_idx.reshape((n_person, n_sample, -1))
                occur = torch.zeros(
                    (n_person, n_sample, batch_risks.shape[-1]),
                    device=gen_logits.device,
                ).long()
                occur = occur.scatter_(
                    dim=-1,
                    index=gen_idx,
                    src=torch.ones_like(gen_idx),
                )
                frac_occur = torch.mean(occur.float(), dim=-2, keepdim=False)
                sampling_risks.append(frac_occur.detach().cpu().numpy())
            elif model.model_type == "motor":
                raise NotImplementedError
            else:
                raise ValueError

            tokens_bef_and_aft = target_idx.clone()
            tokens_bef_and_aft[(target_age > start_age) & (target_age <= end_age)] = 0
            occur_bef_or_aft = torch.zeros_like(batch_risks).long()
            occur_bef_or_aft = occur_bef_or_aft.scatter_(
                index=tokens_bef_and_aft, src=torch.ones_like(target_idx), dim=1
            )

            tokens_in_range = target_idx.clone()
            tokens_in_range[target_age <= start_age] = 0
            tokens_in_range[target_age > end_age] = 0
            occur_during = torch.zeros_like(batch_risks).long()
            occur_during = occur_during.scatter_(
                index=tokens_in_range, src=torch.ones_like(target_idx), dim=1
            )

            occur_during = occur_during.float()
            occur_during[occur_bef_or_aft.bool()] = torch.nan
            labels.append(occur_during.detach().cpu().numpy())

    labels = np.vstack(labels)
    forecast_risks = np.vstack(forecast_risks)
    sampling_risks = np.vstack(sampling_risks)
    baseline_risks = np.vstack(baseline_risks)

    logbook = {}
    for i in range(2, labels.shape[1]):

        disease_labels = labels[:, i]
        exclude = np.isnan(disease_labels)
        is_ctl = disease_labels == 0
        is_dis = disease_labels == 1

        disease = tokenizer.decode(i)
        logbook[disease] = {
            "n_ctl": int(is_ctl.sum()),
            "n_dis": int(is_dis.sum()),
            "forecast": {
                "auc": mann_whitney_auc(
                    x1=forecast_risks[is_ctl, i], x2=forecast_risks[is_dis, i]
                ),
                "calibrate": calibrate(
                    forecast_risks[~exclude, i],
                    disease_labels[~exclude],
                    n_bins=task_args.n_calibration_bins,
                ),
            },
            "sampling_auc": {
                "auc": mann_whitney_auc(
                    x1=sampling_risks[is_ctl, i], x2=sampling_risks[is_dis, i]
                ),
                "calibrate": calibrate(
                    sampling_risks[~exclude, i],
                    disease_labels[~exclude],
                    n_bins=task_args.n_calibration_bins,
                ),
            },
            "baseline_auc": {
                "auc": mann_whitney_auc(
                    x1=baseline_risks[is_ctl, i], x2=baseline_risks[is_dis, i]
                ),
                "calibrate": calibrate(
                    baseline_risks[~exclude, i],
                    disease_labels[~exclude],
                    n_bins=task_args.n_calibration_bins,
                ),
            },
        }

    with open(Path(ckpt) / f"{task_name}.json", "w") as f:
        json.dump(logbook, f, indent=4)
