import os
os.chdir("/hps/nobackup/birney/users/sfan/Delphi")
#

# !pip install ipywidgets

# +
import sys
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

import torch

from delphi.data.ukb import UKBDataset
from delphi.data.utils import eval_iter
from delphi.env import DELPHI_CKPT_DIR
from delphi.generate import legacy_generate
# from delphi.legacy.utils import get_batch, get_p2i
from delphi.model import Delphi2M, Delphi2MConfig

delphi_labels = pd.read_csv('notebook/delphi_labels_chapters_colours_icd.csv')

DAYS_PER_YEAR = 365.25
# +
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default="delphi-2m-og")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--age', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--subsample', type=int, default=None)
parser.add_argument('--n_repeats', type=int, default=1)
parser.add_argument('--stop_at_block_size', type=bool, default=False)

if 'ipykernel' in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
else:
    print(f"running from cli")
    args = parser.parse_args()
# -

ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
device = args.device
age = args.age
batch_size = args.batch_size
subsample = args.subsample
n_repeats = args.n_repeats
stop_at_block_size = args.stop_at_block_size

ckpt = Path(DELPHI_CKPT_DIR) / "delphi-2m-ablation" / "delphi-2m-long-ctx"
stop_at_block_size = True
must_have_lifestyle = True

stop_at_block_size = False
must_have_lifestyle = False

print(f"ckpt: {ckpt}")
print(f"device: {device}")
print(f"age: {age}")
print(f"batch_size: {batch_size}")
print(f"subsample: {subsample}")
print(f"n_repeats: {n_repeats}")
print(f"stop_at_block_size: {stop_at_block_size}")
print(f"must_have_lifestyle: {must_have_lifestyle}")

ckpt_dict = torch.load(
    ckpt / "ckpt.pt",
    map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
)
model = Delphi2M(Delphi2MConfig(**ckpt_dict['model_args']))
print(ckpt_dict["model_args"])
model.load_state_dict(ckpt_dict["model"])
model.to(device)
model.eval();

# +
ds = UKBDataset(
    data_dir="ukb_real_data",
    subject_list="participants/val_fold.bin",
    perturb=False,
    block_size=model.config.block_size,
)

total = subsample if subsample is not None else len(ds)
it = eval_iter(total_size=total, batch_size=batch_size)
d = ds.get_batch(np.arange(total))
# -





# +
n_samples = total

w = np.where((d[1].cpu().detach().numpy() <= age * 365.25).any(1) * (d[3].cpu().detach().numpy() >= age * 365.25).any(1))
u = np.unique(w[0])

d0 = d[0][u[:n_samples]].clone().detach()
d1 = d[1][u[:n_samples]].clone().detach()

d0[d1>age*365.25] = 0
d1[d1>age*365.25] = -10000.

if age > 0:
    d0 = torch.nn.functional.pad(d0, (0,1), 'constant', 1)
    d1 = torch.nn.functional.pad(d1, (0,1), 'constant', age*365.25)

o = d1.argsort(1)
d0 = d0.gather(1, o)
d1 = d1.gather(1, o)
# -



if must_have_lifestyle:
    have_lifestyle = torch.isin(d0, torch.arange(4, 13)).any(dim=1)
    (~have_lifestyle).sum().item(), have_lifestyle.sum().item()
    d0 = d0[have_lifestyle, :]
    d1 = d1[have_lifestyle, :]
    print(f"{d0.shape[0]}/{have_lifestyle.numel()} participants with lifestyle tokens before age {age}")







# +
from tqdm import tqdm

batch_size = 128
n_repeats = 1 # one good make the estimates calculated from synthetic data less noisy by averaging over multiple resamples
oo = []
inc = []
model.to(device)
for _ in range(n_repeats):
    pbar = tqdm(zip(*map(lambda x: torch.split(x, batch_size), (d0,d1))), total=len(d0)//batch_size + 1)
    with torch.no_grad():
        for dd in pbar:
            _idx = dd[0].to(device)
            _age = dd[1].to(device)
            trim_margin = torch.min(torch.sum(_idx == 0, dim=1)).item()
            _idx = _idx[:, trim_margin:]
            _age = _age[:, trim_margin:]
            block_size = _idx.shape[1]
            
            mm = legacy_generate(
                model=model,
                idx=_idx,
                age=_age,
                max_age=85*365.25,
                no_repeat=True,
                max_new_tokens=128,
                termination_tokens=[1269],
                stop_at_block_size=stop_at_block_size
            )

            pbar.set_postfix({
                "trim margin": trim_margin,
                "prompt block size": block_size,
                "total block size": mm[0].shape[1]
            })
            
            oo += [(mm[0],mm[1])]
            log_hazard_rates = mm[2].clone()
            timestep = mm[1].clone()
            batch_tokens = mm[0].clone()
            sort_by_age = torch.argsort(timestep, dim=1)
            log_hazard_rates = torch.take_along_dim(input=log_hazard_rates, indices=sort_by_age.unsqueeze(-1), dim=1)
            timestep = torch.take_along_dim(input=timestep, indices=sort_by_age, dim=1)
            batch_tokens = torch.take_along_dim(input=batch_tokens, indices=sort_by_age, dim=1)

            log_hazard_rates[log_hazard_rates == -torch.inf] = torch.nan
            hazard_rates = log_hazard_rates[:, :-1].exp()

            last_time_by_event = timestep.max(dim=1, keepdim=True)[0].expand(-1, hazard_rates.shape[-1]).clone()
            last_time_by_event = last_time_by_event.scatter_(index=batch_tokens, src=timestep, dim=1)

            risk_by_age = list()
            for a in range(0, 80):

                _timestep = timestep.unsqueeze(-1)
                _timestep = torch.clamp(_timestep, min=a*365.25)
                _timestep = torch.clamp(_timestep, max=last_time_by_event.unsqueeze(1))
                _timestep = torch.clamp(_timestep, max=(a+1)*365.25)
                delta_t = torch.diff(_timestep, dim=1)
                delta_t[delta_t == 0] = torch.nan
                # not_enough_exposure = delta_t.sum(dim=1) < 365.25
                not_enough_exposure = torch.nansum(delta_t, dim=1) < 365.25

                cumul_hazard = delta_t * hazard_rates
                all_nan = torch.isnan(cumul_hazard).all(dim=1)
                cumul_hazard = torch.nansum(cumul_hazard, dim=1)
                # manually set sum of NaNs to Nan because torch.nansum over all NaNs returns 0 
                cumul_hazard[all_nan] = torch.nan

                cumul_hazard[not_enough_exposure] = torch.nan

                risk = 1 - torch.exp(-cumul_hazard)
                risk_by_age.append(risk)
            risk_by_age = torch.stack(risk_by_age, dim=1)
            inc.append(risk_by_age.detach().cpu())
# -


# +
real_idx = d[2][u[:n_samples]].cpu().detach().numpy().copy()
real_age = d[3][u[:n_samples]].cpu().detach().numpy().copy()
if must_have_lifestyle:
    real_idx = real_idx[have_lifestyle, :]
    real_age = real_age[have_lifestyle, :]
    print(f"{real_idx.shape[0]}/{have_lifestyle.numel()} participants with lifestyle tokens before age {age}")
a = real_idx
b = np.nan_to_num(real_age / 365.25, nan=-27).astype('int')#[:,24:]

real_inc = np.zeros((len(delphi_labels), 80))
for t in range(80):
    mask = (b == t)
    counts = np.bincount(a[mask], minlength=len(delphi_labels))
    real_inc[:,t] = counts.astype('float')
# real_inc /= (len(a) - np.histogram(b.max(1) + 1, np.arange(81))[0].cumsum())
real_inc /= (len(a) - np.histogram(b.max(1), np.arange(81))[0].cumsum())
# -



# +
max_len = max(x[1].shape[1] for x in oo)
a = [np.pad(x[0].cpu(), ((0,0), (0, max_len - x[0].shape[1])), constant_values=0) for x in oo]
b = [np.pad(x[1].cpu(), ((0,0), (0, max_len - x[1].shape[1])), constant_values=-10000) for x in oo]
a = np.concatenate(a, axis=0)
syn_idx = a
syn_age = np.concatenate(b, axis=0)
b = np.concatenate(b, axis=0) / 365.25
b = np.nan_to_num(b, nan=-27).astype('int')

syn_inc = np.zeros((len(delphi_labels), 80))
for t in range(80):
    mask = (b == t)
    counts = np.bincount(a[mask], minlength=len(delphi_labels))
    syn_inc[:,t] += counts.astype('float')

syn_inc /= (len(a) - np.histogram(b.max(1), np.arange(81))[0].cumsum())
# +
total_syn = (syn_age > 0).sum()
total_real = (real_age > 0).sum()
print(f"synthetic tokens (total): {total_syn}")
print(f"real tokens (total): {total_real}")

syn_after_prompt = (syn_age > age * 365.25).sum()
real_after_prompt = (real_age > age * 365.25).sum()
print(f"synethtic tokens (after prompt): {syn_after_prompt}")
print(f"real tokens (after prompt): {real_after_prompt}")

plt.hist(
    syn_age[syn_age > 0] / 365.25,
    bins=np.arange(0, 81),
    alpha=0.7
);
plt.hist(
    real_age[real_age > 0] / 365.25,
    bins=np.arange(0, 81),
    alpha=0.7
);
# -

plt.hist(
    syn_age[syn_idx == 1] / 365.25,
    bins=np.arange(0, 81),
    alpha=0.7
);
plt.hist(
    real_age[real_idx == 1] / 365.25,
    bins=np.arange(0, 81),
    alpha=0.7
);



plt.hist(
    syn_age[(syn_age > 0) & (syn_idx > 1)] / 365.25,
    bins=np.arange(0, 81),
    alpha=0.7
);
plt.hist(
    real_age[(real_age > 0) & (real_idx > 1)] / 365.25,
    bins=np.arange(0, 81),
    alpha=0.7
);

inc = torch.cat(inc, dim=0)
inc = torch.nanmean(inc, dim=0)
inc = inc.transpose(1, 0)
calc_surv = torch.cumprod(1 - inc, dim=1)

# +
vocab_size = hazard_rates.shape[-1]

def km_estimator(timestep, tokens):

    total_n = tokens.shape[0]

    surv_time = timestep.max(axis=1)[:, None]
    surv_time = np.repeat(
        surv_time, vocab_size, axis=1
    )
    np.put_along_axis(
        arr=surv_time,
        indices=tokens,
        values=timestep,
        axis=1
    )
    surv_time = surv_time.transpose(1, 0)
    
    occur = np.zeros((timestep.shape[0], 1))
    occur = np.repeat(occur, vocab_size, axis=1)
    np.put_along_axis(
        arr=occur,
        indices=tokens,
        values=1,
        axis=1
    )
    occur = occur.transpose(1, 0)
    
    sort_surv_time = np.argsort(surv_time, axis=1)
    surv_time = np.take_along_axis(surv_time, indices=sort_surv_time, axis=1)
    occur = np.take_along_axis(occur, indices=sort_surv_time, axis=1)

    km_estimator = list()
    km_timestep = list()
    for i in tqdm(range(vocab_size), total=vocab_size, leave=False):
        uniq_time, inverse_indices, n_exit = np.unique(
            surv_time[i, :], return_inverse=True, return_counts=True
        )
        n_exit = np.concatenate(([0], n_exit[:-1]))
        n_occur = np.bincount(inverse_indices, weights=occur[i, :])
        n_surv = total_n - np.cumsum(n_exit)
        hazard_rate = n_occur / n_surv
        km_estimator.append(np.cumprod(1 - hazard_rate))
        km_timestep.append(uniq_time)
        # assert len(uniq_time) == len(hazard_rate)

    return km_estimator, km_timestep


# +
a = [np.pad(x[0].cpu(), ((0,0), (0, max_len - x[0].shape[1])), constant_values=0) for x in oo]
b = [np.pad(x[1].cpu(), ((0,0), (0, max_len - x[1].shape[1])), constant_values=-10000) for x in oo]
tokens = np.concatenate(a, axis=0)
timestep = np.concatenate(b, axis=0)

syn_km_estimator, syn_surv_time = km_estimator(timestep, tokens)


# +
# tokens = d[2][u[:n_samples]].cpu().detach().numpy()
# timestep = d[3][u[:n_samples]].cpu().detach().numpy()

# tokens = tokens[have_lifestyle, :]
# timestep = timestep[have_lifestyle, :]

real_km_estimator, real_surv_time = km_estimator(real_age, real_idx)
# -



def incidence_within_range(surv_time, km, start_age, end_age):

    incidence = list()
    for token in range(vocab_size):
        _surv_time = surv_time[token]
        _prob = km[token]
        in_range = (_surv_time >= start_age * 365.25) & (_surv_time < end_age * 365.25)
        if in_range.sum() > 0 :
            _prob = _prob[in_range]
            incidence.append((_prob.max() - _prob.min()) / _prob.max())
        else:
            incidence.append(float("nan"))
    return np.array(incidence)



# +
start_age = 60
end_age = 79
calc = (calc_surv[:, start_age] - calc_surv[:, end_age]) / calc_surv[:, start_age] 
real = incidence_within_range(real_surv_time, real_km_estimator, start_age=start_age, end_age=end_age)
syn = incidence_within_range(syn_surv_time, syn_km_estimator, start_age=start_age, end_age=end_age)

plt.figure()
plt.scatter(calc[13:], syn[13:], marker=".", c=delphi_labels['color'][13:])
plt.plot([0,1],[0,1], c='k', ls=":")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("calculated")
plt.ylabel("simulated")
plt.title(f"probability of disease between age {start_age} and {end_age}")
plt.xlim(1e-5, 1)
plt.ylim(1e-5, 1)
# -







for start_age in [60, 65, 70]:
    end_age = start_age + 5
    calc = (calc_surv[:, start_age] - calc_surv[:, end_age]) / calc_surv[:, start_age] 
    real = incidence_within_range(real_surv_time, real_km_estimator, start_age=start_age, end_age=end_age)
    syn = incidence_within_range(syn_surv_time, syn_km_estimator, start_age=start_age, end_age=end_age)
    
    # plt.figure()
    # plt.scatter(calc[14:], syn[14:], marker=".", c=delphi_labels['color'][14:])
    # plt.plot([0,1],[0,1], c='k', ls=":")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.xlabel("calculated")
    # plt.ylabel("simulated")
    # plt.xlim(1e-5, 1)
    # plt.ylim(1e-5, 1)
    
    plt.figure()
    plt.scatter(syn[14:], real[14:], marker=".", c=delphi_labels['color'][14:])
    plt.plot([0,1],[0,1], c='k', ls=":")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('KM estimate from simulated data')
    plt.ylabel('KM estimate from observed data')
    plt.title(f"disease incidence from age {start_age} to {end_age}")
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    # plt.savefig(Path(ckpt) / f"gen_km_calibrate_start{start_age}_end{end_age}.png", dpi=300, bbox_inches="tight")



plt.figure()
token = -1
plt.plot(real_surv_time[token] / 365.25, real_km_estimator[token], label="observed", alpha=0.7)
plt.plot(syn_surv_time[token] / 365.25, syn_km_estimator[token], label="simulated", alpha=0.7)
plt.plot(np.arange(0, 80), calc_surv[token, :], label="calculated", alpha=0.7)
plt.legend()
plt.xlim(60, None)
plt.xlabel("age (years)")
plt.ylabel("S(t)")

# +
# a = d[2][u[:n_samples]].cpu().detach().numpy()
# b = np.nan_to_num(d[3].cpu().detach().numpy().copy()[u[:n_samples]] / 365.25, nan=-27).astype('int')#[:,24:]

# real_inc = np.zeros((len(delphi_labels), 80))
# for t in range(80):
#     mask = (b == t)
#     counts = np.bincount(a[mask], minlength=len(delphi_labels))
#     real_inc[:,t] = counts.astype('float')
# # real_inc /= (len(a) - np.histogram(b.max(1) + 1, np.arange(81))[0].cumsum())
# real_inc /= (len(a) - np.histogram(b.max(1), np.arange(81))[0].cumsum())
# -







# +
# chapter_order = ['Technical', 'Sex', 'Smoking, Alcohol and BMI',
#        'I. Infectious Diseases', 'II. Neoplasms', 'III. Blood & Immune Disorders',
#        'IV. Metabolic Diseases', 'V. Mental Disorders',
#        'VI. Nervous System Diseases', 'VII. Eye Diseases',
#        'VIII. Ear Diseases', 'IX. Circulatory Diseases',
#        'X. Respiratory Diseases', 'XI. Digestive Diseases',
#        'XII. Skin Diseases', 'XIII. Musculoskeletal Diseases',
#        'XIV. Genitourinary Diseases', 'XV. Pregnancy & Childbirth',
#        'XVI. Perinatal Conditions', 'XVII. Congenital Abnormalities', 'Death']

# start_age = 70
# end_age = 75
# ages_of_interest = np.arange(start_age, end_age)

# calculated = inc[:, ages_of_interest].mean(-1)
# simulated = syn_inc[:, ages_of_interest].mean(-1)
# observed = real_inc[:, ages_of_interest].mean(-1)
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(12,4))
# ax1.scatter(simulated, calculated, marker=".", c=delphi_labels['color'])
# ax1.set_xlabel("simulated")
# ax1.set_ylabel("calculated")
# ax1.set_xscale("log")
# ax1.set_yscale("log")
# ax1.set_xlim(1e-5,.1)
# ax1.set_ylim(1e-5,.1)
# ax1.plot([0,1],[0,1], c='k', ls=":")

# ax2.scatter(simulated, observed, marker=".", c=delphi_labels['color'])
# ax2.set_xlabel("simulated")
# ax2.set_ylabel("observed")
# ax2.plot([0,1],[0,1], c='k', ls=":")

# ax3.scatter(observed, calculated, marker=".", c=delphi_labels['color'])
# ax3.set_xlabel("observed")
# ax3.set_ylabel("calculated")
# ax3.plot([0,1],[0,1], c='k', ls=":")

# f.suptitle(f"comparison of simulated, calculated, and observed disease incidence \n between {start_age} and {end_age} generated after age {age}")

# import matplotlib.patches as mpatches
# handles = list()
# for chapter in chapter_order[3:-1]:
#     color = delphi_labels.loc[delphi_labels['ICD-10 Chapter (short)'] == chapter, 'color'].values[0]
#     handles.append(mpatches.Circle((0, 0), 1, facecolor=color, edgecolor=color, label=chapter))
# f.legend(handles=handles, loc='center left', bbox_to_anchor=(1.1, 0.5), frameon=False)

# plt.savefig(ckpt / "calc_simul_real.png", dpi=300, bbox_inches="tight")
# -





