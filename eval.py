from scipy.special import logsumexp
import scipy.stats
import scipy
import warnings
import os
import pickle
import torch
from model import DelphiConfig, Delphi
from tqdm import tqdm
import pandas as pd
import numpy as np
import textwrap
import argparse
from utils import get_batch, get_p2i
from pathlib import Path


import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'axes.grid': True,
                     'grid.linestyle': ':',
                     'axes.spines.bottom': False,
                     'axes.spines.left': False,
                     'axes.spines.right': False,
                     'axes.spines.top': False})
plt.rcParams['figure.dpi'] = 72
plt.rcParams['pdf.fonttype'] = 42


delphi_labels = pd.read_csv('delphi_labels_chapters_colours_icd.csv')
labels = pd.read_csv("data/ukb_simulated_data/labels.csv", header=None, sep="\t")

dataset_subset_size = 50_000

def pad_to_length(a, length):
    if len(a) >= length:
        return a
    return np.pad(a, (0, length - len(a)), 'constant', constant_values=-1)


# argparse input path, output path, model weights path
# input path: path to the dataset
# output path: path to the output
# model weights path: path to the model weights
parser = argparse.ArgumentParser(description='Evaluate AUC')
parser.add_argument('--input_path', type=str, help='Path to the dataset')
parser.add_argument('--output_path', type=str, help='Path to the output')
parser.add_argument('--model_ckpt_path', type=str, help='Path to the model weights')
parser.add_argument('--no_event_token_rate', type=int, help='Path to the model weights')

# python evaluate_auc.py --input_path data/ukb_real_data --model_ckpt_path Delphi-2M/ckpt.pt --output_path auc_test/no_perm_kek --no_event_token_rate 5

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
no_event_token_rate = args.no_event_token_rate

# creare output folder
Path(output_path).mkdir(exist_ok=True, parents=True)

# out_dir = model_weights_path
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float32'  # 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = {'float32': torch.float32, 'float64': torch.float64, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# ckpt_path = os.path.join(out_dir, 'ckpt.pt')
ckpt_path = args.model_ckpt_path
checkpoint = torch.load(ckpt_path, map_location=device)
conf = DelphiConfig(**checkpoint['model_args'])
model = Delphi(conf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)

model.eval()
model = model.to(device)

train = np.fromfile(f'{input_path}/train.bin', dtype=np.uint32).reshape(-1, 3).astype(np.int64)
val = np.fromfile(f'{input_path}/val.bin', dtype=np.uint32).reshape(-1, 3).astype(np.int64)

train_p2i = get_p2i(train)
val_p2i = get_p2i(val)

females = train[np.isin(train[:, 0], train[train[:, 2] == 1, 0])]
males = train[np.isin(train[:, 0], train[train[:, 2] == 2, 0])]

token_count_males = np.bincount(males[:, 2], minlength=1269)
token_count_females = np.bincount(females[:, 2], minlength=1269)
df = pd.DataFrame({
    'token_name': labels[0],
    'n Male': [0, 0] + token_count_males.tolist()[1:],
    'n Female': [0, 0] + token_count_females.tolist()[1:]
    })
df['n_total'] = df['n Female'] + df['n Male']
df = df.reset_index()



d100k = get_batch(range(dataset_subset_size), val, val_p2i,
                  select='left', block_size=80,
                  device=device, padding='random', no_event_token_rate=no_event_token_rate)


warnings.filterwarnings('ignore')


def auc(x1, x2):
    n1 = len(x1)
    n2 = len(x2)
    R1 = np.concatenate([x1, x2]).argsort().argsort()[:n1].sum() + n1
    U1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    if n1 == 0 or n2 == 0:
        return np.nan
    return U1 / n1 / n2


def get_calibration_auc(j, k, d, p, offset=365.25, age_groups=range(45, 85, 5), bins=10**np.arange(-6., 1.5, .5), precomputed_idx=None):

    l = len(age_groups)
    age_step = age_groups[1] - age_groups[0]

    # Indexes of cases with disease k
    wk = np.where(d[2] == k)

    if len(wk[0]) < 2:
        return np.repeat(np.nan, l)

    # For controls, we need to exclude cases with disease k
    wc = np.where((d[2] != k) * (~(d[2] == k).any(-1))[..., None])
    
    wall = (np.concatenate([wk[0], wc[0]]), np.concatenate([wk[1], wc[1]])) # All cases and controls

    # We need to take into account the offset t and use the tokens for prediction that at least t before the event
    if precomputed_idx is None:
        pred_idx = (d[1][wall[0]] <= d[3][wall].reshape(-1, 1) - offset).sum(1) - 1
    else:
        pred_idx = precomputed_idx[wall] # It's actually much faster to precompute this

    z = d[1][(wall[0], pred_idx)] # Times of the tokens for prediction
    z = z[pred_idx != -1]

    zk = d[3][wall]  # Target times
    zk = zk[pred_idx != -1]

    x = np.exp(p[..., j][(wall[0], pred_idx)]) * 365.25
    x = x[pred_idx != -1]
    x = 1 - np.exp(-x * age_step) 

    wk = (wk[0][pred_idx[:len(wk[0])] != -1], wk[1][pred_idx[:len(wk[0])] != -1])
    p_idx = wall[0][pred_idx != -1]

    out = []

    for i, aa in enumerate(age_groups):
        a = np.logical_and(z / 365.25 >= aa, z / 365.25 < aa + age_step)
        # a *= zk - z < 365.25  # * age_step

        selected_groups = p_idx[a]
        perm = np.random.permutation(len(selected_groups))
        _, indices = np.unique(selected_groups[perm], return_index=True)
        indices = perm[indices]
        selected = np.zeros(np.sum(a), dtype=bool)
        selected[indices] = True
        a[a] = selected

        y = auc(x[len(wk[0]):][a[len(wk[0]):]], x[:len(wk[0])][a[:len(wk[0])]])
        out_item = {'token': k, 'auc': y,
                    'age': aa, 'n_healthy': len(x[len(wk[0]):][a[len(wk[0]):]]),
                    'n_diseased': len(x[:len(wk[0])][a[:len(wk[0])]])}
        out.append(out_item)
    return out


all_aucs = []

df_filtered = df[df['n_total'] > 100]
diseases_for_auc = df_filtered.iloc[13:].sort_values('n_total', ascending=False).index.values

# split diseases to chunks of 64
diseases_for_auc_chunks = np.array_split(diseases_for_auc, len(diseases_for_auc) // 300 + 1)
pred_idx_precompute = ((d100k[1][:, :, np.newaxis] < d100k[3][:, np.newaxis, :] - 0.1).sum(1) - 1)

for diseases_for_auc in tqdm(diseases_for_auc_chunks):
    # device = 'cuda'
    p100k = []
    model.to(device)
    batch_size = 128
    with torch.no_grad():
        for dd in tqdm(zip(*map(lambda x: torch.split(x, batch_size), d100k)), total=d100k[0].shape[0] // batch_size + 1):
            p100k.append(model(*[x.to(device) for x in dd])[0].cpu().detach()[:, :, diseases_for_auc].numpy().astype('float32'))
    p100k = np.vstack(p100k)

    ii_male = (d100k[0] == 3).sum(1) > 0
    p100k_male = p100k[ii_male.cpu()]

    ii_female = (d100k[0] == 2).sum(1) > 0
    p100k_female = p100k[ii_female.cpu()]

    for j, k in tqdm(list(enumerate(diseases_for_auc))):
        for sex, sex_idx, pp_sex in [('female', 2, p100k_female), ('male', 3, p100k_male)]:
            try:
                ii = (d100k[0] == sex_idx).sum(1) > 0
                out = get_calibration_auc(
                    j,
                    k,
                    (d100k[0][ii].cpu().detach().numpy(),
                    d100k[1][ii].cpu().detach().numpy(),
                        d100k[2][ii].cpu().detach().numpy(),
                        d100k[3][ii].cpu().detach().numpy()),
                    pp_sex,
                    age_groups=np.arange(
                        40,
                        80,
                        5),
                    offset=0.1,
                    precomputed_idx=pred_idx_precompute[ii].cpu().detach().numpy())
                for out_ in out:
                    out_['sex'] = sex
                    out_['token_name'] = labels.iloc[k][0]
                all_aucs += out
            except Exception as e:
                # expect TypeError: 'NoneType' object is not subscriptable
                if 'numpy.float64' not in repr(e):
                    print(repr(e))
                pass


df_auc_unpooled = pd.DataFrame(all_aucs)
df_auc_unpooled.to_csv(f'{output_path}/df_auc_unpooled.csv')

df_auc = df_auc_unpooled.groupby(['token', 'sex', 'token_name'])[['auc']].mean().reset_index()
df_auc = df_auc.pivot(index=['token', 'token_name'], columns='sex', values='auc')
df_auc.columns = ['auc_female', 'auc_male']
df_auc.reset_index(inplace=True)

df_both = df.merge(df_auc, left_on='token_name', right_on='token_name')
df_both.to_csv(f'{output_path}/df_both.csv')