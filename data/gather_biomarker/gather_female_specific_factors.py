import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from utils import (
    all_ukb_participants,
    assessment_age,
    data_key,
    expansion_pack_dir,
    load_visit,
    sex,
    time_key,
)

from delphi import DAYS_PER_YEAR

odir = os.path.join(expansion_pack_dir, "fsf")
with open(os.path.join(odir, "tokenizer.yaml"), "r") as f:
    tokenizer = yaml.safe_load(f)
token_list_dir = Path("config/disease_list")
with open(token_list_dir / "fsf.yaml", "w") as f:
    yaml.dump(
        list(tokenizer.keys()),
        f,
        default_flow_style=False,
        sort_keys=False,
    )

pids = list()
for pid in sex.keys():
    if sex[pid] == 0:
        pids.append(pid)

age_menarche = load_visit(fid="2714", visit_idx=0)
menarche = dict()
for i, pid in enumerate(pids):
    age = age_menarche[pid]
    if age == -3 or age == -1 or np.isnan(age):
        # TODO: unknown menarche age
        continue
    else:
        age *= DAYS_PER_YEAR
        token = tokenizer["menarche"]

    menarche[data_key(pid)] = np.array([token], dtype=np.uint8)
    menarche[time_key(pid)] = np.array([age], dtype=np.uint32)


had_menopause = load_visit(fid="2724", visit_idx=0)
age_menopause = load_visit(fid="3581", visit_idx=0)
menopause = dict()
for i, pid in enumerate(pids):

    if had_menopause[pid] == 0:
        # TODO: not yet reached menopause
        continue
    elif had_menopause[pid] == 1:
        age = age_menopause[pid]
        if age == -3 or age == -1 or np.isnan(age):
            # TODO: reached menopause but age unknown
            continue
        else:
            age *= DAYS_PER_YEAR
            token = tokenizer["menopause"]
    else:
        # TODO: unknown menopause status; underwent hysterectomy before
        continue

    menopause[data_key(pid)] = np.array([token], dtype=np.uint8)
    menopause[time_key(pid)] = np.array([age], dtype=np.uint32)


number_of_live_births = load_visit(fid="2734", visit_idx=0)
age_primiparous = load_visit(fid="3872", visit_idx=0)
age_first_live_birth = load_visit(fid="2754", visit_idx=0)

parity = dict()
for pid in pids:
    n_parity = number_of_live_births[pid]
    if n_parity == 0:
        token = tokenizer["parity_zero"]
    elif n_parity == 1:
        token = tokenizer["parity_one"]
    elif n_parity == 2:
        token = tokenizer["parity_two"]
    elif n_parity >= 3:
        token = tokenizer["parity_three_or_more"]
    else:
        # TODO: unknown parity
        continue

    parity[data_key(pid)] = np.array([token], dtype=np.uint8)
    parity[time_key(pid)] = np.array([assessment_age[pid]], dtype=np.uint32)


first_live_birth = dict()
for pid in pids:
    n_parity = number_of_live_births[pid]
    if n_parity == 0:
        continue
    elif n_parity == 1:
        age = age_primiparous[pid]
        if age == -3 or age == -4 or np.isnan(age):
            # TODO: unknown age of first live birth
            continue
        else:
            age *= DAYS_PER_YEAR
            token = tokenizer["first_live_birth"]
    elif n_parity > 1:
        age = age_first_live_birth[pid]
        if age == -3 or age == -4 or np.isnan(age):
            # TODO: unknown age of first live birth
            continue
        else:
            age *= DAYS_PER_YEAR
            token = tokenizer["first_live_birth"]
    else:
        # TODO: parity unknown
        continue

    first_live_birth[data_key(pid)] = np.array([token], dtype=np.uint8)
    first_live_birth[time_key(pid)] = np.array([age], dtype=np.uint32)


had_failed_pregnancy = load_visit(fid="2774", visit_idx=0)
number_miscarriages = load_visit(fid="3839", visit_idx=0)
termination = dict()
for i, pid in enumerate(pids):

    if had_failed_pregnancy[pid] == 0:
        continue
    elif had_failed_pregnancy[pid] == 1:
        n_miscarriages = number_miscarriages[pid]
        if n_miscarriages == -3 or n_miscarriages == -1 or np.isnan(n_miscarriages):
            # TODO: unknown number of miscarriages
            continue
        elif n_miscarriages == 1:
            token = tokenizer["miscarriage_one"]
        elif n_miscarriages > 1:
            token = tokenizer["miscarriage_multiple"]
        else:
            # TODO: other kinds of pregnancy termination
            continue
    else:
        # TODO: unknown termination status
        continue

    termination[data_key(pid)] = np.array([token], dtype=np.uint8)
    termination[time_key(pid)] = np.array([assessment_age[pid]], dtype=np.uint32)


participants = all_ukb_participants()
data_list = [menarche, menopause, parity, first_live_birth, termination]
tokens = dict()
time_steps = dict()
token_counts = []
for pid in participants:
    if pid not in tokens.keys():
        tokens[pid] = np.array([], dtype=np.uint32)
        time_steps[pid] = np.array([], dtype=np.uint32)
    for data in data_list:
        if data_key(pid) not in data:
            continue
        else:
            tokens[pid] = np.concatenate((tokens[pid], data[data_key(pid)]), axis=0)
            time_steps[pid] = np.concatenate(
                (time_steps[pid], data[time_key(pid)]), axis=0
            )
    token_counts.append(tokens[pid].size)
token_counts = np.array(token_counts)

pids = np.repeat(np.array(participants), token_counts)
tokens = np.concatenate([tokens[pid] for pid in participants], axis=0, dtype=np.uint32)
time_steps = np.concatenate(
    [time_steps[pid] for pid in participants], axis=0, dtype=np.float32
)

p2i = pd.DataFrame(
    {
        "pid": participants,
        "start_pos": np.cumsum(token_counts) - np.array(token_counts),
        "seq_len": token_counts,
    }
)

os.makedirs(odir, exist_ok=True)
p2i = p2i.set_index("pid")
p2i.loc[p2i["seq_len"] == 0, "start_pos"] = 0
p2i.to_csv(os.path.join(odir, "p2i.csv"), index=True)
np.save(os.path.join(odir, "data.npy"), tokens)
np.save(os.path.join(odir, "time.npy"), time_steps)
with open(os.path.join(odir, "tokenizer.yaml"), "w") as f:
    yaml.dump(tokenizer, f, allow_unicode=True)
