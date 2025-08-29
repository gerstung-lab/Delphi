import os

import numpy as np
import yaml
from utils import (
    all_ukb_participants,
    assessment_age,
    build_expansion_pack,
    expansion_pack_dir,
    load_visit,
)

from delphi import DAYS_PER_YEAR

odir = os.path.join(expansion_pack_dir, "fsf")
with open(os.path.join(odir, "tokenizer.yaml"), "r") as f:
    tokenizer = yaml.safe_load(f)

sex = load_visit(fid="31", visit_idx=0)
females = list()
for pid in sex.keys():
    if sex[pid] == 0:
        females.append(pid)
females = np.array(females)
ukb_participants = all_ukb_participants()
females = females[np.isin(females, ukb_participants)].tolist()

first_assess_age = assessment_age(visits=["init_assess"])["init_assess"].to_dict()

age_menarche = load_visit(fid="2714", visit_idx=0)
had_menopause = load_visit(fid="2724", visit_idx=0)
age_menopause = load_visit(fid="3581", visit_idx=0)
number_of_live_births = load_visit(fid="2734", visit_idx=0)
age_primiparous = load_visit(fid="3872", visit_idx=0)
age_first_live_birth = load_visit(fid="2754", visit_idx=0)
had_failed_pregnancy = load_visit(fid="2774", visit_idx=0)
number_miscarriages = load_visit(fid="3839", visit_idx=0)

token_list = list()
time_list = list()
count_list = list()
for pid in females:
    pid_tokens = list()
    pid_times = list()

    age = age_menarche[pid]
    if age == -3 or age == -1 or np.isnan(age):
        pass  # TODO: unknown menarche age
    else:
        pid_tokens.append(tokenizer["menarche"])
        pid_times.append(age * DAYS_PER_YEAR)

    if had_menopause[pid] == 0:
        pass  # TODO: not yet reached menopause
    elif had_menopause[pid] == 1:
        age = age_menopause[pid]
        if age == -3 or age == -1 or np.isnan(age):
            pass  # TODO: reached menopause but age unknown
        else:
            pid_tokens.append(tokenizer["menopause"])
            pid_times.append(age * DAYS_PER_YEAR)
    else:
        pass  # TODO: unknown menopause status; underwent hysterectomy before

    n_parity = number_of_live_births[pid]
    if n_parity >= 0:
        if n_parity == 0:
            token = tokenizer["parity_zero"]
        elif n_parity == 1:
            token = tokenizer["parity_one"]
        elif n_parity == 2:
            token = tokenizer["parity_two"]
        else:
            assert n_parity >= 3
            token = tokenizer["parity_three_or_more"]
        pid_tokens.append(token)
        pid_times.append(first_assess_age[pid])
    else:
        pass  # TODO: unknown parity

    if n_parity == 1:
        age = age_primiparous[pid]
        if age == -3 or age == -4 or np.isnan(age):
            pass  # TODO: unknown age of first live birth
        else:
            pid_tokens.append(tokenizer["first_live_birth"])
            pid_times.append(age * DAYS_PER_YEAR)
    elif n_parity > 1:
        age = age_first_live_birth[pid]
        if age == -3 or age == -4 or np.isnan(age):
            pass  # TODO: unknown age of first live birth
        else:
            pid_tokens.append(tokenizer["first_live_birth"])
            pid_times.append(age * DAYS_PER_YEAR)
    else:
        pass  # TODO: parity unknown

    if had_failed_pregnancy[pid] == 1:
        n_miscarriages = number_miscarriages[pid]
        if n_miscarriages == -3 or n_miscarriages == -1 or np.isnan(n_miscarriages):
            pass  # TODO: unknown number of miscarriages
        elif n_miscarriages == 1:
            pid_tokens.append(tokenizer["miscarriage_one"])
            pid_times.append(first_assess_age[pid])
        elif n_miscarriages > 1:
            pid_tokens.append(tokenizer["miscarriage_multiple"])
            pid_times.append(first_assess_age[pid])
        else:
            pass  # TODO: other kinds of pregnancy termination
    else:
        pass  # TODO: unknown termination status

    token_list.extend(pid_tokens)
    time_list.extend(pid_times)
    assert len(pid_tokens) == len(pid_times)
    count_list.append(len(pid_tokens))


build_expansion_pack(
    token_np=np.array(token_list),
    time_np=np.array(time_list),
    count_np=np.array(count_list),
    subjects=np.array(females),
    tokenizer=tokenizer,
    expansion_pack="fsf",
)
