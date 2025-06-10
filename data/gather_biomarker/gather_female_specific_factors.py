import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml
from utils import (
    MULTIMODAL_INPUT_DIR,
    all_ukb_participants,
    assessment_age,
    expansion_pack_dir,
    sex,
)

from delphi import DAYS_PER_YEAR
from delphi.data.lmdb import data_key, time_key

fsf_dir = Path(os.path.join(MULTIMODAL_INPUT_DIR, "female-specific-factors"))
fsf = {}
for file in fsf_dir.rglob("*"):
    if file.is_file():
        key = file.stem
        key = key.split("_")[-1]
        fsf[key] = str(file)


def load_first_visit(field_id: Union[str, int]) -> dict:

    df = pd.read_csv(fsf[str(field_id)], delimiter="\t", index_col="f.eid")

    return df.iloc[:, 0].to_dict()


with open("data/gather_biomarker/female_specific_factors/dictionary.yaml", "r") as f:
    field_dict = yaml.safe_load(f)

with open(
    "data/gather_biomarker/female_specific_factors/tokenizer/verbose.yaml", "r"
) as f:
    tokenizer = yaml.safe_load(f)

pids = list()
for pid in sex.keys():
    if sex[pid] == 0:
        pids.append(pid)

age_menarche = load_first_visit(field_dict["age_menarche"]["id"])
menarche = dict()
for i, pid in enumerate(pids):
    age = age_menarche[pid]
    if age == -3 or age == -1 or np.isnan(age):
        # TODO: unknown menarche age
        # should we assume everyone has reached menarche by assessment age?
        continue
    else:
        age *= DAYS_PER_YEAR
        token = tokenizer["menarche"]

    menarche[data_key(pid)] = np.array([token], dtype=np.uint8)
    menarche[time_key(pid)] = np.array([age], dtype=np.uint32)


had_menopause = load_first_visit(field_dict["had_menopause"]["id"])
age_menopause = load_first_visit(field_dict["age_menopause"]["id"])
menopause = dict()
for i, pid in enumerate(pids):

    if had_menopause[pid] == 0:
        # TODO: not yet reached menopause
        continue
    elif had_menopause[pid] == 1:
        age = age_menopause[pid]
        if age in field_dict["age_menopause"]["unknown"] or np.isnan(age):
            # TODO: reached menopause but age unknown
            # how many participants actually fall in this category?
            continue
        else:
            age *= DAYS_PER_YEAR
            token = tokenizer["menopause"]
    else:
        # TODO: unknown menopause status; underwent hysterectomy before
        # how many participants actually fall in this category?
        continue

    menopause[data_key(pid)] = np.array([token], dtype=np.uint8)
    menopause[time_key(pid)] = np.array([age], dtype=np.uint32)


number_of_live_births = load_first_visit(field_dict["number_of_live_births"]["id"])
age_primiparous = load_first_visit(field_dict["age_primiparous"]["id"])
age_first_live_birth = load_first_visit(field_dict["age_first_live_birth"]["id"])

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
        if age in field_dict["age_primiparous"]["unknown"] or np.isnan(age):
            # TODO: unknown age of first live birth
            continue
        else:
            age *= DAYS_PER_YEAR
            token = tokenizer["first_live_birth"]
    elif n_parity > 1:
        age = age_first_live_birth[pid]
        if age in field_dict["age_first_live_birth"]["unknown"] or np.isnan(age):
            # TODO: unknown age of first live birth
            continue
        else:
            age *= DAYS_PER_YEAR
            token = tokenizer["first_live_birth"]
    else:
        # TODO: parity unknown
        # how many participants actually fall in this category?
        continue

    first_live_birth[data_key(pid)] = np.array([token], dtype=np.uint8)
    first_live_birth[time_key(pid)] = np.array([age], dtype=np.uint32)


had_failed_pregnancy = load_first_visit(
    field_dict["had_stillbirth_miscarriage_termination"]["id"]
)
number_miscarriages = load_first_visit(field_dict["number_miscarriages"]["id"])
termination = dict()
for i, pid in enumerate(pids):

    if had_failed_pregnancy[pid] == 0:
        continue
    elif had_failed_pregnancy[pid] == 1:
        n_miscarriages = number_miscarriages[pid]
        if n_miscarriages in field_dict["number_miscarriages"]["unknown"] or np.isnan(
            n_miscarriages
        ):
            # TODO: unknown number of miscarriages
            # how many participants actually fall in this category?
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
    [time_steps[pid] for pid in participants], axis=0, dtype=np.uint32
)

lookup = pd.DataFrame(
    {
        "pid": participants,
        "start_pos": np.cumsum(token_counts) - np.array(token_counts),
        "seq_len": token_counts,
    }
)

write_dir = os.path.join(expansion_pack_dir, "fsf")
os.makedirs(write_dir, exist_ok=True)
lookup = lookup.set_index("pid")
lookup.to_csv(os.path.join(write_dir, "p2i.csv"), index=True)
tokens.tofile(os.path.join(write_dir, "data.bin"))
time_steps.tofile(os.path.join(write_dir, "time.bin"))
with open(os.path.join(write_dir, "tokenizer.yaml"), "w") as f:
    yaml.dump(tokenizer, f, allow_unicode=True)


# from delphi.data.lmdb import estimate_write_size, write_lmdb
# fsf_db = dict()
# for data in data_list:
#     for key, value in data.items():
#         if key not in fsf_db:
#             fsf_db[key] = value
#         else:
#             fsf_db[key] = np.concatenate((fsf_db[key], value), axis=0)
# map_size = estimate_write_size(fsf_db, safety_margin=1.0)
# db_path = os.path.join(write_dir, "data.lmdb")
# write_lmdb(db=fsf_db, db_path=db_path, map_size=map_size)
# with open(os.path.join(write_dir, "tokenizer.yaml"), "w") as f:
#     yaml.dump(tokenizer, f, allow_unicode=True)
