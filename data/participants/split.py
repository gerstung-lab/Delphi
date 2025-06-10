import os

import numpy as np
import pandas as pd

from delphi.data.dataset import get_p2i

# from delphi.data.lmdb import estimate_write_size, write_lmdb
from delphi.env import DELPHI_DATA_DIR


def n_duplicate(arr: np.ndarray) -> int:
    return arr.size - np.unique(arr).size


base_dir = os.path.join(DELPHI_DATA_DIR, "ukb_real_data")
train_path = os.path.join(base_dir, "train.bin")
train_data = np.fromfile(train_path, dtype=np.uint32).reshape(-1, 3)

train_p2i = get_p2i(train_data)
train_subjects = train_data[:, 0][train_p2i[:, 0]]
if n_duplicate(train_subjects) > 0:
    print(f"found {n_duplicate(train_subjects)} duplicate subjects in train fold")
    subjects, counts = np.unique(train_subjects, return_counts=True)
    duplicates = subjects[counts > 1]
    for dup in duplicates:
        dup_tokens = train_data[train_data[:, 0] == dup, 2]
        dup_p2i = train_p2i[train_subjects == dup]
        assert n_duplicate(dup_tokens) == 0

sort_idx = np.lexsort((train_data[:, 1], train_data[:, 0]))
train_data = train_data[sort_idx]
train_p2i = get_p2i(train_data)
train_subjects = train_data[:, 0][train_p2i[:, 0]]
n_train = train_p2i.shape[0]

val_path = os.path.join(base_dir, "val.bin")
val_data = np.fromfile(val_path, dtype=np.uint32).reshape(-1, 3)
val_p2i = get_p2i(val_data)
val_subjects = val_data[:, 0][val_p2i[:, 0]]
if n_duplicate(val_subjects) > 0:
    print(f"found {n_duplicate(val_subjects)} duplicate subjects in val fold")
    subjects, counts = np.unique(val_subjects, return_counts=True)
    duplicates = subjects[counts > 1]
    for dup in duplicates:
        dup_tokens = val_data[val_data[:, 0] == dup, 2]
        dup_p2i = val_p2i[val_subjects == dup]
        assert n_duplicate(dup_tokens) == 0

sort_idx = np.lexsort((val_data[:, 1], val_data[:, 0]))
val_data = val_data[sort_idx]
val_p2i = get_p2i(val_data)
val_subjects = val_data[:, 0][val_p2i[:, 0]]
n_val = val_p2i.shape[0]

assert np.isin(
    train_subjects, val_subjects, invert=True
).all(), "Train and validation subjects overlap."
combined = np.concatenate((train_data, val_data), axis=0)
p2i = get_p2i(combined)
seq_len = p2i[:, 1]
start_pos = np.cumsum(p2i[:, 1]) - p2i[:, 1]
subjects = combined[:, 0][p2i[:, 0]]

lookup = pd.DataFrame({"pid": subjects, "start_pos": start_pos, "seq_len": seq_len})
lookup = lookup.set_index("pid")
lookup.to_csv(os.path.join(base_dir, "p2i.csv"), index=True)

# lookup = dict()
# for i, (subject, start, length) in enumerate(zip(subjects, start_pos, seq_len)):
#     lookup[str(subject).encode("utf-8")] = np.array([start, length], dtype=np.uint32)

# map_size = estimate_write_size(lookup, safety_margin=0.5, n_samples=100)
# write_lmdb(lookup, os.path.join(base_dir, "ukb.lmdb"), map_size)

participant_dir = os.path.join(base_dir, "participants")
train_subjects.tofile(os.path.join(participant_dir, "train_fold.bin"))
val_subjects.tofile(os.path.join(participant_dir, "val_fold.bin"))
timesteps = combined[:, 1]
tokens = combined[:, 2]
timesteps.tofile(os.path.join(base_dir, "time.bin"))
tokens.tofile(os.path.join(base_dir, "data.bin"))
