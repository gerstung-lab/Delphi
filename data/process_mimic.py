import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from safetensors import safe_open
from tqdm import tqdm

from delphi.env import DELPHI_DATA_DIR
from delphi.tokenizer import FEMALE, MALE, NO_EVENT, PADDING

raw_mimic_dir = Path(os.environ["RAW_MIMIC_DIR"])
mimic_dir = Path(DELPHI_DATA_DIR) / "mimic"
os.makedirs(mimic_dir, exist_ok=True)

vocab_df = pd.read_csv(raw_mimic_dir / "train/vocab_t4542.csv", header=None)
vocab_df = vocab_df.replace({"GENDER//F": FEMALE, "GENDER//M": MALE})
token_names = vocab_df[0].tolist()
token_names = np.array(token_names)
tokenizer_dict = dict(zip(token_names, np.arange(len(token_names))))

time_intervals = [
    "5m-15m",
    "15m-45m",
    "45m-1h15m",
    "1h15m-2h",
    "2h-3h",
    "3h-5h",
    "5h-8h",
    "8h-12h",
    "12h-18h",
    "18h-1d",
    "1d-2d",
    "2d-4d",
    "4d-7d",
    "7d-12d",
    "12d-20d",
    "20d-30d",
    "30d-2mt",
    "2mt-6mt",
    "=6mt",
]
time_tokens = [tokenizer_dict[interval] for interval in time_intervals]

non_time_tokens = token_names[~np.isin(token_names, time_intervals)]
non_time_tokens = np.concatenate((np.array([PADDING, NO_EVENT]), non_time_tokens))
remapped_tokenizer = dict(
    zip(non_time_tokens.tolist(), np.arange(len(non_time_tokens)).tolist())
)
tokenizer_path = Path(mimic_dir) / "tokenizer.yaml"
with open(tokenizer_path, "w") as f:
    yaml.dump(remapped_tokenizer, f, default_flow_style=False, sort_keys=False)


token_map = np.arange(len(tokenizer_dict))
is_time_token = np.zeros(len(tokenizer_dict))
is_time_token[time_tokens] = 1
token_map = token_map - np.cumsum(is_time_token)
token_map = token_map + 2  # +1 for padding and +1 for no event tokens


train_dir = mimic_dir / "train"
train_shard_fps = sorted(train_dir.glob("[0-9]*.safetensors"))
test_dir = mimic_dir / "test"
test_shard_fps = sorted(test_dir.glob("[0-9]*.safetensors"))
shard_fps = train_shard_fps + test_shard_fps

train_static_data = pickle.load((train_dir / "static_data.pickle").open("rb"))
test_static_data = pickle.load((test_dir / "static_data.pickle").open("rb"))

tokens = list()
timesteps = list()
pids = list()
start_pos = list()
seq_len = list()

train_pids = list()
test_pids = list()

offset = 0
for shard_fp in tqdm(shard_fps):

    with safe_open(shard_fp, framework="numpy") as f:

        shard_tokens = f.get_slice("tokens")
        shard_timesteps = f.get_slice("times")
        shard_pids = f.get_tensor("patient_ids")
        shard_start_pos = f.get_tensor("patient_offsets")

        for i in range(len(shard_pids)):

            pid = shard_pids[i]
            start = shard_start_pos[i]
            if i < len(shard_pids) - 1:
                end = shard_start_pos[i + 1]
            else:
                end = None
            pid_slice = slice(start, end)

            pid_tokens = shard_tokens[pid_slice]
            pid_timesteps = shard_timesteps[pid_slice]
            is_time_token = np.isin(pid_tokens, time_tokens)

            pid_tokens = pid_tokens[~is_time_token]
            pid_timesteps = pid_timesteps[~is_time_token]

            if "train" in str(shard_fp):
                pid_static = train_static_data[pid]
            elif "test" in str(shard_fp):
                pid_static = test_static_data[pid]
            else:
                raise ValueError

            if "MEDS_BIRTH" not in pid_static.keys():
                print(f"{pid} has no MEDS_BIRTH")
                continue

            time_of_birth = pid_static["MEDS_BIRTH"]["time"]
            pid_timesteps -= time_of_birth
            pid_timesteps = pid_timesteps.astype(float)

            tokens.extend(pid_tokens)
            timesteps.extend(pid_timesteps)
            pids.append(pid)
            if "train" in str(shard_fp):
                train_pids.append(pid)
            elif "test" in str(shard_fp):
                test_pids.append(pid)
            else:
                raise ValueError
            seq_len.append(len(pid_tokens))
            start_pos.append(offset)

            offset += len(pid_tokens)

tokens = np.array(tokens)
tokens = token_map[tokens]
timesteps = np.array(timesteps) / 1e6 / 60  # convert from microsec to minutes

p2i = pd.DataFrame({"start_pos": start_pos, "seq_len": seq_len, "pid": pids})
p2i.to_csv(mimic_dir / "p2i.csv")
tokens.astype(np.uint32).tofile(mimic_dir / "data.bin")
timesteps.astype(np.uint32).tofile(mimic_dir / "time.bin")

train_pids = np.array(train_pids)
test_pids = np.array(test_pids)
subject_lst_dir = mimic_dir / "participants"
os.makedirs(subject_lst_dir, exist_ok=True)
train_pids.astype(np.uint32).tofile(subject_lst_dir / "train_fold.bin")
test_pids.astype(np.uint32).tofile(subject_lst_dir / "val_fold.bin")
