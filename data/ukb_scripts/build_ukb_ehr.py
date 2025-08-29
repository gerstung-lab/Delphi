from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from utils import build_expansion_pack

from delphi.data.utils import update_tokenizer

data_dir = Path("data/ukb_real_data")

expansion_packs = ["..", "meds", "ops", "prescriptions", "summary_ops"]


offsets = list()
tokens = list()
timesteps = list()
p2is = list()
for i, expansion in enumerate(expansion_packs):
    expansion_dir = data_dir / "expansion_packs" / expansion
    with open(expansion_dir / "tokenizer.yaml", "r") as f:
        addon_tokenizer = yaml.safe_load(f)
    if i == 0:
        tokenizer = addon_tokenizer
        offset = 0
    else:
        tokenizer, offset = update_tokenizer(tokenizer, addon_tokenizer)
    offsets.append(offset)
    tokens.append(np.memmap(expansion_dir / "data.bin", dtype=np.uint32, mode="r"))
    timesteps.append(np.memmap(expansion_dir / "time.bin", dtype=np.uint32, mode="r"))
    p2is.append(pd.read_csv(expansion_dir / "p2i.csv", index_col="pid"))


all_tokens = list()
all_timesteps = list()
p2i = p2is[0]
participants = p2i.index.tolist()
# participants = participants[:1000]
seq_lens = list()
start_poses = list()
start_pos = 0
for i in tqdm(participants, total=len(p2i), leave=False):
    seq_len = 0
    for j in range(len(expansion_packs)):
        s = p2is[j].loc[i, "start_pos"]
        l = p2is[j].loc[i, "seq_len"]
        all_tokens.append(tokens[j][s : s + l] + offsets[j])
        all_timesteps.append(timesteps[j][s : s + l])
        seq_len += l
    seq_lens.append(seq_len)
    start_poses.append(start_pos)
    start_pos += seq_len

all_tokens = np.concatenate(all_tokens)
all_timesteps = np.concatenate(all_timesteps)

build_expansion_pack(
    token_np=all_tokens,
    time_np=all_timesteps,
    count_np=np.array(seq_lens),
    subjects=np.array(participants),
    tokenizer=tokenizer,
    expansion_pack="ukb_ehr",
)
