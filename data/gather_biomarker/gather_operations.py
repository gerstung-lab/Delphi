from pathlib import Path

import numpy as np
import pandas as pd
from utils import all_ukb_participants, build_expansion_pack

from delphi import DAYS_PER_YEAR

idir = Path("data/multimodal/operations")

vocab = pd.read_csv(
    idir / "coding.txt",
    sep="\t",
)
reject_vals = [-1, 99999]
vocab = vocab.loc[~vocab["coding"].isin(reject_vals)]
code_vals = vocab["coding"].unique()
code_map = {code: i + 1 for i, code in enumerate(code_vals)}
vocab = vocab.set_index("coding")
tokenizer_keys = (
    vocab.loc[code_vals, "meaning"].str.replace(" ", "_").str.lower().tolist()
)
tokenizer_values = code_map.values()
tokenizer = dict(zip(tokenizer_keys, tokenizer_values))

max_key = max(code_map.keys())
lookup = np.zeros((max_key + 1,), dtype=int)
for k, v in code_map.items():
    lookup[k] = v

token_df = pd.read_csv(
    idir / "operation_code_20004.txt",
    sep="\t",
    index_col="f.eid",
)
time_df = pd.read_csv(
    idir / "age_when_operation_20011.txt",
    sep="\t",
    index_col="f.eid",
)
ops_participants = time_df.index.to_numpy().astype(int)
ukb_participants = all_ukb_participants()
is_valid = np.isin(ops_participants, ukb_participants)
valid_participants = ops_participants[is_valid]

time_df = time_df.loc[valid_participants]
token_df = token_df.loc[valid_participants]
time_np = time_df.to_numpy().astype(np.float32)
time_np *= DAYS_PER_YEAR
token_np = token_df.to_numpy().astype(int)

accept_mask = (token_np > 0) * (token_np < 99999) * (time_np > 0)
count_np = np.sum(accept_mask, axis=1)
token_np = token_np[accept_mask].ravel()
time_np = time_np[accept_mask].ravel()
token_np = lookup[token_np]

build_expansion_pack(
    token_np=token_np,
    time_np=time_np,
    count_np=count_np,
    subjects=valid_participants,
    tokenizer=tokenizer,
    expansion_pack="ops",
)
