import numpy as np
import polars as pl
from utils import (
    all_ukb_participants,
    build_expansion_pack,
    load_coding,
    load_fid,
    month_of_birth,
)

vocab = load_coding(240)
vocab = vocab.loc[~vocab["coding"].str.contains("Chapter")]
is_parent = vocab["selectable"] == "N"
is_child = vocab["selectable"] == "Y"
parent_codes = vocab.loc[is_parent, "coding"]
parent2id = {code: int(i + 1) for i, code in enumerate(parent_codes)}
child_codes = vocab.loc[is_child, "coding"]

vocab.loc[is_child, "parent"] = vocab.loc[is_child, "coding"].str[0:3]
vocab.loc[is_parent, "parent"] = vocab.loc[is_parent, "coding"]

vocab["parent_id"] = vocab["parent"].map(parent2id)
print(vocab[vocab["parent_id"].isna()])
vocab = vocab[~vocab["parent_id"].isna()]
vocab = vocab.set_index("coding")
mapping = vocab["parent_id"].astype(int).to_dict()

tokenizer_keys = (
    vocab.loc[vocab["parent"].unique(), "meaning"]
    .str.replace(" ", "_")
    .str.lower()
    .tolist()
)
tokenizer_values = vocab.loc[vocab["parent"].unique(), "parent_id"].astype(int).tolist()
tokenizer = dict(zip(tokenizer_keys, tokenizer_values))

token_df = pl.from_pandas(load_fid("41200").reset_index())
summary_ops_participants = token_df["f.eid"].to_numpy().astype(int)
token_df = token_df.drop("f.eid")
token_df = token_df.select(pl.all().replace_strict(mapping, default=0))
token_np = token_df.to_numpy().astype(int)

time_df = pl.from_pandas(load_fid("41260").reset_index())
time_df = time_df.drop("f.eid")
time_df = time_df.with_columns(
    pl.all().str.strptime(pl.Datetime, strict=False, format="%Y-%m-%d")
)

mob_df = pl.from_pandas(month_of_birth())

time_df = time_df.select(
    (pl.all() - mob_df["year_month"]).cast(pl.Duration).dt.total_days()
)
time_np = time_df.to_numpy().astype(np.float32)

ukb_participants = all_ukb_participants()
is_valid = np.isin(summary_ops_participants, ukb_participants)
valid_participants = summary_ops_participants[is_valid]

accept_mask = (token_np > 0) * (~np.isnan(time_np))
count_np = np.sum(accept_mask[is_valid], axis=1)
token_np = token_np[accept_mask].ravel()
time_np = time_np[accept_mask].ravel()

build_expansion_pack(
    token_np=token_np,
    time_np=time_np,
    count_np=count_np,
    subjects=valid_participants,
    tokenizer=tokenizer,
    expansion_pack="summary_ops",
)
