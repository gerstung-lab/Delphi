from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from utils import (
    all_ukb_participants,
    assessment_age,
    build_expansion_pack,
)

idir = Path("data/multimodal/medications")

vocab = pd.read_csv(
    idir / "coding.txt",
    sep="\t",
)
vocab["id"] = np.arange(1, len(vocab) + 1)
token_map = vocab.set_index("coding")

orig_token_df = pl.read_csv(
    idir / "medication_code_20003.txt",
    separator="\t",
)
medication_participants = orig_token_df["f.eid"].to_numpy().astype(int)
orig_token_df = orig_token_df.drop("f.eid")
token_df = orig_token_df.select(
    pl.all().replace_strict(token_map["id"].to_dict(), default=0)
)
vocab = vocab.set_index("id")

token_np = token_df.to_numpy().astype(int)
unique_tokens, token_counts = np.unique(token_np, return_counts=True)
accept_mask = unique_tokens != 0
unique_tokens = unique_tokens[accept_mask]
token_counts = token_counts[accept_mask]
vocab.loc[unique_tokens, "counts"] = token_counts.astype(int)
vocab = vocab.reset_index()
vocab = vocab.sort_values(by="counts", ascending=False)
vocab.to_csv(idir / "vocab.csv", index=False)

vocab = vocab[vocab["counts"] > 100]
vocab = vocab[vocab["coding"] != 99999]
vocab["id"] = np.arange(1, len(vocab) + 1)
token_map = vocab.set_index("coding")
vocab["meaning"] = (
    vocab["coding"].astype(str)
    + "_"
    + vocab["meaning"].str.replace(" ", "_").str.lower()
)
tokenizer = vocab.set_index("meaning")
tokenizer = tokenizer["id"].to_dict()

ukb_participants = all_ukb_participants()
first_assess_age = assessment_age(visits=["init_assess"])["init_assess"]
assessment_participants = list(first_assess_age.index.astype(int))
missing_meds = len(set(ukb_participants) - set(medication_participants))
print(f"# participants with missing medication data: {missing_meds}")
missing_age = len(set(ukb_participants) - set(assessment_participants))
print(f"# participants with missing age data: {missing_age}")
is_valid = np.isin(medication_participants, ukb_participants)
is_valid &= np.isin(medication_participants, assessment_participants)
valid_participants = medication_participants[is_valid]

token_df = orig_token_df.select(
    pl.all().replace_strict(token_map["id"].to_dict(), default=0)
)
token_np = token_df.to_numpy().astype(int)
accept_mask = token_np > 0
count_np = np.sum(accept_mask[is_valid], axis=1).astype(np.int32)
token_np = token_np[is_valid[:, np.newaxis] * accept_mask]
time_np = first_assess_age.loc[valid_participants].to_numpy()
time_np = np.repeat(time_np, count_np)

build_expansion_pack(
    token_np=token_np,
    time_np=time_np,
    count_np=count_np,
    subjects=valid_participants,
    tokenizer=tokenizer,
    expansion_pack="meds",
)
