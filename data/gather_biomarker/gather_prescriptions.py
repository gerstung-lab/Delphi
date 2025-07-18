import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from utils import all_ukb_participants, expansion_pack_dir, multimodal_dir

idir = Path(multimodal_dir) / "primary_care"
odir = Path(expansion_pack_dir) / "prescriptions"

bnf_lkp = pd.read_csv(idir / "bnf_lkp.csv")
is_dummy = bnf_lkp["BNF_Subparagraph"].str.contains("DUMMY", na=False)
bnf_lkp = bnf_lkp.loc[~is_dummy]
bnf_codes = bnf_lkp["BNF_Presentation_Code"].str.replace(".", "").str[:6]
is_duplicate = bnf_codes.duplicated(keep="first")

unique_bnf_codes = bnf_codes.loc[~is_duplicate].values
tokenizer_keys = bnf_lkp.loc[~is_duplicate, "BNF_Subparagraph"]
tokenizer_values = np.arange(len(tokenizer_keys.unique()), dtype=np.int32) + 1
tokenizer = pd.Series(tokenizer_values, index=tokenizer_keys.unique())
with open(odir / "tokenizer.yaml", "w") as f:
    saved_tokenizer = {}
    for key, value in tokenizer.to_dict().items():
        key = key.lower().replace(" ", "_")
        saved_tokenizer[key] = int(value)
    yaml.dump(saved_tokenizer, f, default_flow_style=False)
token_list_dir = Path("config/disease_list")
with open(token_list_dir / "prescriptions.yaml", "w") as f:
    yaml.dump(
        list(saved_tokenizer.keys()),
        f,
        default_flow_style=False,
        sort_keys=False,
    )

code2token = pd.Series(
    tokenizer[tokenizer_keys].values,
    index=unique_bnf_codes,
)

read2bnf = pd.read_csv(idir / "read_v2_drugs_bnf.csv")
read2bnf["read_code"] = read2bnf["read_code"].str.replace(" ", "")
read2bnf = read2bnf.set_index("read_code")["bnf_code"]

df = pd.read_csv(
    idir / "gp_scripts.txt",
    sep="\t",
    chunksize=int(1e6),
    encoding="ISO-8859-1",
    converters={
        "eid": int,
        "data_provider": str,
        "issue_date": str,
        "bnf_code": str,
        "read_2": str,
        "dmd_code": str,
        "drug_name": str,
        "quantity": str,
    },
)

mob_txt = Path(multimodal_dir) / "general" / "year_and_month_of_birth.txt"
mob_df = pd.read_csv(mob_txt, sep="\t", index_col="eid")
mob_df["year_month"] = pd.to_datetime(mob_df["year_month"], format="%Y%m")
mob_participants = mob_df.index.astype(int).to_numpy()

ukb_participants = all_ukb_participants()
p2i = pd.DataFrame(
    {
        "pid": ukb_participants,
        "start_pos": 0,
        "seq_len": 0,
    }
)
p2i = p2i.set_index("pid")
all_tokens = []
all_timesteps = []
offset = 0
n_unmapped = 0
hold_out_chunk = None
for chunk in tqdm(df, leave=False):
    is_last_chunk = len(chunk) < int(1e6)

    # keep only those with month of birth data
    has_mob = chunk["eid"].isin(mob_participants)
    chunk = chunk.loc[has_mob].copy()
    chunk["mob"] = mob_df.loc[chunk["eid"], "year_month"].values  # type: ignore

    if hold_out_chunk is not None:
        chunk = pd.concat([hold_out_chunk, chunk], ignore_index=False)
    subs = chunk["eid"].unique()
    if not is_last_chunk:
        hold_out_sub = subs[-1]
        hold_out_chunk = chunk.loc[chunk["eid"] == hold_out_sub].copy()
        chunk = chunk.loc[chunk["eid"] != hold_out_sub].copy()

    # first occurrence only
    chunk = chunk.drop_duplicates(subset=["eid", "bnf_code"], keep="first")

    chunk["issue_date"] = pd.to_datetime(chunk["issue_date"], format="%d/%m/%Y")
    chunk["timesteps"] = (chunk["issue_date"] - chunk["mob"]).dt.days.values.astype(
        np.float32
    )
    have_dates = (chunk["timesteps"].notna()) & (chunk["timesteps"] >= 0)

    bnf_codes = chunk["bnf_code"].copy()
    chunk["read_2"] = chunk["read_2"].str.replace("00", "")
    read_2 = chunk["read_2"]
    read_2_empty = read_2 == ""
    bnf_empty = bnf_codes == ""
    to_map = bnf_empty & ~read_2_empty
    if to_map.sum() > 0:
        mapped_bnf_codes = read2bnf.loc[chunk.loc[to_map, "read_2"]].values
    bnf_codes.loc[to_map] = mapped_bnf_codes
    bnf_codes = bnf_codes.str.replace(".", "")
    bnf_codes = bnf_codes.str[:6]
    have_tokens = bnf_codes.isin(code2token.index)

    are_valid = have_dates & have_tokens
    tokens = code2token.loc[bnf_codes[are_valid]].values.tolist()  # type: ignore
    timesteps = chunk.loc[are_valid, "timesteps"].values.tolist()

    unique_subs = chunk.loc[are_valid, "eid"].unique()
    seq_len = chunk.loc[are_valid, "eid"].value_counts()
    seq_len = seq_len.loc[unique_subs].values
    start_pos = np.cumsum(np.concatenate(([0], seq_len[:-1]))) + offset  # type: ignore
    p2i.loc[unique_subs, "start_pos"] = start_pos
    p2i.loc[unique_subs, "seq_len"] = seq_len
    n_unmapped += (~are_valid).sum()
    offset += len(tokens)
    all_tokens.extend(tokens)
    all_timesteps.extend(timesteps)

all_tokens = np.array(all_tokens, dtype=np.int64)
all_timesteps = np.array(all_timesteps, dtype=np.float32)

os.makedirs(odir, exist_ok=True)
np.save(odir / "data.npy", all_tokens)
np.save(odir / "time.npy", all_timesteps)
p2i.to_csv(odir / "p2i.csv", index_label="pid")
