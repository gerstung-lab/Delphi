import numpy as np
import pandas as pd
import yaml
from utils import (
    all_ukb_participants,
    build_expansion_pack,
    expansion_pack_dir,
    load_fid,
)

odir = expansion_pack_dir / "family_hx"
with open(odir / "tokenizer.yaml", "r") as f:
    tokenizer = yaml.safe_load(f)

with open("data/gather_biomarker/family_hx_coding.yaml", "r") as f:
    map_config = yaml.safe_load(f)
lookup = np.zeros((max(map_config.keys()) + 1,), dtype=np.int32)
for key, value in map_config.items():
    lookup[int(key)] = int(value)


def load_visit(fid: str, visit_idx: int = 0) -> pd.DataFrame:
    df = load_fid(fid=fid)
    in_visit = df.columns.str.contains(f"f.{fid}.{visit_idx}")
    return df.iloc[:, in_visit]


father_hx_df = load_visit("20107", visit_idx=0)
mother_hx_df = load_visit("20110", visit_idx=0)
sibling_hx_df = load_visit("20111", visit_idx=0)

family_hx_participants = father_hx_df.index.astype(int).to_numpy()
ukb_participants = all_ukb_participants()
n_missing = len(set(ukb_participants) - set(family_hx_participants))
print(f"# participants with missing family_hx: {n_missing}")
is_valid = np.isin(family_hx_participants, ukb_participants)
valid_participants = family_hx_participants[is_valid]

all_hx_df = pd.concat([father_hx_df, mother_hx_df, sibling_hx_df], axis=1)
token_np = all_hx_df.values
accept_mask = (token_np > 0) * (~np.isnan(token_np))
count_np = accept_mask[is_valid].sum(axis=1)
token_np = token_np[accept_mask * is_valid.reshape(-1, 1)]
token_np = token_np.astype(int)
token_np = lookup[token_np]
time_np = np.zeros_like(token_np)

build_expansion_pack(
    token_np=token_np,
    time_np=time_np,
    count_np=count_np,
    subjects=valid_participants,
    tokenizer=tokenizer,
    expansion_pack="family_hx",
)
