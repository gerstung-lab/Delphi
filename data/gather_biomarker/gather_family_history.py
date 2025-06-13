import os

import numpy as np
import pandas as pd
import yaml
from utils import MULTIMODAL_INPUT_DIR, MULTIMODAL_OUTPUT_DIR, all_ukb_participants

from delphi.data.lmdb import data_key, estimate_write_size, time_key, write_lmdb

data_dtype = np.int64
time_dtype = np.uint8

family_hx_dir = os.path.join(MULTIMODAL_INPUT_DIR, "family_history")
father_hx_txt = os.path.join(family_hx_dir, "father_illness.txt")
mother_hx_txt = os.path.join(family_hx_dir, "mother_illness.txt")
sibling_hx_txt = os.path.join(family_hx_dir, "sibling_illness.txt")

father_hx_df = pd.read_csv(father_hx_txt, delimiter="\t", index_col="f.eid")
mother_hx_df = pd.read_csv(mother_hx_txt, delimiter="\t", index_col="f.eid")
sibling_hx_df = pd.read_csv(sibling_hx_txt, delimiter="\t", index_col="f.eid")

assert (father_hx_df.index == mother_hx_df.index).all()
assert (sibling_hx_df.index == mother_hx_df.index).all()
participants_with_family_hx = father_hx_df.index.astype(int).to_numpy()
participants = all_ukb_participants()
n_missing = len(set(participants) - set(participants_with_family_hx))
print(f"# participants with missing family_hx: {n_missing}")
valid_participants = set(participants) & set(participants_with_family_hx)

with open("data/gather_biomarker/family_hx/map.yaml", "r") as f:
    map_config = yaml.safe_load(f)
lookup_max = max(map_config.keys())
lookup_min = min(map_config.keys())
lookup_size = lookup_max - lookup_min + 1
lookup = np.zeros((lookup_size,), dtype=np.int32)
for key, value in map_config.items():
    lookup[int(key) - lookup_min] = int(value)

hx_db = {}
n_unknown = 0
for pid in participants:
    hx_db[data_key(pid)] = np.array([], dtype=data_dtype)
    hx_db[time_key(pid)] = np.array([], dtype=time_dtype)

for pid in valid_participants:

    f = father_hx_df.loc[pid].to_numpy()
    m = mother_hx_df.loc[pid].to_numpy()
    s = sibling_hx_df.loc[pid].to_numpy()
    H = np.concatenate([f, m, s])
    H = H[~np.isnan(H)].astype(int)
    H = lookup[H - lookup_min]
    H = np.unique(H)
    if H.size == 0 or np.array_equal(H, np.array([1])):
        n_unknown += 1
        continue
    if H.max() > 1:
        H = H[H > 1]
    hx_db[data_key(pid)] = H[np.newaxis, :].astype(data_dtype)
    hx_db[time_key(pid)] = np.array([0]).astype(time_dtype)

print(f"# participants with unknown family_hx: {n_unknown}")

db_path = os.path.join(MULTIMODAL_OUTPUT_DIR, "family_hx.lmdb")
map_size = estimate_write_size(db=hx_db)
write_lmdb(db=hx_db, db_path=db_path, map_size=map_size)
