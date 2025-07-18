import os

import numpy as np
import yaml
from tqdm import tqdm
from utils import (
    all_ukb_participants,
    biomarker_dir,
    init_p2i,
    load_fid,
)

with open("data/ukb_real_data/biomarkers/family_hx/tokenizer.yaml", "r") as f:
    map_config = yaml.safe_load(f)
lookup_max = max(map_config.keys())
lookup_min = min(map_config.keys())
lookup_size = lookup_max - lookup_min + 1
lookup = np.zeros((lookup_size,), dtype=np.int32)
for key, value in map_config.items():
    lookup[int(key) - lookup_min] = int(value)

father_hx_df = load_fid("20107")
mother_hx_df = load_fid("20110")
sibling_hx_df = load_fid("20111")

participants_with_family_hx = father_hx_df.index.astype(int).to_numpy()
ukb_participants = all_ukb_participants()
n_missing = len(set(ukb_participants) - set(participants_with_family_hx))
valid_participants = list(set(ukb_participants) & set(participants_with_family_hx))
print(f"# participants with missing family_hx: {n_missing}")


data_p2i = init_p2i()
data_lst = []
n_unknown = 0
start_pos = []
seq_len = []
offset = 0
truly_valid_participants = []
for pid in tqdm(valid_participants):

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

    truly_valid_participants.append(pid)
    data_lst.extend(H.tolist())
    start_pos.append(offset)
    seq_len.append(H.size)
    offset += H.size

print(f"# participants with unknown family_hx: {n_unknown}")
data_p2i.loc[truly_valid_participants, "start_pos"] = np.array(
    start_pos, dtype=np.int32
)
data_p2i.loc[truly_valid_participants, "seq_len"] = np.array(seq_len, dtype=np.int32)
data_p2i.loc[truly_valid_participants, "time"] = 0.0
data_np = np.array(data_lst, dtype=np.int64)

odir = os.path.join(biomarker_dir, "family_hx")
os.makedirs(odir, exist_ok=True)
np.save(os.path.join(odir, "data.npy"), data_np)
data_p2i.to_csv(
    os.path.join(odir, "p2i.csv"),
    index_label="pid",
)
