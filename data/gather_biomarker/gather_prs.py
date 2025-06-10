import os

import numpy as np
import pandas as pd
from utils import MULTIMODAL_INPUT_DIR, MULTIMODAL_OUTPUT_DIR, all_ukb_participants

from delphi.data.lmdb import data_key, estimate_write_size, time_key, write_lmdb

prs_txt = os.path.join(MULTIMODAL_INPUT_DIR, "PRS", "PRS_standard_37.txt")
prs_df = pd.read_csv(prs_txt, delimiter="\t", index_col="eid")
participants_with_prs = prs_df.index.values
has_nan = prs_df.isna().any(axis=1)
print(f"# participants with NaNs in PRS: {has_nan.sum()}")
participants_with_prs = participants_with_prs[~has_nan]

participants = all_ukb_participants()
n_prs_missing = len(set(participants) - set(participants_with_prs))
print(f"# participants without PRS or NaNs in PRS: {n_prs_missing}")
valid_participants = set(participants) & set(participants_with_prs)

prs_db = {}
for pid in participants:
    prs_db[data_key(pid)] = np.array([], dtype=np.float32)
    prs_db[time_key(pid)] = np.array([], dtype=np.uint8)
for pid in valid_participants:
    prs_scores = prs_df.loc[pid].to_numpy(dtype=np.float32)
    prs_db[data_key(pid)] = prs_scores[np.newaxis, :]
    prs_db[time_key(pid)] = np.array([0], dtype=np.uint8)

map_size = estimate_write_size(db=prs_db, n_samples=10)
db_path = os.path.join(MULTIMODAL_OUTPUT_DIR, "prs.lmdb")
write_lmdb(db=prs_db, db_path=db_path, map_size=map_size)
