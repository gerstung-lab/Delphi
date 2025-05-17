import io
import os
import numpy as np
import pandas as pd
import lmdb

prs_txt = "/nfs/research/birney/controlled_access/ukb-cnv/data_fetch/baskets/2017133/PRS_standard_37.txt"

prs_df = pd.read_csv(prs_txt, delimiter="\t", index_col="eid")

print(prs_df.columns)

prs_P = prs_df.index.values
print(f"prs participants: {prs_P.shape}")

prs_X = prs_df.values

data_dir = "/hps/nobackup/birney/users/sfan/delphi-data/ukb_real_data"
train_PTX = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint32, mode='r').reshape(-1, 3)
train_P = np.unique(train_PTX[:, 0])
print(f"train participants: {train_P.shape}")
val_PTX = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint32, mode='r').reshape(-1, 3)
val_P = np.unique(val_PTX[:, 0])
print(f"val participants: {val_P.shape}")

all_P = np.concatenate([train_P, val_P])
print((~np.isin(prs_P, all_P)).sum())
print(f"train + val participants without PRS scores: {(~np.isin(all_P, prs_P)).sum()}")

has_nan = np.isnan(prs_X).sum(axis=1) > 0
print(f"prs participants with NaNs in their PRS scores: {has_nan.sum()}")

prs_P = prs_P[~has_nan]
prs_X = prs_X[~has_nan]
print(f"prs participants after NaN removal: {prs_P.shape}")

# estimate map_size
test_buffer = io.BytesIO()
np.save(test_buffer, prs_X[0, :])
array_size = len(test_buffer.getvalue())
record_size = len(str(int(prs_P[0])).encode('utf-8')) + array_size + 16
estimated_size = record_size * prs_P.shape[0]
map_size = int(estimated_size * 1.5)  # 50% safety margin

print(f"Estimated database size: {estimated_size / (1024**3):.2f} GB")
print(f"Using map_size: {map_size / (1024**3):.2f} GB")

db_path = "/hps/nobackup/birney/users/sfan/delphi-data/ukb_real_data/prs.lmdb"

if os.path.exists(db_path):
	import shutil
	shutil.rmtree(db_path)
	print(f"Deleted LMDB directory: {db_path}")

env = lmdb.open(db_path, map_size=map_size)

with env.begin(write=True) as txn:
	for i in range(prs_P.shape[0]):
	    participant_id = str(int(prs_P[i])).encode('utf-8')  # Convert the first column to the key (as a string)
	    prs_scores = prs_X[i, :]  # Remaining columns are the values

	    # Convert values to bytes using pickle or np.save
	    buffer = io.BytesIO()
	    np.save(buffer, prs_scores)
	    value_bytes = buffer.getvalue()

	    txn.put(participant_id, value_bytes) 

print(participant_id)
print(prs_scores)

env.close()


