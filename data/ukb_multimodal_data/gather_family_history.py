import io
import os
import numpy as np
import pandas as pd
import lmdb

father_illness_txt = "/hps/nobackup/birney/users/sfan/delphi-data/multimodal/family_history/father_illness.txt"
mother_illness_txt = "/hps/nobackup/birney/users/sfan/delphi-data/multimodal/family_history/mother_illness.txt"
sibling_illness_txt = "/hps/nobackup/birney/users/sfan/delphi-data/multimodal/family_history/sibling_illness.txt"

father_df = pd.read_csv(father_illness_txt, delimiter="\t", index_col="f.eid")
mother_df = pd.read_csv(mother_illness_txt, delimiter="\t", index_col="f.eid")
sibling_df = pd.read_csv(sibling_illness_txt, delimiter="\t", index_col="f.eid")

print(father_df.shape)
print(mother_df.shape)
print(sibling_df.shape)

father_illness_P = father_df.index.values
mother_illness_P = mother_df.index.values
sibling_illness_P = sibling_df.index.values
assert (father_illness_P == mother_illness_P).all()

family_hx_P = father_illness_P

data_dir = "/hps/nobackup/birney/users/sfan/delphi-data/ukb_real_data"
train_PTX = np.memmap(
    os.path.join(data_dir, "train.bin"), dtype=np.uint32, mode="r"
).reshape(-1, 3)
train_P = np.unique(train_PTX[:, 0])
print(f"train participants: {train_P.shape}")
val_PTX = np.memmap(
    os.path.join(data_dir, "val.bin"), dtype=np.uint32, mode="r"
).reshape(-1, 3)
val_P = np.unique(val_PTX[:, 0])
print(f"val participants: {val_P.shape}")

all_P = np.concatenate([train_P, val_P])
print((~np.isin(family_hx_P, all_P)).sum())
print(f"train and validation participants without family history: {(~np.isin(all_P, family_hx_P)).sum()}")

fhx = father_df.values
mhx = mother_df.values
shx = sibling_df.values


estimated_size = 0
num_participants = len(family_hx_P)

for hx in [fhx, mhx, shx]:
    # For each participant, estimate key + value size
    for i in range(num_participants):
        # Key size: participant ID as string
        participant_id = family_hx_P[i]
        key_size = len(str(participant_id).encode("utf-8"))
        # Value size: serialize the row data
        test_buffer = io.BytesIO()
        row_data = hx[i]
        # Remove NaN values from the row
        clean_row = (
            row_data[~np.isnan(row_data)] if np.any(np.isnan(row_data)) else row_data
        )
        np.save(test_buffer, clean_row)
        value_size = len(test_buffer.getvalue())

        value_size *= 2

        # Add overhead for LMDB entry (key + value + metadata)
        estimated_size += key_size + value_size + 32  # 32 bytes overhead per entry

map_size = int(estimated_size * 1.5)  # 50% safety margin for LMDB

print(f"Estimated database size: {estimated_size / (1024**3):.2f} GB")
print(f"Using map_size: {map_size / (1024**3):.2f} GB")

db_path = "/hps/nobackup/birney/users/sfan/delphi-data/ukb_real_data/family_hx.db"

if os.path.exists(db_path):
    import shutil

    shutil.rmtree(db_path)
    print(f"Deleted LMDB directory: {db_path}")

env = lmdb.open(db_path, map_size=map_size)


with env.begin(write=True) as txn:
    for i in range(father_illness_P.shape[0]):
        participant_id = str(int(father_illness_P[i])).encode("utf-8")
        f = fhx[i, :]
        m = mhx[i, :]
        s = shx[i, :]

        f = f[~np.isnan(f)]
        m = m[~np.isnan(m)]
        s = s[~np.isnan(s)]

        H = np.concatenate([f, m, s])
        F = np.concatenate(
            [np.zeros_like(f), np.zeros_like(m) + 1, np.zeros_like(s) + 2]
        )
        data = np.stack([H, F], axis=0)

        buffer = io.BytesIO()
        np.save(buffer, data)
        value_bytes = buffer.getvalue()

        txn.put(participant_id, value_bytes)

print(participant_id)
print(data)


env.close()
