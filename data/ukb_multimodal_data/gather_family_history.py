import io
import os
import numpy as np
import pandas as pd
import lmdb

family_hx_dir = "/hps/nobackup/birney/users/sfan/delphi-data/multimodal/family_history"
father_hx_txt = os.path.join(family_hx_dir, "father_illness.txt")
mother_hx_txt = os.path.join(family_hx_dir, "mother_illness.txt")
sibling_hx_txt = os.path.join(family_hx_dir, "sibling_illness.txt")

father_hx_df = pd.read_csv(father_hx_txt, delimiter="\t", index_col="f.eid")
mother_hx_df = pd.read_csv(mother_hx_txt, delimiter="\t", index_col="f.eid")
sibling_hx_df = pd.read_csv(sibling_hx_txt, delimiter="\t", index_col="f.eid")

print(father_hx_df.shape)
print(mother_hx_df.shape)
print(sibling_hx_df.shape)

father_hx_P = father_hx_df.index.values
mother_hx_P = mother_hx_df.index.values
sibling_hx_P = sibling_hx_df.index.values
assert (father_hx_P == mother_hx_P).all()
assert (sibling_hx_P == mother_hx_P).all()
family_hx_P = father_hx_P

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
print(
    f"train and validation participants without family history: {(~np.isin(all_P, family_hx_P)).sum()}"
)

fhx = father_hx_df.values
mhx = mother_hx_df.values
shx = sibling_hx_df.values


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

        clean_hx = hx[i][~np.isnan(hx[i])]
        np.save(test_buffer, clean_hx)
        value_size = len(test_buffer.getvalue()) * 2

        # Add overhead for LMDB entry (key + value + metadata)
        estimated_size += key_size + value_size + 32  # 32 bytes overhead per entry

map_size = int(estimated_size * 1.5)  # 50% safety margin for LMDB

print(f"Estimated database size: {estimated_size / (1024**3):.2f} GB")
print(f"Using map_size: {map_size / (1024**3):.2f} GB")

db_path = "/hps/nobackup/birney/users/sfan/delphi-data/ukb_real_data/family_hx.lmdb"

if os.path.exists(db_path):
    import shutil

    shutil.rmtree(db_path)
    print(f"Deleted LMDB directory: {db_path}")

env = lmdb.open(db_path, map_size=map_size)

allnan_n = 0
with env.begin(write=True) as txn:
    for i in range(father_hx_P.shape[0]):
        participant_id = str(int(father_hx_P[i])).encode("utf-8")
        f = fhx[i, :]
        m = mhx[i, :]
        s = shx[i, :]

        f = f[~np.isnan(f)]
        m = m[~np.isnan(m)]
        s = s[~np.isnan(s)]

        H = np.concatenate([f, m, s])
        if len(H) == 0:
            allnan_n += 1
            continue
        F = np.concatenate(
            [np.zeros_like(f), np.zeros_like(m) + 1, np.zeros_like(s) + 2]
        )
        data = np.stack([H, F], axis=0)

        buffer = io.BytesIO()
        np.save(buffer, data)
        value_bytes = buffer.getvalue()

        txn.put(participant_id, value_bytes)

print(f"participants with all NaNs in family_hx: {allnan_n}")
print(participant_id)
print(data)


env.close()
