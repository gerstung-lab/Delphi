import io
import os

import lmdb
import numpy as np
import pandas as pd

from delphi.data.lmdb import get_all_pids

blood_txt = os.path.join(
    os.environ["MULTIMODAL_INPUT_DIR"],
    "blood/biomarkers_biochem_bloods_imputed_clean_zscore.txt",
)
print(f"reading blood mice biomarkers from: {blood_txt}")
data_dir = os.path.join(os.environ["DELPHI_DATA_DIR"], "ukb_real_data")
print(f"loading train and val participants from: {data_dir}")
assess_date_db_path = os.path.join(
    os.environ["MULTIMODAL_OUTPUT_DIR"], "assess-date.lmdb"
)
print(f"loading assessment date from: {assess_date_db_path}")
blood_db_path = os.path.join(os.environ["MULTIMODAL_OUTPUT_DIR"], "blood-mice.lmdb")
print(f"saving LMDB database to: {blood_db_path}")

blood_df = pd.read_csv(blood_txt, sep="\t", index_col="eid")
blood_df = blood_df.drop(blood_df.columns[0], axis=1)  # drop the first column (eid)
print(f"blood mice biomarkers: {blood_df.shape[1]}")
print(blood_df.columns)
blood_P = blood_df.index.values
assert blood_df.index.is_unique
print(f"blood mice participants: {len(blood_P)}")
blood_X = blood_df.values

assess_date_P = get_all_pids(db_path=assess_date_db_path)
assert np.isin(
    blood_P, assess_date_P
).all(), "some blood mice participants not in assess date database"

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
print(f"participants without blood mice: {(~np.isin(all_P, blood_P)).sum()}")

# estimate map_size
test_buffer = io.BytesIO()
np.save(test_buffer, blood_X[0, :])
array_size = len(test_buffer.getvalue())
record_size = len(str(int(blood_P[0])).encode("utf-8")) + array_size + 16
estimated_size = record_size * blood_P.shape[0]
map_size = int(estimated_size * 1.5)  # 50% safety margin
print(f"estimated database size: {estimated_size / (1024**3):.2f} GB")
print(f"using map_size: {map_size / (1024**3):.2f} GB")


if os.path.exists(blood_db_path):
    print(f"found existing LMDB directory: {blood_db_path}")
    import shutil

    shutil.rmtree(blood_db_path)
    print(f"deleted existing LMDB directory: {blood_db_path}")

env = lmdb.open(blood_db_path, map_size=map_size)

with env.begin(write=True) as txn:
    for i in range(blood_P.shape[0]):
        participant_id = str(int(blood_P[i])).encode(
            "utf-8"
        )  # Convert the first column to the key (as a string)
        blood = blood_X[i, :]  # Remaining columns are the values

        # Convert values to bytes using pickle or np.save
        buffer = io.BytesIO()
        np.save(buffer, blood)
        value_bytes = buffer.getvalue()

        txn.put(participant_id, value_bytes)

print(participant_id)
print(blood)

env.close()
