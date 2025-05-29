import io
import os

import lmdb
import numpy as np
import pandas as pd

from delphi.data.lmdb import estimate_size_by_first_record

mob_txt = os.path.join(
    os.environ["MULTIMODAL_INPUT_DIR"],
    "blood/year_and_month_of_birth.txt",
)
print(f"reading month of birth from: {mob_txt}")
assess_date_txt = os.path.join(
    os.environ["MULTIMODAL_INPUT_DIR"],
    "blood/Date_of_attending_assessment_centre_53.txt",
)
print(f"reading assessment date from: {assess_date_txt}")
data_dir = os.path.join(os.environ["DELPHI_DATA_DIR"], "ukb_real_data")
print(f"loading train and val participants from: {data_dir}")
assess_date_db_path = os.path.join(
    os.environ["MULTIMODAL_OUTPUT_DIR"], "assess-date.lmdb"
)
print(f"saving LMDB database to: {assess_date_db_path}")

mob_df = pd.read_csv(mob_txt, sep="\t", index_col="eid")
mob_df["year_month"] = pd.to_datetime(mob_df["year_month"], format="%Y%m")
assess_date_df = pd.read_csv(assess_date_txt, sep="\t", index_col="f.eid")
assess_date_df["first_visit_date"] = pd.to_datetime(
    assess_date_df["f.53.0.0"], format="%Y-%m-%d"
)
assert (mob_df.index == assess_date_df.index).all()
assess_P = assess_date_df.index.values

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
print(f"participants without assess date: {(~np.isin(all_P, assess_P)).sum()}")

assess_age_in_days = (assess_date_df["first_visit_date"] - mob_df["year_month"]).dt.days

map_size = estimate_size_by_first_record(
    keys=assess_P, values=np.array(assess_age_in_days.values)
)

if os.path.exists(assess_date_db_path):
    print(f"found existing LMDB directory: {assess_date_db_path}")
    import shutil

    shutil.rmtree(assess_date_db_path)
    print(f"deleted existing LMDB directory: {assess_date_db_path}")

env = lmdb.open(assess_date_db_path, map_size=map_size)

with env.begin(write=True) as txn:
    for i in range(assess_P.shape[0]):
        participant_id = str(int(assess_P[i])).encode(
            "utf-8"
        )  # Convert the first column to the key (as a string)

        age = assess_age_in_days[assess_P[i]]
        buffer = io.BytesIO()
        np.save(buffer, age)
        value_bytes = buffer.getvalue()

        txn.put(participant_id, value_bytes)

print(participant_id)
print(age)

env.close()
