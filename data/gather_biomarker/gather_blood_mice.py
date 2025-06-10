import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from utils import (
    MULTIMODAL_INPUT_DIR,
    MULTIMODAL_OUTPUT_DIR,
    all_ukb_participants,
    assessment_age,
)

from delphi.data.lmdb import data_key, estimate_write_size, time_key, write_lmdb

blood_panel_dir = "data/gather_biomarker/blood/panel"
blood_txt = os.path.join(
    MULTIMODAL_INPUT_DIR,
    "blood",
    "biomarkers_biochem_bloods_imputed_clean_zscore.txt",
)

blood_df = pd.read_csv(blood_txt, sep="\t", index_col="eid")
assert blood_df.index.is_unique
blood_df = blood_df.drop(blood_df.columns[0], axis=1)  # drop the first column (eid)
blood_df.columns = blood_df.columns.str.replace(".txt", "")

all_biomarkers = list(blood_df.columns)
with open(os.path.join(blood_panel_dir, "blood_all.yaml"), "w") as f:
    yaml.dump(
        all_biomarkers,
        f,
        default_flow_style=False,
        sort_keys=False,
    )
all_biomarkers = np.array(all_biomarkers)

participants = all_ukb_participants()
participants_with_blood = blood_df.index.astype(int).to_numpy()
participants_with_assessment_age = list(assessment_age.keys())

blood_panel_yamls = Path(blood_panel_dir).glob("*.yaml")
for blood_panel_yaml in blood_panel_yamls:

    blood_panel = blood_panel_yaml.stem

    with open(blood_panel_yaml, "r") as file:
        panel_biomarkers = yaml.safe_load(file)

    blood_panel_db_path = os.path.join(MULTIMODAL_OUTPUT_DIR, f"{blood_panel}.lmdb")
    print(f"saving lmdb database to: {blood_panel_db_path}")

    blood_panel_db = {}
    for pid in participants:
        blood_panel_db[data_key(pid)] = np.array([], dtype=np.float32)
        blood_panel_db[time_key(pid)] = np.array([], dtype=np.uint32)

    n_blood_missing = len(set(participants) - set(participants_with_blood))
    n_age_missing = len(set(participants) - set(participants_with_assessment_age))
    valid_participants = (
        set(participants)
        & set(participants_with_blood)
        & set(participants_with_assessment_age)
    )

    for pid in valid_participants:
        blood_val = blood_df.loc[pid, list(panel_biomarkers)].to_numpy()
        pid_data = blood_val[np.newaxis, :].astype(np.float32)
        blood_time = assessment_age[pid]
        pid_time = np.array([blood_time], dtype=np.uint32)
        blood_panel_db[data_key(pid)] = pid_data
        blood_panel_db[time_key(pid)] = pid_time

    print(f"number of participants with missing blood data: {n_blood_missing}")
    print(f"number of participants with missing age data: {n_age_missing}")

    map_size = estimate_write_size(db=blood_panel_db, n_samples=10)

    write_lmdb(
        db=blood_panel_db,
        db_path=blood_panel_db_path,
        map_size=map_size,
    )
