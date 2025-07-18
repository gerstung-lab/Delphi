import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from utils import (
    all_ukb_participants,
    assessment_age,
    biomarker_dir,
    multimodal_dir,
)

blood_panel_dir = "data/gather_biomarker/blood/panel"
blood_txt = os.path.join(
    multimodal_dir,
    "blood",
    "biomarkers_biochem_bloods_imputed_clean_zscore.txt",
)

blood_df = pd.read_csv(blood_txt, sep="\t", index_col="eid")
assert blood_df.index.is_unique
blood_df = blood_df.drop(blood_df.columns[0], axis=1)
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

ukb_participants = all_ukb_participants()
participants_with_data = blood_df.index.astype(int).to_numpy()
participants_with_assessment_age = list(assessment_age.keys())

blood_panel_yamls = Path(blood_panel_dir).glob("*.yaml")
for blood_panel_yaml in blood_panel_yamls:

    blood_panel = blood_panel_yaml.stem
    with open(blood_panel_yaml, "r") as file:
        panel_biomarkers = yaml.safe_load(file)

    participants_without_data = list(
        set(ukb_participants) - set(participants_with_data)
    )
    participants_without_age = list(
        set(ukb_participants) - set(participants_with_assessment_age)
    )
    valid_participants = list(
        set(ukb_participants)
        & set(participants_with_data)
        & set(participants_with_assessment_age)
    )
    print(
        f"# participants with missing {blood_panel} data: {len(participants_without_data)}"
    )
    print(f"# participants with missing age data: {len(participants_without_age)}")

    data_np = (
        blood_df.loc[valid_participants, list(panel_biomarkers)]
        .to_numpy()
        .astype(np.float32)
    )
    time_np = assessment_age.loc[valid_participants].to_numpy()

    data_p2i = pd.DataFrame(
        {
            "pid": ukb_participants,
            "start_pos": np.zeros(len(ukb_participants), dtype=np.int32),
            "seq_len": np.zeros(len(ukb_participants), dtype=np.int32),
            "time": np.full(len(ukb_participants), -1e4, dtype=np.float32),
        },
    )
    data_p2i = data_p2i.set_index("pid")
    seq_len = data_np.shape[1]
    data_p2i.loc[valid_participants, "start_pos"] = (
        np.cumsum(np.full(len(valid_participants), seq_len)) - seq_len
    ).astype(np.int32)
    data_p2i.loc[valid_participants, "seq_len"] = seq_len
    data_p2i.loc[valid_participants, "time"] = time_np

    odir = os.path.join(biomarker_dir, blood_panel)
    os.makedirs(odir, exist_ok=True)
    np.save(
        os.path.join(odir, "data.npy"),
        data_np.ravel(),
    )
    data_p2i.index = data_p2i.index.astype(str)
    data_p2i.to_csv(
        os.path.join(odir, "p2i.csv"),
        index_label="pid",
    )
