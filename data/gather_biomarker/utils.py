import os
from curses.ascii import DEL

import numpy as np
import pandas as pd

from delphi.env import DELPHI_DATA_DIR

MULTIMODAL_INPUT_DIR = os.environ["MULTIMODAL_INPUT_DIR"]
MULTIMODAL_OUTPUT_DIR = os.environ["MULTIMODAL_OUTPUT_DIR"]

expansion_pack_dir = os.path.join(DELPHI_DATA_DIR, "ukb_real_data", "expansion_packs")


def all_ukb_participants():

    data_dir = os.path.join(DELPHI_DATA_DIR, "ukb_real_data")
    train_PTX = np.memmap(
        os.path.join(data_dir, "train.bin"), dtype=np.uint32, mode="r"
    ).reshape(-1, 3)
    train_P = train_PTX[:, 0]
    val_PTX = np.memmap(
        os.path.join(data_dir, "val.bin"), dtype=np.uint32, mode="r"
    ).reshape(-1, 3)
    val_P = val_PTX[:, 0]
    all_P = np.concatenate([train_P, val_P])
    all_P = np.unique(all_P)

    return all_P


assessment_age = pd.read_csv(
    "data/multimodal/general/age_at_assess.csv", index_col="pid"
)["age"].to_dict()

sex = pd.read_csv(
    "data/multimodal/general/sex_31.txt", delimiter="\t", index_col="f.eid"
)["f.31.0.0"].to_dict()
