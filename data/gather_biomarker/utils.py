import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml

from delphi.env import DELPHI_DATA_DIR

multimodal_dir = Path(DELPHI_DATA_DIR) / "multimodal"
tab_dir = multimodal_dir / "tab"
ukb_tab = {}
for file in tab_dir.rglob("*"):
    if file.is_file():
        key = file.stem
        key = key.split("_")[-1]
        ukb_tab[key] = str(file)
ukb_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data"
expansion_pack_dir = ukb_dir / "expansion_packs"
biomarker_dir = ukb_dir / "biomarkers"
token_list_dir = Path("config/disease_list")


def all_ukb_participants():

    participant_dir = ukb_dir / "participants"
    participants = np.fromfile(participant_dir / "all.bin", dtype=np.uint32)

    return participants


def load_fid(fid: str) -> pd.DataFrame:

    return pd.read_csv(ukb_tab[fid], delimiter="\t", index_col="f.eid")


def load_visit(fid: str, visit_idx: int = 0) -> dict:
    """
    return a dictionary that maps participant IDs to a measurement from a given visit specified by visit_idx
    """

    df = pd.read_csv(ukb_tab[fid], delimiter="\t", index_col="f.eid")
    assert visit_idx < df.shape[1], "visit index out of bounds"

    return df.iloc[:, visit_idx].to_dict()


def build_dataframe(fids: list, visit_idx: int = 0) -> pd.DataFrame:

    df = {}
    for fid in fids:
        df[str(fid)] = load_visit(fid=str(fid), visit_idx=visit_idx)

    return pd.DataFrame(df)


def init_p2i():

    ukb_subjects = all_ukb_participants()

    data_p2i = pd.DataFrame(
        {
            "pid": ukb_subjects,
            "start_pos": np.zeros(len(ukb_subjects), dtype=np.int32),
            "seq_len": np.zeros(len(ukb_subjects), dtype=np.int32),
            "time": np.full(len(ukb_subjects), -1e4, dtype=np.float32),
        },
    )
    data_p2i = data_p2i.set_index("pid")

    return data_p2i


def init_expansion_pack_p2i():

    ukb_subjects = all_ukb_participants()
    p2i = pd.DataFrame(
        {
            "pid": ukb_subjects,
            "start_pos": 0,
            "seq_len": 0,
        }
    )
    p2i = p2i.set_index("pid")

    return p2i


def build_expansion_pack(
    token_np: np.ndarray,
    time_np: np.ndarray,
    count_np: np.ndarray,
    subjects: np.ndarray,
    tokenizer: dict,
    expansion_pack: str,
):
    assert token_np.size == time_np.size
    assert count_np.sum() == token_np.size
    assert subjects.size == count_np.size

    p2i = init_expansion_pack_p2i()
    p2i.loc[subjects, "seq_len"] = count_np
    p2i.loc[subjects, "start_pos"] = np.cumsum(count_np) - count_np
    p2i.loc[p2i["seq_len"] == 0, "start_pos"] = 0

    odir = Path(expansion_pack_dir) / expansion_pack
    os.makedirs(odir, exist_ok=True)
    p2i.to_csv(odir / "p2i.csv")
    np.save(odir / "data.npy", token_np)
    np.save(odir / "time.npy", time_np)

    with open(odir / "tokenizer.yaml", "w") as f:
        yaml.dump(
            tokenizer,
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    with open(token_list_dir / f"{expansion_pack}.yaml", "w") as f:
        yaml.dump(
            list(tokenizer.keys()),
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def build_biomarker(
    biomarker_df: pd.DataFrame,
    time_df: pd.Series,
    biomarker: str,
    data_dtype=np.float32,
):

    nan_free = biomarker_df.notna().all(axis=1)
    print(
        f"# participants with NaN in biomarker data: {len(biomarker_df) - nan_free.sum()}"
    )

    biomarker_subjects = biomarker_df.index.astype(int).to_numpy()
    biomarker_subjects = biomarker_subjects[nan_free]

    time_subjects = time_df.index.astype(int).to_numpy()

    ukb_subjects = all_ukb_participants()

    valid_subjects = list(
        set(biomarker_subjects) & set(time_subjects) & set(ukb_subjects)
    )

    data_np = biomarker_df.loc[valid_subjects].to_numpy().astype(data_dtype)

    data_p2i = init_p2i()
    seq_len = data_np.shape[1]
    data_p2i.loc[valid_subjects, "start_pos"] = (
        np.cumsum(np.full(len(valid_subjects), seq_len)) - seq_len
    ).astype(np.int32)
    data_p2i.loc[valid_subjects, "seq_len"] = seq_len
    data_p2i.loc[valid_subjects, "time"] = (
        time_df.loc[valid_subjects].to_numpy().astype(np.float32)
    )

    odir = biomarker_dir / biomarker
    os.makedirs(odir, exist_ok=True)
    np.save(odir / "data.npy", data_np.ravel())
    data_p2i.to_csv(odir / "p2i.csv", index_label="pid")


def data_key(pid: Union[int, str]) -> str:
    return f"{pid}.data"


def time_key(pid: Union[int, str]) -> str:
    return f"{pid}.time"


assessment_age = pd.read_csv(
    os.path.join(DELPHI_DATA_DIR, "multimodal/general/age_at_assess.csv"),
    index_col="pid",
)["age"]

sex = pd.read_csv(
    os.path.join(DELPHI_DATA_DIR, "multimodal/general/sex_31.txt"),
    delimiter="\t",
    index_col="f.eid",
)["f.31.0.0"].to_dict()
