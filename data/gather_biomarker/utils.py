import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from delphi import DAYS_PER_YEAR
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

with open("data/gather_biomarker/panel.yaml", "r") as f:
    biomarkers = yaml.safe_load(f)


def all_ukb_participants():

    participant_dir = ukb_dir / "participants"
    participants = np.fromfile(participant_dir / "all.bin", dtype=np.uint32)

    return participants


def load_fid(fid: str) -> pd.DataFrame:

    return pd.read_csv(ukb_tab[str(fid)], delimiter="\t", index_col="f.eid")


def load_coding(scheme: int) -> pd.DataFrame:

    coding_path = multimodal_dir / "coding" / f"{str(scheme)}.txt"
    if not coding_path.exists():
        raise FileNotFoundError(f"Coding file {coding_path} does not exist.")

    return pd.read_csv(coding_path, sep="\t")


def index_by_visit(df: pd.DataFrame, visits: list[str]) -> pd.Series:

    n = df.shape[0]
    l = df.shape[1]
    assert l == len(
        visits
    ), "Number of visits does not match number of columns in DataFrame"
    vals = np.concatenate([df[col].to_numpy() for col in df.columns], axis=0)
    visit_types = np.repeat(np.array(visits), n)
    subjects = np.tile(df.index.to_numpy(), l)

    return pd.Series(
        data=vals,
        index=pd.MultiIndex.from_arrays(
            [subjects, visit_types], names=["pid", "visit"]
        ),
    )


def load_biomarker_df(fids: list, visits: list[str]) -> pd.DataFrame:

    markers = []
    for fid in fids:
        marker = load_fid(str(fid))
        marker = index_by_visit(df=marker, visits=visits)
        markers.append(marker)
    long_df = pd.concat(markers, axis=1)

    return long_df


def load_visit(fid: str, visit_idx: int = 0) -> dict:
    """
    return a dictionary that maps participant IDs to a measurement from a given visit specified by visit_idx
    """

    df = load_fid(fid=fid)
    assert visit_idx < df.shape[1], "visit index out of bounds"

    return df.iloc[:, visit_idx].to_dict()


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
    print(expansion_pack)
    assert token_np.size == time_np.size
    assert count_np.sum() == token_np.size
    assert subjects.size == count_np.size
    print(f"\t - total tokens: {token_np.size}")
    print(f"\t - subjects: {subjects.size}")
    print(f"\t - avg tokens per subject: {count_np.mean()}")
    print(f"\t - max tokens per subject: {count_np.max()}")
    print(f"\t - vocab size: {len(tokenizer)}")

    p2i = init_expansion_pack_p2i()
    p2i.loc[subjects, "seq_len"] = count_np
    p2i.loc[subjects, "start_pos"] = np.cumsum(count_np) - count_np
    p2i.loc[p2i["seq_len"] == 0, "start_pos"] = 0

    odir = Path(expansion_pack_dir) / expansion_pack
    os.makedirs(odir, exist_ok=True)
    p2i.to_csv(odir / "p2i.csv")
    token_np.astype(np.uint32).tofile(odir / "data.bin")
    time_np = time_np.astype(np.uint32)
    print(
        f"\t - time points from {time_np.min() / DAYS_PER_YEAR} to {time_np.max() / DAYS_PER_YEAR}"
    )
    time_np.tofile(odir / "time.bin")

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
    time_series: pd.Series,
    biomarker: str,
    data_dtype=np.float32,
):

    print(biomarker)
    subjects = biomarker_df.reset_index()["pid"].to_numpy().astype(np.int32)
    visits = biomarker_df.reset_index()["visit"].to_numpy().astype(str)

    ukb_subjects = all_ukb_participants()
    not_in_ukb_subjects = ~np.isin(subjects, ukb_subjects)
    print(f"\t - not found in Delphi cohort: {not_in_ukb_subjects.sum()}")
    is_valid = ~not_in_ukb_subjects

    time_np = time_series[biomarker_df.index].to_numpy().astype(np.float32)
    has_nan_time = np.isnan(time_np)
    print(f"\t - has NaN in time: {has_nan_time.sum()}")
    is_valid *= ~has_nan_time

    data_np = biomarker_df.to_numpy().astype(data_dtype)
    has_nan_data = biomarker_df.isna().any(axis=1)
    print(f"\t - has NaN in data: {has_nan_data.sum()}")
    is_valid *= ~has_nan_data

    print(f"\t - total remaining: {is_valid.sum()}")
    histogram = (
        biomarker_df.loc[is_valid]
        .reset_index()["pid"]
        .value_counts()
        .value_counts()
        .to_dict()
    )
    print(f"\t - histogram: {histogram}")

    data_np = data_np[is_valid]
    time_np = time_np[is_valid]
    subjects = subjects[is_valid]
    visits = visits[is_valid]

    seq_len = data_np.shape[1]
    p2i = pd.DataFrame.from_dict(
        data={
            "pid": subjects,
            "visit": visits,
            "start_pos": (np.cumsum(np.full(is_valid.sum(), seq_len)) - seq_len).astype(
                np.int32
            ),
            "seq_len": seq_len,
            "time": time_np,
        }
    )

    miss_subjects = list(set(ukb_subjects) - set(subjects))
    miss_p2i = pd.DataFrame.from_dict(
        data={
            "pid": miss_subjects,
            "visit": "none",
            "start_pos": 0,
            "seq_len": 0,
            "time": -1e4,
        }
    )

    p2i = pd.concat([p2i, miss_p2i], axis=0)

    odir = biomarker_dir / biomarker
    os.makedirs(odir, exist_ok=True)
    data_np.ravel().astype(np.float32).tofile(odir / "data.bin")
    p2i.to_csv(odir / "p2i.csv", index=False)


def month_of_birth() -> pd.DataFrame:

    mob = pd.read_csv(
        multimodal_dir / "year_and_month_of_birth.txt", sep="\t", index_col="eid"
    )
    mob["year_month"] = pd.to_datetime(mob["year_month"], format="%Y%m")

    return mob


def assessment_age(visits: list):

    mob = month_of_birth()

    assess_date = load_fid(fid="53")
    assess_date = assess_date.rename(
        columns={
            "f.53.0.0": "init_assess",
            "f.53.1.0": "1st_repeat_assess",
            "f.53.2.0": "img",
            "f.53.3.0": "1st_repeat_img",
        }
    )

    assert set(visits).issubset(set(assess_date.columns))
    assess_age = pd.DataFrame(columns=visits, index=assess_date.index)
    for col in visits:
        assess_date[col] = pd.to_datetime(assess_date[col], format="%Y-%m-%d")
        assess_age[col] = (assess_date[col] - mob["year_month"]).dt.days.astype(float)

    return assess_age
