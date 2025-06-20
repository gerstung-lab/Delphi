import os

import numpy as np
import pandas as pd
import pytest
import yaml

from data.gather_biomarker.utils import all_ukb_participants
from delphi.env import DELPHI_DATA_DIR
from delphi.multimodal import Modality

biomarker_path = os.path.join(DELPHI_DATA_DIR, "ukb_real_data", "biomarkers")
all_biomarkers = [modality.lower() for modality in Modality.__members__]
discrete_biomarkers = [Modality.FAMILY_HX.name.lower(), Modality.MEDS.name.lower()]


def biomarkers_have_all_participants(p2i: pd.DataFrame, pids: np.ndarray) -> bool:

    return bool(np.isin(pids, p2i.index.astype(int).to_numpy()).all())


def data_is_1d(data: np.ndarray) -> bool:
    return data.ndim == 1


def no_nan_data(data: np.ndarray) -> bool:
    return not np.isnan(data).any()


def total_dimensions_match(p2i: pd.DataFrame, data: np.ndarray) -> bool:

    total_seq_len = p2i["seq_len"].sum()
    return data.size == total_seq_len


def no_duplicate_start_pos(p2i: pd.DataFrame) -> bool:

    start_pos = p2i["start_pos"].to_numpy()
    nonzero_start_pos = start_pos[start_pos != 0]
    is_unique = len(nonzero_start_pos) == len(set(nonzero_start_pos))

    return is_unique


def placeholder_time_where_no_data(
    p2i: pd.DataFrame,
) -> bool:

    no_data_where_time_is_placeholder = (
        p2i.loc[p2i["time"] == -1e4, "seq_len"] == 0
    ).all()

    placeholder_time_where_no_data = (
        p2i.loc[p2i["seq_len"] == 0, "time"] == -1e4
    ).all()

    return no_data_where_time_is_placeholder and placeholder_time_where_no_data


def real_time_where_data_exists(
    p2i: pd.DataFrame,
) -> bool:

    return (p2i.loc[p2i["seq_len"] > 0, "time"] >= 0).all()


@pytest.mark.parametrize(
    "biomarker_path",
    [os.path.join(biomarker_path, biomarker) for biomarker in all_biomarkers],
)
def test_biomarkers(biomarker_path):

    data = np.load(
        os.path.join(biomarker_path, "data.npy"), allow_pickle=True, mmap_mode="r"
    )
    p2i = pd.read_csv(
        os.path.join(biomarker_path, "p2i.csv"),
        index_col="pid",
    )
    pids = all_ukb_participants()

    assert biomarkers_have_all_participants(p2i=p2i, pids=pids)
    assert data_is_1d(data=data)
    assert no_nan_data(data=data)
    assert total_dimensions_match(p2i=p2i, data=data)
    assert no_duplicate_start_pos(p2i=p2i)
    assert placeholder_time_where_no_data(p2i=p2i)
    assert real_time_where_data_exists(p2i=p2i)


def all_tokens_valid(tokenizer: dict, data: np.ndarray) -> bool:

    valid_tokens = set(tokenizer.values())
    max_token = max(valid_tokens)
    min_token = min(valid_tokens)
    if min_token <= 0:
        return False

    return (np.max(data) <= max_token) and (np.min(data) >= min_token)


@pytest.mark.parametrize(
    "biomarker_path",
    [os.path.join(biomarker_path, biomarker) for biomarker in discrete_biomarkers],
)
def test_discrete_biomarkers(biomarker_path):

    with open(os.path.join(biomarker_path, "tokenizer.yaml"), "rb") as f:
        tokenizer = yaml.safe_load(f)

    data = np.load(
        os.path.join(biomarker_path, "data.npy"), allow_pickle=True, mmap_mode="r"
    )

    assert all_tokens_valid(tokenizer=tokenizer, data=data)
