import io
import os

import lmdb
import numpy as np
import pandas as pd
import pytest

from delphi.data.lmdb import data_key, get_all_pids, time_key
from delphi.env import DELPHI_DATA_DIR

expansion_pack_path = os.path.join(DELPHI_DATA_DIR, "ukb_real_data", "expansion_packs")
biomarker_path = os.path.join(DELPHI_DATA_DIR, "ukb_real_data", "biomarkers")


def lookup_path_exists(expanion_pack_path: str) -> bool:
    lookup_path = os.path.join(expanion_pack_path, "p2i.csv")
    return os.path.exists(lookup_path)


def expansion_pack_has_all_participants(expansion_pack_path: str) -> bool:

    from gather_biomarker.utils import all_ukb_participants

    lookup_path = os.path.join(expansion_pack_path, "p2i.csv")
    lookup_df = pd.read_csv(lookup_path, index_col="pid")
    participants_in_lookup = lookup_df.index.astype(int).to_numpy()
    all_participants = all_ukb_participants()

    return bool(np.isin(all_participants, participants_in_lookup).all())


def all_start_pos_valid(expansion_pack_path: str) -> bool:

    lookup_path = os.path.join(expansion_pack_path, "p2i.csv")
    lookup_df = pd.read_csv(lookup_path, index_col="pid")
    start_pos = lookup_df["start_pos"].to_numpy()

    data_path = os.path.join(expansion_pack_path, "data.bin")
    tokens = np.memmap(data_path, dtype=np.uint32, mode="r")
    n_tokens = tokens.size

    within_range = np.all(start_pos < n_tokens) & np.all(start_pos >= 0)
    non_decreasing = np.all(np.diff(start_pos) >= 0)

    return bool(within_range and non_decreasing)


def total_seq_len_add_up(expansion_pack_path: str) -> int:
    lookup_path = os.path.join(expansion_pack_path, "p2i.csv")
    lookup_df = pd.read_csv(lookup_path, index_col="pid")
    seq_len = lookup_df["seq_len"].to_numpy()

    data_path = os.path.join(expansion_pack_path, "data.bin")
    tokens = np.memmap(data_path, dtype=np.uint32, mode="r")

    return np.sum(seq_len) == tokens.size


def tokenizer_exists(expansion_pack_path: str) -> bool:
    tokenizer_path = os.path.join(expansion_pack_path, "tokenizer.yaml")
    return os.path.exists(tokenizer_path)


def data_memmap_exists(expansion_pack_path: str) -> bool:
    data_path = os.path.join(expansion_pack_path, "data.bin")
    return os.path.exists(data_path)


def tokens_within_range(expansion_pack_path: str) -> bool:
    data_path = os.path.join(expansion_pack_path, "data.bin")
    tokens = np.memmap(data_path, dtype=np.uint32, mode="r")
    min_token = np.min(tokens)
    max_token = np.max(tokens)

    tokenizer_path = os.path.join(expansion_pack_path, "tokenizer.yaml")
    with open(tokenizer_path, "r") as f:
        import yaml

        tokenizer = yaml.safe_load(f)
    max_in_tokenizer = max(tokenizer.values())
    min_in_tokenizer = min(tokenizer.values())

    return min_token >= min_in_tokenizer and max_token <= max_in_tokenizer


def time_memmap_exists(expansion_pack_path: str) -> bool:
    time_path = os.path.join(expansion_pack_path, "time.bin")
    return os.path.exists(time_path)


def data_and_time_size_match(expansion_pack_path: str) -> bool:

    data_path = os.path.join(expansion_pack_path, "data.bin")
    time_path = os.path.join(expansion_pack_path, "time.bin")
    tokens = np.memmap(data_path, dtype=np.uint32, mode="r")
    time_steps = np.memmap(time_path, dtype=np.uint32, mode="r")
    print(f"Data size: {tokens.size}, Time steps size: {time_steps.size}")

    return tokens.size == time_steps.size


@pytest.mark.parametrize(
    "expansion_pack_path",
    [
        os.path.join(DELPHI_DATA_DIR, "ukb_real_data"),
        os.path.join(expansion_pack_path, "fsf"),
    ],
)
def test_expansion_pack(expansion_pack_path):

    assert lookup_path_exists(expansion_pack_path)
    assert tokenizer_exists(expansion_pack_path)
    assert data_memmap_exists(expansion_pack_path)
    assert time_memmap_exists(expansion_pack_path)

    assert data_and_time_size_match(expansion_pack_path)
    assert tokens_within_range(expansion_pack_path)

    assert expansion_pack_has_all_participants(expansion_pack_path)
    assert all_start_pos_valid(expansion_pack_path)
    assert total_seq_len_add_up(expansion_pack_path)


def data_and_time_keys_match(biomarker_path: str) -> bool:

    pids = get_all_pids(biomarker_path)
    env = lmdb.open(biomarker_path, readonly=True, lock=False)
    with env.begin() as txn:
        for pid in pids:
            data_bytes = txn.get(data_key(pid))
            time_bytes = txn.get(time_key(pid))
            if data_bytes is None or time_bytes is None:
                return False

    return True


def biomarkers_have_all_participants(biomarker_path: str) -> bool:

    pids = get_all_pids(biomarker_path)
    env = lmdb.open(biomarker_path, readonly=True, lock=False)
    with env.begin() as txn:
        for pid in pids:
            data_bytes = txn.get(data_key(pid))
            if data_bytes is None:
                return False
    return True


def data_is_2d(biomarker_path: str) -> bool:

    pids = get_all_pids(biomarker_path)
    env = lmdb.open(biomarker_path, readonly=True, lock=False)
    with env.begin() as txn:
        for pid in pids:
            data_bytes = txn.get(data_key(pid))
            biomarker_data = np.load(io.BytesIO(data_bytes))

            if biomarker_data.size == 0:
                continue
            else:
                if biomarker_data.ndim != 2:
                    return False

    return True


def time_is_1d(biomarker_path: str) -> bool:

    pids = get_all_pids(biomarker_path)
    env = lmdb.open(biomarker_path, readonly=True, lock=False)
    with env.begin() as txn:
        for pid in pids:
            time_bytes = txn.get(time_key(pid))
            biomarker_time = np.load(io.BytesIO(time_bytes))

            if biomarker_time.size == 0:
                continue
            else:
                if biomarker_time.ndim != 1:
                    return False

    return True


def data_and_time_dimensions_match(biomarker_path: str) -> bool:

    pids = get_all_pids(biomarker_path)
    env = lmdb.open(biomarker_path, readonly=True, lock=False)
    with env.begin() as txn:
        for pid in pids:
            data_bytes = txn.get(data_key(pid))
            time_bytes = txn.get(time_key(pid))
            biomarker_data = np.load(io.BytesIO(data_bytes))
            biomarker_time = np.load(io.BytesIO(time_bytes))

            if biomarker_data.shape[0] != biomarker_time.size:
                return False

    return True


_, biomarkers, _ = next(os.walk(biomarker_path))


@pytest.mark.parametrize(
    "biomarker_path",
    [os.path.join(biomarker_path, biomarker) for biomarker in biomarkers],
)
def test_biomarkers(biomarker_path):

    assert data_and_time_keys_match(biomarker_path)
    assert biomarkers_have_all_participants(biomarker_path)
    assert data_is_2d(biomarker_path)
    assert time_is_1d(biomarker_path)
    assert data_and_time_dimensions_match(biomarker_path)
