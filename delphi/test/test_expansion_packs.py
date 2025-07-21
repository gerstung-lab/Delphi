import os

import numpy as np
import pandas as pd

from delphi.test.test_biomarkers import has_all_participants


def required_files_exist(expansion_pack_path: str):

    tokenizer_path = os.path.join(expansion_pack_path, "tokenizer.yaml")
    data_path = os.path.join(expansion_pack_path, "data.bin")
    time_path = os.path.join(expansion_pack_path, "time.bin")
    lookup_path = os.path.join(expansion_pack_path, "p2i.csv")

    return (
        os.path.exists(tokenizer_path)
        and os.path.exists(data_path)
        and os.path.exists(time_path)
        and os.path.exists(lookup_path)
    )


def all_start_pos_within_range(p2i: pd.DataFrame, tokens: np.ndarray) -> bool:

    start_pos = p2i["start_pos"].to_numpy()
    n_tokens = tokens.size
    within_range = np.all(start_pos < n_tokens) & np.all(start_pos >= 0)

    return bool(within_range)


def no_duplicate_start_pos(p2i: pd.DataFrame) -> bool:

    start_pos = p2i["start_pos"].to_numpy()
    nonzero_start_pos = start_pos[start_pos != 0]
    is_unique = len(nonzero_start_pos) == len(set(nonzero_start_pos))

    return is_unique


def total_seq_len_add_up(p2i: pd.DataFrame, tokens: np.ndarray) -> int:

    seq_len = p2i["seq_len"].to_numpy()

    return np.sum(seq_len) == tokens.size


def tokens_within_range(tokens: np.ndarray, tokenizer: dict) -> bool:
    min_token = np.min(tokens)
    max_token = np.max(tokens)

    max_in_tokenizer = max(tokenizer.values())
    min_in_tokenizer = min(tokenizer.values())

    return min_token >= min_in_tokenizer and max_token <= max_in_tokenizer


def no_nan_tokens(tokens: np.ndarray) -> bool:

    return not np.isnan(tokens).any()


def no_nan_time_steps(time_steps: np.ndarray) -> bool:

    return not np.isnan(time_steps).any()


def time_steps_within_range(time_steps: np.ndarray) -> bool:
    return bool(np.all(time_steps >= 0) and np.all(time_steps <= 100 * 365.25))


def data_and_time_size_match(tokens: np.ndarray, time_steps: np.ndarray) -> bool:

    return tokens.size == time_steps.size


def tokenizer_contiguous(tokenizer: dict):

    token_vals = list(tokenizer.values())

    return (len(token_vals) == len(set(token_vals))) and (
        max(token_vals) == len(token_vals)
    )


def test_expansion_pack(expansion_pack_path, all_participants):

    assert required_files_exist(expansion_pack_path)

    tokenizer_path = os.path.join(expansion_pack_path, "tokenizer.yaml")
    with open(tokenizer_path, "r") as f:
        import yaml

        tokenizer = yaml.safe_load(f)
    lookup_path = os.path.join(expansion_pack_path, "p2i.csv")
    p2i = pd.read_csv(lookup_path, index_col="pid")
    data_path = os.path.join(expansion_pack_path, "data.bin")
    tokens = np.fromfile(data_path, dtype=np.uint32)
    time_path = os.path.join(expansion_pack_path, "time.bin")
    time_steps = np.fromfile(time_path, dtype=np.uint32)

    assert data_and_time_size_match(tokens=tokens, time_steps=time_steps)
    assert tokens_within_range(tokens=tokens, tokenizer=tokenizer)
    assert has_all_participants(p2i=p2i, pids=all_participants)
    assert all_start_pos_within_range(p2i=p2i, tokens=tokens)
    assert no_duplicate_start_pos(p2i=p2i)
    assert total_seq_len_add_up(p2i=p2i, tokens=tokens)
    assert no_nan_tokens(tokens=tokens)
    assert no_nan_time_steps(time_steps=time_steps)
    assert time_steps_within_range(time_steps=time_steps)
    assert tokenizer_contiguous(tokenizer=tokenizer)
