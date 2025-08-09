import os

import numpy as np
import pandas as pd

from delphi.test.test_biomarkers import has_all_participants
from delphi.test.test_expansion_packs import (
    all_start_pos_within_range,
    data_and_time_size_match,
    no_duplicate_start_pos,
    no_nan_time_steps,
    no_nan_tokens,
    required_files_exist,
    total_seq_len_add_up,
)
from delphi.tokenizer import FEMALE, MALE, NO_EVENT, PADDING


def tokenizer_contiguous(tokenizer: dict):

    token_vals = list(tokenizer.values())

    return (len(token_vals) == len(set(token_vals))) and (
        max(token_vals) == len(token_vals) - 1
    )


def tokenizer_contains_required_pairs(tokenizer: dict):

    mandatory_tokens = [PADDING, NO_EVENT, MALE, FEMALE]
    for token in mandatory_tokens:
        if token not in tokenizer.keys():
            return False

    return True


def tokens_within_range(tokens: np.ndarray, tokenizer: dict) -> bool:
    min_token = np.min(tokens)
    max_token = np.max(tokens)

    max_in_tokenizer = max(tokenizer.values())
    min_in_tokenizer = min(tokenizer.values())

    return (
        min_token >= min_in_tokenizer
        and max_token <= max_in_tokenizer
        and min_token >= 2
    )


def positive_timesteps(time_steps: np.ndarray) -> bool:
    return np.all(time_steps >= 0).astype(bool)


def test_data(dataset_dir, all_participants):

    assert required_files_exist(dataset_dir)

    tokenizer_path = os.path.join(dataset_dir, "tokenizer.yaml")
    with open(tokenizer_path, "r") as f:
        import yaml

        tokenizer = yaml.safe_load(f)
    lookup_path = os.path.join(dataset_dir, "p2i.csv")
    p2i = pd.read_csv(lookup_path)
    data_path = os.path.join(dataset_dir, "data.bin")
    tokens = np.fromfile(data_path, dtype=np.uint32)
    time_path = os.path.join(dataset_dir, "time.bin")
    time_steps = np.fromfile(time_path, dtype=np.uint32)

    assert data_and_time_size_match(tokens=tokens, time_steps=time_steps)
    assert tokens_within_range(tokens=tokens, tokenizer=tokenizer)
    assert has_all_participants(p2i=p2i, pids=all_participants)
    assert all_start_pos_within_range(p2i=p2i, tokens=tokens)
    assert no_duplicate_start_pos(p2i=p2i)
    assert total_seq_len_add_up(p2i=p2i, tokens=tokens)
    assert no_nan_tokens(tokens=tokens)
    assert no_nan_time_steps(time_steps=time_steps)
    assert positive_timesteps(time_steps=time_steps)
    assert tokenizer_contiguous(tokenizer=tokenizer)
    assert tokenizer_contains_required_pairs(tokenizer=tokenizer)
