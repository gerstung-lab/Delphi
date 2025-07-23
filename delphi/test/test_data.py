import os

import numpy as np
import pandas as pd

from delphi.test.config import DATASET_DIR, PARTICIPANTS
from delphi.test.test_biomarkers import has_all_participants
from delphi.test.test_expansion_packs import (
    all_start_pos_within_range,
    data_and_time_size_match,
    no_duplicate_start_pos,
    no_nan_time_steps,
    no_nan_tokens,
    required_files_exist,
    time_steps_within_range,
    total_seq_len_add_up,
)
from delphi.tokenizer import CoreEvents


def tokenizer_contiguous(tokenizer: dict):

    token_vals = list(tokenizer.values())

    return (len(token_vals) == len(set(token_vals))) and (
        max(token_vals) == len(token_vals) - 1
    )


def tokenizer_contains_required_pairs(tokenizer: dict):

    mandatory_tokens = [CoreEvents.PADDING.value, CoreEvents.NO_EVENT.value]
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


def test_data():

    assert required_files_exist(DATASET_DIR)

    tokenizer_path = os.path.join(DATASET_DIR, "tokenizer.yaml")
    with open(tokenizer_path, "r") as f:
        import yaml

        tokenizer = yaml.safe_load(f)
    lookup_path = os.path.join(DATASET_DIR, "p2i.csv")
    p2i = pd.read_csv(lookup_path, index_col="pid")
    data_path = os.path.join(DATASET_DIR, "data.bin")
    tokens = np.fromfile(data_path, dtype=np.uint32)
    time_path = os.path.join(DATASET_DIR, "time.bin")
    time_steps = np.fromfile(time_path, dtype=np.uint32)

    assert data_and_time_size_match(tokens=tokens, time_steps=time_steps)
    assert tokens_within_range(tokens=tokens, tokenizer=tokenizer)
    assert has_all_participants(p2i=p2i, pids=PARTICIPANTS)
    assert all_start_pos_within_range(p2i=p2i, tokens=tokens)
    assert no_duplicate_start_pos(p2i=p2i)
    assert total_seq_len_add_up(p2i=p2i, tokens=tokens)
    assert no_nan_tokens(tokens=tokens)
    assert no_nan_time_steps(time_steps=time_steps)
    assert time_steps_within_range(time_steps=time_steps)
    assert tokenizer_contiguous(tokenizer=tokenizer)
    assert tokenizer_contains_required_pairs(tokenizer=tokenizer)
