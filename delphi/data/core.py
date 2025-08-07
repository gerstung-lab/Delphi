import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import coo_array

from delphi.data.transform import (
    add_no_event,
    crop_contiguous,
    sort_by_time,
    trim_margin,
)
from delphi.env import DELPHI_DATA_DIR
from delphi.tokenizer import Tokenizer


def get_p2i(data):
    """
    Get the patient to index mapping. (patient index in data -> length of sequence)
    """

    px = data[:, 0].astype("int")
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
        if i == len(px) - 1:
            # add last participant
            p2i.append([j, i - j + 1])
    return np.array(p2i)


def subsample_tricolumnar(
    XT: np.ndarray,
    logits: np.ndarray,
    subsample: Optional[int] = None,
):
    """
    subsample the number of trajectories to avoid excessive RAM usage during subsequent tricolumnar_to_2d conversion
    """

    p2i = get_p2i(XT)
    N = p2i.shape[0]
    if subsample is not None and subsample <= N:
        n_to_keep = p2i[:subsample, 1].sum()
        XT = XT[:n_to_keep]
        logits = logits[:n_to_keep]

    return XT, logits


def tricolumnar_to_2d(data):
    """
    Convert a tricolumnar array to a 2D array.
    The first column is the participant index, the second column is the time step,
    and the third column is the token.
    """
    p2i = get_p2i(data)
    sub_idx = np.repeat(np.arange(p2i.shape[0]), p2i[:, 1])
    pos_idx = np.concatenate([np.arange(p2i[i, 1]) for i in range(p2i.shape[0])])

    X = coo_array(
        (data[:, 2], (sub_idx, pos_idx)), shape=(p2i.shape[0], p2i[:, 1].max())
    ).toarray()

    T = np.full(X.shape, -10000, dtype=np.float32)
    T[sub_idx, pos_idx] = data[:, 1]

    sort_by_time = np.argsort(T, axis=1)
    X = np.take_along_axis(X, sort_by_time, axis=1)
    T = np.take_along_axis(T, sort_by_time, axis=1)

    return X, T


def eval_iter(total_size: int, batch_size: int) -> Iterator[np.ndarray]:

    batch_start_pos = np.arange(0, total_size, batch_size)
    batch_end_pos = batch_start_pos + batch_size
    batch_end_pos[-1] = total_size

    for start, end in zip(batch_start_pos, batch_end_pos):
        yield np.arange(start, end)


def train_iter(
    rng: np.random.Generator, total_size: int, batch_size: int
) -> Iterator[np.ndarray]:

    while True:
        yield rng.integers(total_size, size=(batch_size,))


def collate_batch_data(batch_data: list[np.ndarray]) -> np.ndarray:

    if len(batch_data) > 0:
        max_len = max([bd.size for bd in batch_data])
        collated_batch = np.full(
            shape=(len(batch_data), max_len),
            fill_value=0,
            dtype=batch_data[0].dtype,
        )
        for i, bd in enumerate(batch_data):
            collated_batch[i, : bd.size] = bd
    else:
        return np.empty(shape=(0, 0), dtype=np.float64)

    return collated_batch


def collate_batch_time(batch_time: list[np.ndarray]) -> np.ndarray:

    max_len = max([bt.size for bt in batch_time])
    collated_batch = np.full(
        shape=(len(batch_time), max_len), fill_value=-1e4, dtype=np.float32
    )
    for i, bt in enumerate(batch_time):
        collated_batch[i, : bt.size] = bt

    return collated_batch


@dataclass
class BaseDataConfig:
    data_dir: str = "ukb_real_data"
    subject_list: str = "participants.bin"
    seed: int = 42
    no_event_interval: Optional[float] = 5
    block_size: Optional[int] = 64


class BaseDataset:

    def __init__(self, cfg: BaseDataConfig, memmap: bool = False):

        (
            tokenizer,
            self.start_pos,
            self.seq_len,
            self.participants,
            self.tokens,
            self.time_steps,
            self.rng,
        ) = load_core_data_package(cfg=cfg, memmap=memmap)
        self.tokenizer = Tokenizer(tokenizer)

        if cfg.no_event_interval is not None:
            self.add_no_event = functools.partial(
                add_no_event,
                rng=self.rng,
                interval=cfg.no_event_interval,
                token=self.tokenizer["no_event"],
            )
        else:
            self.add_no_event = lambda *args: args

        if cfg.block_size is not None:
            self.crop_block_size = functools.partial(
                crop_contiguous, block_size=cfg.block_size, rng=self.rng
            )
        else:
            self.crop_block_size = lambda *args: args

    def __len__(self):
        return self.participants.size

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def get_batch(self, batch_idx: np.ndarray):

        P = self.participants[batch_idx]
        X = []
        T = []
        for i, pid in enumerate(P):
            i = self.start_pos[pid]
            l = self.seq_len[pid]
            x_pid = self.tokens[i : i + l]
            t_pid = self.time_steps[i : i + l]
            X.append(x_pid)
            T.append(t_pid)

        X = collate_batch_data(X)
        T = collate_batch_time(T)

        X, T = self.add_no_event(X, T)
        T, X = sort_by_time(T, X)
        X, T = trim_margin(X, T, trim_val=0)
        X, T = self.crop_block_size(X, T)

        return X, T


def build_dataset(cfg: dict):

    return BaseDataset(BaseDataConfig(**cfg))


def build_datasets(train_cfg_dict, val_cfg_dict):

    train_cfg = BaseDataConfig(**train_cfg_dict)
    val_cfg = BaseDataConfig(**val_cfg_dict)

    val_cfg.no_event_interval = train_cfg.no_event_interval
    val_cfg.block_size = train_cfg.block_size

    train_ds = BaseDataset(train_cfg)
    val_ds = BaseDataset(val_cfg)

    return train_ds, val_ds


def load_sequences(
    it: Iterator,
    dataset: BaseDataset,
) -> Iterator:

    for idx in it:

        X, T = dataset.get_batch(idx)
        X = torch.tensor(X, dtype=torch.long)
        T = torch.tensor(T, dtype=torch.float32)

        yield X, T


def load_prompt_sequences(it: Iterator, dataset: BaseDataset, start_age: float):

    for idx in it:

        X, T = dataset.get_batch(idx)

        too_old = T > start_age
        X[too_old] = 0
        T[too_old] = -1e4

        pad = ((0, 0), (0, 1))
        no_event_token = dataset.tokenizer["no_event"]
        X = np.pad(X, pad, "constant", constant_values=no_event_token)
        T = np.pad(T, pad, "constant", constant_values=start_age)

        T, X = sort_by_time(T, X)
        X, T = trim_margin(X, T, trim_val=0)

        X = torch.tensor(X, dtype=torch.long)
        T = torch.tensor(T, dtype=torch.float32)

        yield X, T


def duplicate_participants(X: torch.Tensor, T: torch.Tensor, n_repeat: int):

    return X.repeat(n_repeat, 1), T.repeat(n_repeat, 1)


def load_core_data_package(cfg: BaseDataConfig, memmap: bool = False):

    dataset_dir = Path(DELPHI_DATA_DIR) / cfg.data_dir
    print(f"building dataset at {dataset_dir}")
    tokenizer_path = dataset_dir / "tokenizer.yaml"
    print(f"\t– loading tokenizer from {tokenizer_path}")
    with open(tokenizer_path, "r") as f:
        tokenizer = yaml.safe_load(f)

    p2i = pd.read_csv(dataset_dir / "p2i.csv", index_col="pid")
    start_pos = p2i["start_pos"].to_dict()
    seq_len = p2i["seq_len"].to_dict()

    participants_path = Path(DELPHI_DATA_DIR) / cfg.subject_list
    tokens_path = dataset_dir / "data.bin"
    time_steps_path = dataset_dir / "time.bin"
    if memmap:
        participants = np.memmap(participants_path, dtype=np.uint32, mode="r")
        tokens = np.memmap(tokens_path, dtype=np.uint32, mode="r")
        timesteps = np.memmap(time_steps_path, dtype=np.uint32, mode="r")
    else:
        participants = np.fromfile(participants_path, dtype=np.uint32)
        tokens = np.fromfile(tokens_path, dtype=np.uint32)
        timesteps = np.fromfile(time_steps_path, dtype=np.uint32)

    rng = np.random.default_rng(cfg.seed)

    return tokenizer, start_pos, seq_len, participants, tokens, timesteps, rng
