import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from delphi.data.transform import append_no_event, crop_contiguous, trim_margin
from delphi.env import DELPHI_DATA_DIR


def _identity_transform(*args):
    return args


def sort_by_time(t: np.ndarray, *args: np.ndarray):
    s = np.argsort(t)
    t = t[s]
    return t, *[arg[s] for arg in args]


def perturb_time(
    x: np.ndarray,
    t: np.ndarray,
    tokens: np.ndarray,
    rng: np.random.Generator,
    low: int = int(-20 * 365.25),
    high: int = int(40 * 365.25),
):
    to_perturb = np.isin(x, tokens)
    t[to_perturb] += rng.integers(low=low, high=high, size=(to_perturb.sum(),))
    return x, t


def collate_batch(
    batch_data: list[np.ndarray], fill_value: int | float = 0, pad_left: bool = True
) -> np.ndarray:

    max_len = max([bd.size for bd in batch_data])
    collated_batch = np.full(
        shape=(len(batch_data), max_len),
        fill_value=fill_value,
        dtype=batch_data[0].dtype,
    )
    for i, bd in enumerate(batch_data):
        if pad_left:
            collated_batch[i, -bd.size :] = bd
        else:
            collated_batch[i, : bd.size] = bd

    return collated_batch


@dataclass
class UKBDataConfig:
    data_dir: str = "ukb_real_data"
    subject_list: str = "participants/train_fold.bin"
    seed: int = 42
    no_event_interval: Optional[float] = 5 * 365.25
    block_size: Optional[int] = 64


class UKBDataset:

    def __init__(
        self,
        data_dir: str = "ukb_real_data",
        subject_list: str = "participants/train_fold.bin",
        no_event_interval: Optional[float] = 5 * 365.25,
        block_size: Optional[int] = None,
        perturb: bool = True,
        perturb_tokens: Optional[list] = None,
        seed: int = 42,
        memmap: bool = False,
    ):

        (
            self.tokenizer,
            self.start_pos,
            self.seq_len,
            self.participants,
            self.tokens,
            self.time_steps,
        ) = load_core_data_package(
            data_dir=data_dir, subject_list=subject_list, memmap=memmap
        )
        self.rng = np.random.default_rng(seed)

        if no_event_interval is not None:
            self.append_no_event = functools.partial(
                append_no_event,
                rng=self.rng,
                interval=no_event_interval,
                token=self.tokenizer["no_event"],
            )
        else:
            self.append_no_event = _identity_transform

        if perturb:
            if perturb_tokens is None:
                perturb_tokens = [
                    "bmi_low",
                    "bmi_mid",
                    "bmi_high",
                    "smoking_low",
                    "smoking_mid",
                    "smoking_high",
                    "alcohol_low",
                    "alcohol_mid",
                    "alcohol_high",
                ]
            self.perturb_tokens = np.array(
                [self.tokenizer[event] for event in perturb_tokens]
            )
            self.perturb_time = functools.partial(
                perturb_time, tokens=self.perturb_tokens, rng=self.rng
            )
        else:
            self.perturb_time = _identity_transform

        if block_size is not None:
            self.crop_block_size = functools.partial(
                crop_contiguous, block_size=block_size, rng=self.rng
            )
        else:
            self.crop_block_size = _identity_transform

    def __len__(self):
        return self.participants.size

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def __getitem__(self, idx: int):

        pid = self.participants[idx]
        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l]
        t_pid = self.time_steps[i : i + l]
        x_pid, t_pid = self.append_no_event(x_pid, t_pid)
        x_pid, t_pid = self.perturb_time(x_pid, t_pid)
        t_pid, x_pid = sort_by_time(t_pid, x_pid)
        x_pid, t_pid = self.crop_block_size(x_pid, t_pid)

        return x_pid, t_pid

    def get_batch(self, batch_idx: Iterable, cut: bool = True):

        X = []
        T = []
        for idx in batch_idx:
            x, t = self[idx]
            X.append(x)
            T.append(t)

        X = collate_batch(X)
        T = collate_batch(T, fill_value=-1e4)

        X = torch.tensor(X, dtype=torch.long)
        T = torch.tensor(T, dtype=torch.float32)

        if cut:
            return X[:, :-1], T[:, :-1], X[:, 1:], T[:, 1:]
        else:
            return X, T

    def get_prompt_batch(self, batch_idx: Iterable, start_age: float):

        X = []
        T = []
        for idx in batch_idx:
            x, t = self[idx]
            too_old = t > start_age
            x[too_old] = 0
            t[too_old] = -1e4
            no_event_token = self.tokenizer["no_event"]
            x = np.pad(x, (0, 1), "constant", constant_values=no_event_token)
            t = np.pad(t, (0, 1), "constant", constant_values=start_age)
            X.append(x)
            T.append(t)

        X = collate_batch(X)
        T = collate_batch(T, fill_value=-1e4)
        X, T = trim_margin(X, T, trim_val=0)

        X = torch.tensor(X, dtype=torch.long)
        T = torch.tensor(T, dtype=torch.float32)

        return X, T


def load_core_data_package(data_dir: str, subject_list: str, memmap: bool = False):

    dataset_dir = Path(DELPHI_DATA_DIR) / data_dir
    tokenizer_path = dataset_dir / "tokenizer.yaml"
    with open(tokenizer_path, "r") as f:
        tokenizer = yaml.safe_load(f)

    p2i = pd.read_csv(dataset_dir / "p2i.csv", index_col="pid")
    start_pos = p2i["start_pos"].to_dict()
    seq_len = p2i["seq_len"].to_dict()

    participants_path = dataset_dir / subject_list
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

    return tokenizer, start_pos, seq_len, participants, tokens, timesteps
