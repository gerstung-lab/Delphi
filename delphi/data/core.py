import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import coo_array

from delphi.data.transform import add_no_event, crop_contiguous, sort_by_time
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
    seed: int,
    total_size: int,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    step: int = 0,
) -> Iterator[np.ndarray]:

    while True:
        seed_with_offset = seed + step * world_size + rank
        rng = np.random.default_rng(seed_with_offset)
        batch_idx = rng.integers(total_size, size=(batch_size,))
        step += 1

        yield batch_idx


def move_batch_to_device(args: Iterable, device: str):

    return tuple([arg.to(device) for arg in args])


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

    def __getitem__(self, idx: int):

        pid = self.participants[idx]
        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l]
        t_pid = self.time_steps[i : i + l]
        x_pid, t_pid = self.add_no_event(x_pid, t_pid)
        x_pid, t_pid = self.crop_block_size(x_pid, t_pid)

        return x_pid, t_pid

    def get_batch(self, batch_idx: Iterator):

        X = []
        T = []
        for idx in batch_idx:
            x, t = self[idx]
            X.append(x)
            T.append(t)

        X = collate_batch_data(X)
        T = collate_batch_time(T)
        T, X = sort_by_time(T, X)

        X = torch.tensor(X, dtype=torch.long)
        T = torch.tensor(T, dtype=torch.float32)

        return X[:, :-1], T[:, :-1], X[:, 1:], T[:, 1:]


def build_dataset(cfg: dict):

    return BaseDataset(BaseDataConfig(**cfg))


def load_prompt_sequences(it: Iterator, dataset: BaseDataset, start_age: float):

    for batch_idx in it:
        x_prompt_lst, t_prompt_lst = list(), list()
        x_lst, t_lst = list(), list()

        for idx in batch_idx:
            x, t = dataset[idx]
            x_lst.append(x.copy())
            t_lst.append(t.copy())

            too_old = t > start_age
            x[too_old] = 0
            t[too_old] = -1e4
            no_event_token = dataset.tokenizer["no_event"]
            x = np.pad(x, (0, 1), "constant", constant_values=no_event_token)
            t = np.pad(t, (0, 1), "constant", constant_values=start_age)
            x_prompt_lst.append(x)
            t_prompt_lst.append(t)

        X = collate_batch_data(x_lst)
        T = collate_batch_time(t_lst)
        T, X = sort_by_time(T, X)
        X = torch.tensor(X, dtype=torch.long)
        T = torch.tensor(T, dtype=torch.float32)

        X_prompt = collate_batch_data(x_prompt_lst)
        T_prompt = collate_batch_time(t_prompt_lst)
        T_prompt, X_prompt = sort_by_time(T_prompt, X_prompt)
        X_prompt = torch.tensor(X_prompt, dtype=torch.long)
        T_prompt = torch.tensor(T_prompt, dtype=torch.float32)

        yield X_prompt, T_prompt, X, T


def duplicate_participants(X: torch.Tensor, T: torch.Tensor, n_repeat: int):

    return X.repeat(n_repeat, 1), T.repeat(n_repeat, 1)


def load_core_data_package(cfg: BaseDataConfig, memmap: bool = False):

    dataset_dir = Path(DELPHI_DATA_DIR) / cfg.data_dir
    tokenizer_path = dataset_dir / "tokenizer.yaml"
    with open(tokenizer_path, "r") as f:
        tokenizer = yaml.safe_load(f)

    p2i = pd.read_csv(dataset_dir / "p2i.csv", index_col="pid")
    start_pos = p2i["start_pos"].to_dict()
    seq_len = p2i["seq_len"].to_dict()

    participants_path = dataset_dir / cfg.subject_list
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
