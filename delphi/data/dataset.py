import os
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
import torch
from scipy.sparse import coo_array

from delphi.data.transform import TransformArgs, parse_transform
from delphi.tokenizer import load_tokenizer_from_yaml


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

    return X, T


def sort_by_time(X: np.ndarray, T: np.ndarray):

    s = np.argsort(T, axis=1)
    X = np.take_along_axis(X, s, axis=1)
    T = np.take_along_axis(T, s, axis=1)

    return X, T


def squeeze(X: np.ndarray, T: np.ndarray):

    squeeze_margin = np.min(np.sum(X == 0, axis=1))
    X = X[:, squeeze_margin:]
    T = T[:, squeeze_margin:]

    return X, T


def eval_iter(total_size: int, batch_size: int) -> Iterator[np.ndarray]:

    batch_start_pos = np.arange(0, total_size, batch_size)
    batch_end_pos = batch_start_pos + batch_size
    batch_end_pos[-1] = total_size

    for start, end in zip(batch_start_pos, batch_end_pos):
        yield np.arange(start, end)


def train_iter(total_size: int, batch_size: int) -> Iterator[np.ndarray]:

    while True:
        yield np.random.randint(total_size, size=(batch_size,))


def load_sequences(
    it: Iterator,
    dataset: "Dataset",
) -> Iterator:

    for idx in it:

        P, X, T = dataset.get_batch(idx)

        X, T = sort_by_time(X=X, T=T)

        X, T = squeeze(X=X, T=T)

        P = torch.tensor(P, dtype=torch.long)
        X = torch.tensor(X, dtype=torch.long)
        T = torch.tensor(T, dtype=torch.float)

        yield P, X, T


def build_prefetch_loader(loader: Iterator) -> Iterator:

    sentinel = object()
    next_batch = next(loader)

    while True:
        yield next_batch
        next_batch = next(loader, sentinel)
        if next_batch is sentinel:
            return


@dataclass
class UKBDataConfig:
    data_dir: str = "data/ukb_simulated_data"
    memmap_fname: str = "train.bin"
    tokenizer_fname: str = "tokenizer.yaml"
    transforms: Optional[List[TransformArgs]] = None


class Dataset:

    def __init__(
        self,
        cfg: UKBDataConfig,
    ):
        tokenizer_path = os.path.join(cfg.data_dir, cfg.tokenizer_fname)
        self.tokenizer = load_tokenizer_from_yaml(tokenizer_path)

        memmap_path = os.path.join(cfg.data_dir, cfg.memmap_fname)
        data = np.fromfile(memmap_path, dtype=np.uint32).reshape(-1, 3)

        p2i = get_p2i(data)
        self.participants = p2i[:, 0]

        self.seq_len = p2i[:, 1]
        assert self.seq_len.size == self.participants.size

        self.start_pos = np.cumsum(p2i[:, 1]) - p2i[:, 1]
        assert self.start_pos.size == self.participants.size

        self.tokens = data[:, 2] + 1
        self.time_steps = data[:, 1]
        assert self.tokens.size == self.time_steps.size

        self.transforms = []
        if cfg.transforms is not None:
            for transform in cfg.transforms:
                transform = parse_transform(transform, tokenizer=self.tokenizer)
                self.transforms.append(transform)

    def __len__(self):

        return self.participants.size

    def get_raw_batch(self, batch_idx):

        batch_size = batch_idx.size
        batch_seq_len = self.seq_len[batch_idx]
        batch_seq_slice = np.concatenate(
            [
                np.arange(s, s + l)
                for s, l in zip(self.start_pos[batch_idx], batch_seq_len)
            ]
        )
        batch_tokens = self.tokens[batch_seq_slice]
        batch_time_steps = self.time_steps[batch_seq_slice]
        row_idx = np.repeat(np.arange(batch_size), batch_seq_len)
        col_idx = np.concatenate([np.arange(length) for length in batch_seq_len])

        P = self.participants[batch_idx]

        X = coo_array(
            (batch_tokens, (row_idx, col_idx)), shape=(batch_size, batch_seq_len.max())
        ).toarray()
        T = np.full(X.shape, -10000, dtype=np.float32)
        T[row_idx, col_idx] = batch_time_steps

        return P, X, T

    def get_batch(self, batch_idx):

        P, X, T = self.get_raw_batch(batch_idx)

        for transform in self.transforms:
            X, T = transform(X, T)

        return P, X, T


class PromptDataset(Dataset):

    def __init__(
        self,
        cfg: UKBDataConfig,
        start_age_in_years: float,
    ):

        super().__init__(cfg=cfg)

        self.start_age = start_age_in_years * 365.25
        first_timestep = self.time_steps[self.start_pos]
        last_timestep = self.time_steps[self.start_pos + self.seq_len - 1]
        time_mask = (first_timestep < self.start_age) & (last_timestep > self.start_age)

        self.participants = self.participants[time_mask]
        self.seq_len = self.seq_len[time_mask]
        self.start_pos = self.start_pos[time_mask]

    def get_batch(self, batch_idx):

        P, X, T = super().get_raw_batch(batch_idx)

        X[T > self.start_age] = 0
        T[T > self.start_age] = -1e4

        X = np.pad(
            X,
            ((0, 0), (0, 1)),
            mode="constant",
            constant_values=self.tokenizer["no_event"],
        )
        T = np.pad(T, ((0, 0), (0, 1)), mode="constant", constant_values=self.start_age)

        return P, X, T
