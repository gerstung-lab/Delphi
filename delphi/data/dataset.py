import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np
import torch
from scipy.sparse import coo_array

from delphi.data.family_hx import FamilyHxConfig, FamilyHxDataset
from delphi.data.prs import PRSConfig, PRSDataset
from delphi.data.transform import TransformArgs, parse_transform
from delphi.multimodal import Modality, modality_X_dtype
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

    sort_by_time = np.argsort(T, axis=1)
    X = np.take_along_axis(X, sort_by_time, axis=1)
    T = np.take_along_axis(T, sort_by_time, axis=1)

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


def biomarker_to_tensor(
    biomarker_X: dict[Modality, np.ndarray],
    biomarker_T: dict[Modality, np.ndarray],
):

    for modality in biomarker_X.keys():
        m_X = biomarker_X[modality]
        m_T = biomarker_T[modality]
        biomarker_X[modality] = torch.tensor(m_X, dtype=modality_X_dtype[modality])  # type: ignore
        biomarker_T[modality] = torch.tensor(m_T, dtype=torch.float)  # type: ignore

    return biomarker_X, biomarker_T


def remove_trailing_biomarkers(
    M: np.ndarray,
    biomarker_X: dict[Modality, np.ndarray] = {},
):

    l = M.shape[1]

    for modality in biomarker_X.keys():

        biomarker_idx = torch.nonzero(M == modality.value)
        is_trailing = biomarker_idx[:, 1] == l - 1
        biomarker_X[modality] = biomarker_X[modality][~is_trailing]

        assert biomarker_X[modality].shape[0] == (~is_trailing).sum()

    return biomarker_X


def multimodal_T_and_X(
    X: np.ndarray,
    T: np.ndarray,
    biomarker_T: dict[Modality, np.ndarray] = {},
):

    M = np.zeros_like(X, dtype=np.int8)
    M[X > 0] = 1

    for modality in biomarker_T.keys():

        m_T = biomarker_T[modality]

        m_M = np.zeros_like(m_T, dtype=np.int8)
        m_M[m_T != -1e4] = modality.value

        m_X = np.zeros_like(m_T, dtype=np.int8)

        T = np.concatenate((T, m_T), axis=1)
        M = np.concatenate((M, m_M), axis=1)
        X = np.concatenate((X, m_X), axis=1)

    return M, T, X


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


def load_sequences(
    it: Iterator,
    dataset: "Dataset",
) -> Iterator:

    for idx in it:

        P, X, T, biomarker_X, biomarker_T = dataset.get_batch(idx)

        M, multi_T, multi_X = multimodal_T_and_X(X=X, T=T, biomarker_T=biomarker_T)

        s = np.argsort(multi_T, axis=1)
        M = np.take_along_axis(M, s, axis=1)
        multi_T = np.take_along_axis(multi_T, s, axis=1)
        multi_X = np.take_along_axis(multi_X, s, axis=1)

        squeeze_margin = np.min(np.sum(M == 0, axis=1))
        M = M[:, squeeze_margin:]
        multi_X = multi_X[:, squeeze_margin:]
        multi_T = multi_T[:, squeeze_margin:]

        P = torch.tensor(P, dtype=torch.long)
        multi_X = torch.tensor(multi_X, dtype=torch.long)
        multi_T = torch.tensor(multi_T, dtype=torch.float)
        M = torch.tensor(M, dtype=torch.long)
        biomarker_X, biomarker_T = biomarker_to_tensor(
            biomarker_X=biomarker_X, biomarker_T=biomarker_T
        )

        yield P, multi_X, multi_T, M, biomarker_X


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
    prs: PRSConfig = field(default_factory=PRSConfig)
    family_hx: FamilyHxConfig = field(default_factory=FamilyHxConfig)
    seed: int = 42
    transforms: Optional[List[TransformArgs]] = None


class Dataset:

    def __init__(
        self,
        cfg: UKBDataConfig,
    ):
        print(f"\nbuilding dataset at {cfg.data_dir}")
        tokenizer_path = os.path.join(cfg.data_dir, cfg.tokenizer_fname)
        print(f" – loading tokenizer from {tokenizer_path}")
        self.tokenizer = load_tokenizer_from_yaml(tokenizer_path)

        memmap_path = os.path.join(cfg.data_dir, cfg.memmap_fname)
        data = np.fromfile(memmap_path, dtype=np.uint32).reshape(-1, 3)
        print(f" – loading memmap from {memmap_path}")

        p2i = get_p2i(data)
        self.participants = data[:, 0][
            p2i[:, 0]
        ]  # todo: investigate duplicate participants

        self.seq_len = p2i[:, 1]
        assert self.seq_len.size == self.participants.size

        self.start_pos = np.cumsum(p2i[:, 1]) - p2i[:, 1]
        assert self.start_pos.size == self.participants.size

        self.tokens = data[:, 2] + 1
        self.time_steps = data[:, 1]
        assert self.tokens.size == self.time_steps.size

        self.mod_ds = {}
        if cfg.prs.include or cfg.prs.must:
            prs_path = os.path.join(cfg.data_dir, cfg.prs.lmdb_fname)
            assert os.path.exists(prs_path)
            print(f" – found prs lmdb dataset at {prs_path}")
            prs_dataset = PRSDataset(db_path=prs_path)
            prs_participants = prs_dataset.get_all_pids()
            if cfg.prs.include:
                self.mod_ds[Modality.PRS] = prs_dataset
            if cfg.prs.must:
                keep_participants = np.isin(self.participants, prs_participants)
                print(
                    f"keeping {np.sum(keep_participants)}/{self.participants.size} participants with prs"
                )
                self.participants = self.participants[keep_participants]
                self.seq_len = self.seq_len[keep_participants]
                self.start_pos = self.start_pos[keep_participants]

        if cfg.family_hx.include or cfg.family_hx.must:
            family_hx_path = os.path.join(cfg.data_dir, cfg.family_hx.lmdb_fname)
            assert os.path.exists(family_hx_path)
            print(f" – found family_hx lmdb dataset at {family_hx_path}")
            family_hx_dataset = FamilyHxDataset(
                db_path=family_hx_path, map_yaml=cfg.family_hx.map_yaml
            )
            family_hx_participants = family_hx_dataset.get_all_pids()
            if cfg.family_hx.include:
                self.mod_ds[Modality.FAMILY_HX] = family_hx_dataset
            if cfg.family_hx.must:
                keep_participants = np.isin(self.participants, family_hx_participants)
                print(
                    f"keeping {np.sum(keep_participants)}/{self.participants.size} participants with family hx"
                )
                self.participants = self.participants[keep_participants]
                self.seq_len = self.seq_len[keep_participants]
                self.start_pos = self.start_pos[keep_participants]

        self.transforms = []
        if cfg.transforms is not None:
            for transform in cfg.transforms:
                transform = parse_transform(
                    transform, tokenizer=self.tokenizer, seed=cfg.seed
                )
                self.transforms.append(transform)

        print(f"built dataset!")

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

        biomarker_X = {}
        biomarker_T = {}
        for modality, dataset in self.mod_ds.items():
            m_X, m_T = dataset.get_raw_batch([str(P_i) for P_i in P])
            biomarker_X[modality] = m_X
            biomarker_T[modality] = m_T

        return P, X, T, biomarker_X, biomarker_T

    def get_batch(self, batch_idx):

        P, X, T, biomarker_X, biomarker_T = self.get_raw_batch(batch_idx)

        for transform in self.transforms:
            X, T = transform(X, T)

        return P, X, T, biomarker_X, biomarker_T


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

        P, X, T, biomarker_X, biomarker_T = super().get_raw_batch(batch_idx)

        X[T > self.start_age] = 0
        T[T > self.start_age] = -1e4

        X = np.pad(
            X,
            ((0, 0), (0, 1)),
            mode="constant",
            constant_values=self.tokenizer["no_event"],
        )
        T = np.pad(T, ((0, 0), (0, 1)), mode="constant", constant_values=self.start_age)

        # TODO: remove biomarkers after prompt_age
        return P, X, T, biomarker_X, biomarker_T
