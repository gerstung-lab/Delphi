import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import coo_array

from delphi.data.transform import TransformArgs, parse_transform
from delphi.env import DELPHI_DATA_DIR
from delphi.multimodal import Modality
from delphi.tokenizer import Tokenizer, update_tokenizer


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


def biomarker_to_tensor(
    biomarker_X: dict[Modality, np.ndarray],
    biomarker_T: dict[Modality, np.ndarray],
    biomarker_C: dict[Modality, np.ndarray],
):

    for modality in biomarker_X.keys():
        m_X = biomarker_X[modality]
        m_T = biomarker_T[modality]
        biomarker_X[modality] = torch.from_numpy(m_X)  # type: ignore
        biomarker_T[modality] = torch.tensor(m_T)  # type: ignore
        biomarker_C[modality] = torch.from_numpy(biomarker_C[modality])  # type: ignore

    return biomarker_X, biomarker_T, biomarker_C


def pad_trailing_biomarkers(
    X: torch.Tensor,
    T: torch.Tensor,
    M: torch.Tensor,
):

    trailing_M = M[:, -1:]
    if trailing_M.max() > 1:
        M = torch.cat([M, trailing_M * 0], dim=1)
        X = torch.cat([X, torch.zeros_like(X[:, :1], dtype=X.dtype)], dim=1)
        T = torch.cat([T, T[:, -1:]], dim=1)

    return X, T, M


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

        P, X, T, biomarker_X, biomarker_T, biomarker_C = dataset.get_batch(idx)

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
        multi_T = torch.tensor(multi_T)
        M = torch.tensor(M, dtype=torch.long)
        biomarker_X, biomarker_T, biomarker_C = biomarker_to_tensor(
            biomarker_X=biomarker_X,
            biomarker_T=biomarker_T,
            biomarker_C=biomarker_C,
        )

        yield P, multi_X, multi_T, M, biomarker_X, biomarker_T, biomarker_C


def collate_batch_data(batch_data: list[np.ndarray]) -> np.ndarray:

    max_len = max([bd.size for bd in batch_data])
    collated_batch = np.full(
        shape=(len(batch_data), max_len),
        fill_value=0,
        dtype=batch_data[0].dtype,
    )
    for i, bd in enumerate(batch_data):
        collated_batch[i, : bd.size] = bd

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
class UKBDataConfig:
    data_dir: str = "ukb_real_data"
    subject_list: str = "participants.bin"
    expansion_pack_dir: str = "ukb_real_data/expansion_packs"
    expansion_packs: list[str] = field(default_factory=list)
    biomarker_dir: str = "ukb_real_data/biomarkers"
    biomarkers: list[str] = field(default_factory=list)
    seed: int = 42
    transforms: Optional[List[TransformArgs]] = None


class Biomarker:

    def __init__(self, path: str, memmap: bool = False):
        self.path = path
        data_path = os.path.join(path, "data.bin")
        if memmap:
            self.data = np.memmap(data_path, dtype=np.float32, mode="r")
        else:
            self.data = np.fromfile(data_path, dtype=np.float32)
        p2i = pd.read_csv(
            os.path.join(path, "p2i.csv"),
        )
        self.p2i = p2i.groupby("pid")

    def __repr__(self):
        return f"Biomarker(path={self.path})"

    def get_raw_batch(
        self, pids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        batch_data = []
        batch_time = []
        batch_count = []

        for pid in pids:
            pid_grp = self.p2i.get_group(pid)
            pid_time = []
            for i, l, t in zip(
                pid_grp["start_pos"], pid_grp["seq_len"], pid_grp["time"]
            ):
                if l > 0:
                    x = self.data[i : i + l]
                    batch_data.append(x)
                    batch_count.append(1)
                else:
                    batch_count.append(0)
                pid_time.append(t)
            batch_time.append(np.array(pid_time))

        batch_data = collate_batch_data(batch_data)
        batch_time = collate_batch_time(batch_time)
        batch_count = np.array(batch_count)

        return batch_data, batch_time, batch_count


class ExpansionPack:

    def __init__(self, path: str, offset: int, memmap: bool = False):

        p2i = pd.read_csv(os.path.join(path, "p2i.csv"), index_col="pid")
        self.offset = offset
        self.start_pos = p2i["start_pos"].to_dict()
        self.seq_len = p2i["seq_len"].to_dict()
        data_path = os.path.join(path, "data.bin")
        time_path = os.path.join(path, "time.bin")
        if memmap:
            self.tokens = np.memmap(data_path, dtype=np.uint32, mode="r")
            self.time_steps = np.memmap(time_path, dtype=np.uint32, mode="r")
        else:
            self.tokens = np.fromfile(data_path, dtype=np.uint32)
            self.time_steps = np.fromfile(data_path, dtype=np.uint32)

    def get_expansion(self, pid: int) -> tuple[np.ndarray, np.ndarray]:

        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l] + self.offset
        t_pid = self.time_steps[i : i + l]

        return x_pid, t_pid


class Dataset:

    def __init__(self, cfg: UKBDataConfig, memmap: bool = False):
        self.data_dir = os.path.join(DELPHI_DATA_DIR, cfg.data_dir)
        print(f"building dataset at {self.data_dir}")
        tokenizer_path = os.path.join(self.data_dir, "tokenizer.yaml")
        print(f"\t– loading tokenizer from {tokenizer_path}")
        with open(tokenizer_path, "r") as f:
            base_tokenizer = yaml.safe_load(f)

        p2i_path = os.path.join(self.data_dir, "p2i.csv")
        self.p2i = pd.read_csv(p2i_path, index_col="pid")
        self.start_pos = self.p2i["start_pos"].to_dict()
        self.seq_len = self.p2i["seq_len"].to_dict()
        participants_path = os.path.join(DELPHI_DATA_DIR, cfg.subject_list)
        tokens_path = os.path.join(self.data_dir, "data.bin")
        time_steps_path = os.path.join(self.data_dir, "time.bin")
        if memmap:
            self.participants = np.memmap(participants_path, dtype=np.uint32, mode="r")
            self.tokens = np.memmap(tokens_path, dtype=np.uint32, mode="r")
            self.time_steps = np.memmap(time_steps_path, dtype=np.uint32, mode="r")
        else:
            self.participants = np.fromfile(participants_path, dtype=np.uint32)
            self.tokens = np.fromfile(tokens_path, dtype=np.uint32)
            self.time_steps = np.fromfile(time_steps_path, dtype=np.uint32)

        cfg.expansion_packs.sort()
        self.expansion_packs = []
        self.expansion_pack_tokenizers = []
        for pack in cfg.expansion_packs:
            print(f"\t– loading expansion pack: {pack}")
            pack_path = os.path.join(DELPHI_DATA_DIR, cfg.expansion_pack_dir, pack)
            assert os.path.exists(pack_path), FileNotFoundError(
                f"expansion pack {pack_path} not found"
            )
            tokenizer_path = os.path.join(pack_path, "tokenizer.yaml")
            with open(tokenizer_path, "r") as f:
                add_tokenizer = yaml.safe_load(f)

            base_tokenizer, offset = update_tokenizer(
                base_tokenizer=base_tokenizer, add_tokenizer=add_tokenizer
            )
            self.expansion_pack_tokenizers.append(add_tokenizer)
            self.expansion_packs.append(
                ExpansionPack(path=pack_path, offset=offset, memmap=memmap)
            )
        self.tokenizer = Tokenizer(base_tokenizer)

        self.biomarker_dir = os.path.join(DELPHI_DATA_DIR, cfg.biomarker_dir)
        self.mod_ds = {}
        for modality in cfg.biomarkers:
            modality = Modality[modality.upper()]
            print(f"\t– loading biomarker: {modality.name}")
            biomarker_path = os.path.join(self.biomarker_dir, modality.name.lower())
            dataset = Biomarker(path=biomarker_path, memmap=memmap)
            self.mod_ds[modality] = dataset

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

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def expansion_tokens(self):
        token_keys = [
            k for tokenizer in self.expansion_pack_tokenizers for k in tokenizer.keys()
        ]
        return token_keys

    def get_raw_batch(self, batch_idx: np.ndarray):

        P = self.participants[batch_idx]
        X = []
        T = []
        for i, pid in enumerate(P):
            i = self.start_pos[pid]
            l = self.seq_len[pid]
            x_pid = self.tokens[i : i + l] + 1
            t_pid = self.time_steps[i : i + l]
            for expansion_pack in self.expansion_packs:
                pack_x_pid, pack_t_pid = expansion_pack.get_expansion(pid)
                x_pid = np.concatenate((x_pid, pack_x_pid))
                t_pid = np.concatenate((t_pid, pack_t_pid))
            X.append(x_pid)
            T.append(t_pid)

        X = collate_batch_data(X)
        T = collate_batch_time(T)

        biomarker_X = {}
        biomarker_T = {}
        biomarker_C = {}
        for modality, dataset in self.mod_ds.items():
            m_X, m_T, m_C = dataset.get_raw_batch(P)
            biomarker_X[modality] = m_X
            biomarker_T[modality] = m_T
            biomarker_C[modality] = m_C

        return P, X, T, biomarker_X, biomarker_T, biomarker_C

    def get_batch(self, batch_idx):

        P, X, T, biomarker_X, biomarker_T, biomarker_C = self.get_raw_batch(batch_idx)

        for transform in self.transforms:
            X, T = transform(X, T)

        return P, X, T, biomarker_X, biomarker_T, biomarker_C


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

        P, X, T, biomarker_X, biomarker_T, biomarker_C = super().get_raw_batch(
            batch_idx
        )

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
        return P, X, T, biomarker_X, biomarker_T, biomarker_C
