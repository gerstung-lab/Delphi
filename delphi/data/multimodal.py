import os
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import yaml

from delphi.data.core import (
    BaseDataConfig,
    collate_batch_data,
    collate_batch_time,
    load_core_data_package,
    load_transforms,
)
from delphi.data.transform import sort_by_time, trim_margin
from delphi.env import DELPHI_DATA_DIR
from delphi.multimodal import Modality
from delphi.tokenizer import Tokenizer, update_tokenizer


@dataclass
class UKBDataConfig(BaseDataConfig):
    expansion_pack_dir: str = "ukb_real_data/expansion_packs"
    expansion_packs: list[str] = field(default_factory=list)
    biomarker_dir: str = "ukb_real_data/biomarkers"
    biomarkers: list[str] = field(default_factory=list)
    must_have_biomarkers: list[str] = field(default_factory=list)
    first_time_only: bool = True


class Biomarker:

    def __init__(self, path: str, memmap: bool = False, first_time_only: bool = True):
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
        self.first_time_only = first_time_only

    def __repr__(self):
        return f"Biomarker(path={self.path})"

    @property
    def data_participants(self):

        have_data = self.p2i.obj["seq_len"] > 0
        participants_with_data = self.p2i.obj.loc[have_data, "pid"].unique()

        return participants_with_data

    def get_batch(self, pids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        batch_data = []
        batch_time = []
        batch_count = []

        for pid in pids:
            pid_grp = self.p2i.get_group(pid)
            pid_time = pid_grp["time"].to_numpy()
            s = np.argsort(pid_time)
            pid_time = pid_time[s]
            pid_seq_len = pid_grp["seq_len"].to_numpy()[s]
            pid_start_pos = pid_grp["start_pos"].to_numpy()[s]
            for i, l in zip(pid_start_pos, pid_seq_len):
                if l > 0:
                    x = self.data[i : i + l]
                    batch_data.append(x)
                    batch_count.append(1)
                else:
                    batch_count.append(0)
                if self.first_time_only:
                    continue
            batch_time.append(pid_time)

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


class M4Dataset:

    def __init__(self, cfg: UKBDataConfig, memmap: bool = False):

        (
            base_tokenizer,
            self.start_pos,
            self.seq_len,
            self.participants,
            self.tokens,
            self.time_steps,
            self.rng,
        ) = load_core_data_package(cfg=cfg, memmap=memmap)

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
                base_tokenizer=base_tokenizer, add_tokenizer=add_tokenizer  # type: ignore
            )
            self.expansion_pack_tokenizers.append(add_tokenizer)
            self.expansion_packs.append(
                ExpansionPack(path=pack_path, offset=offset, memmap=memmap)
            )
        self.tokenizer = Tokenizer(base_tokenizer)  # type: ignore

        self.biomarker_dir = os.path.join(DELPHI_DATA_DIR, cfg.biomarker_dir)
        self.mod_ds = {}
        for modality in cfg.biomarkers:
            modality = Modality[modality.upper()]
            print(f"\t– loading biomarker: {modality.name}")
            biomarker_path = os.path.join(self.biomarker_dir, modality.name.lower())
            dataset = Biomarker(
                path=biomarker_path, memmap=memmap, first_time_only=cfg.first_time_only
            )
            self.mod_ds[modality] = dataset
        print(f"only using first occurrence of biomarkers: {cfg.first_time_only}")

        print(f"keeping participants with biomarkers: {cfg.must_have_biomarkers}")
        old_n = self.participants.size
        for modality in cfg.must_have_biomarkers:
            modality = Modality[modality.upper()]
            data_participants = self.mod_ds[modality].data_participants
            self.participants = self.participants[
                np.isin(self.participants, data_participants)
            ]
        print(f"{self.participants.size}/{old_n} remaining")

        self.transforms = load_transforms(cfg=cfg, tokenizer=self.tokenizer)

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

    def get_batch(self, batch_idx: np.ndarray):

        P = self.participants[batch_idx]
        X = []
        T = []
        for i, pid in enumerate(P):
            i = self.start_pos[pid]
            l = self.seq_len[pid]
            x_pid = self.tokens[i : i + l]
            t_pid = self.time_steps[i : i + l]
            for expansion_pack in self.expansion_packs:
                pack_x_pid, pack_t_pid = expansion_pack.get_expansion(pid)
                x_pid = np.concatenate((x_pid, pack_x_pid))
                t_pid = np.concatenate((t_pid, pack_t_pid))
            X.append(x_pid)
            T.append(t_pid)

        X = collate_batch_data(X)
        T = collate_batch_time(T)

        M = np.zeros_like(X, dtype=np.int8)
        M[X > 0] = 1
        biomarker_X = {}
        for modality, dataset in self.mod_ds.items():
            m_X, m_T, _ = dataset.get_batch(P)
            biomarker_X[modality] = m_X

            m_M = np.zeros_like(m_T, dtype=np.int8)
            m_M[m_T != -1e4] = modality.value

            T = np.concatenate((T, m_T), axis=1)
            M = np.concatenate((M, m_M), axis=1)
            X = np.concatenate((X, np.zeros_like(m_T, dtype=np.int8)), axis=1)

        for transform in self.transforms:
            X, T, M, biomarker_X = transform(
                X=X,
                T=T,
                M=M,
                biomarker_X=biomarker_X,
            )

        T, M, X = sort_by_time(T, M, X)
        M, T, X = trim_margin(M, T, X, trim_val=0)

        return X, T, M, biomarker_X


def biomarker_to_tensor(
    biomarker_data: dict[Modality, np.ndarray],
):

    for modality in biomarker_data.keys():
        m_X = biomarker_data[modality]
        biomarker_data[modality] = torch.from_numpy(m_X)  # type: ignore

    return biomarker_data


def load_sequences(
    it: Iterator,
    dataset: "M4Dataset",
) -> Iterator:

    for idx in it:

        X, T, M, biomarker_data = dataset.get_batch(idx)

        X = torch.tensor(X, dtype=torch.long)
        T = torch.tensor(T)
        M = torch.tensor(M, dtype=torch.long)
        biomarker_data = biomarker_to_tensor(biomarker_data=biomarker_data)

        yield X, T, M, biomarker_data


def load_prompt_sequences(
    it: Iterator, dataset: "M4Dataset", start_age: float, no_event_token: int
) -> Iterator:

    for idx in it:

        X, T, M, biomarker_data = dataset.get_batch(idx)

        too_old = T > start_age
        to_keep = ~too_old
        for modality, modality_data in biomarker_data.items():
            sub_idx, pos_idx = np.where(M == modality.value)
            biomarker_data[modality] = modality_data[to_keep[sub_idx, pos_idx], :]
        X[too_old] = 0
        T[too_old] = -1e4
        M[too_old] = 0

        pad = ((0, 0), (0, 1))
        X = np.pad(X, pad, "constant", constant_values=no_event_token)
        T = np.pad(T, pad, "constant", constant_values=start_age)
        M = np.pad(M, pad, "constant", constant_values=1)

        T, M, X = sort_by_time(T, M, X)
        M, T, X = trim_margin(M, T, X, trim_val=0)

        X = torch.tensor(X, dtype=torch.long)
        T = torch.tensor(T)
        M = torch.tensor(M, dtype=torch.long)
        biomarker_data = biomarker_to_tensor(biomarker_data=biomarker_data)

    yield X, T, M, biomarker_data


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
