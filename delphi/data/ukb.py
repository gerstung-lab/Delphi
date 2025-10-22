import functools
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch
import yaml

from delphi.data.transform import (
    append_no_event,
    crop_contiguous,
    crop_contiguous_multimodal,
    exclude_tokens,
    identity_transform,
    perturb_time,
    sort_by_time,
)
from delphi.env import DELPHI_DATA_DIR
from delphi.multimodal import Modality


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


lifestyle = [
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


class UKBDataset:

    def __init__(
        self,
        data_dir: str = "ukb_real_data",
        subject_list: str = "participants/train_fold.bin",
        no_event_interval: None | float = 5 * 365.25,
        no_event_mode: str = "legacy-random",
        block_size: None | int = None,
        perturb: bool = True,
        perturb_list: None | list = None,
        exclude: bool = False,
        exclude_list: None | list = None,
        crop_mode: Literal["left", "right", "random"] = "right",
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

        self.no_event_interval = no_event_interval
        if no_event_interval is not None:
            self.append_no_event = functools.partial(
                append_no_event,
                rng=self.rng,
                interval=no_event_interval,
                token=self.tokenizer["no_event"],
                mode=no_event_mode,
            )
        else:
            self.append_no_event = identity_transform

        if exclude:
            if exclude_list is None:
                exclude_list = lifestyle
            tokens_to_exclude = np.array(
                [self.tokenizer[event] for event in exclude_list]
            )
            self.exclude_tokens = functools.partial(
                exclude_tokens, blacklist=tokens_to_exclude
            )
        else:
            self.exclude_tokens = identity_transform

        if perturb:
            if perturb_list is None:
                perturb_list = lifestyle
            tokens_to_perturb = np.array(
                [self.tokenizer[event] for event in perturb_list]
            )
            self.perturb_time = functools.partial(
                perturb_time, tokens=tokens_to_perturb, rng=self.rng
            )
        else:
            self.perturb_time = identity_transform

        if block_size is not None:
            self.crop_block_size = functools.partial(
                crop_contiguous, block_size=block_size, rng=self.rng, mode=crop_mode
            )
        else:
            self.crop_block_size = identity_transform

    def __len__(self):
        return self.participants.size

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def __getitem__(self, idx: int):

        pid = self.participants[idx]
        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l].astype(np.uint32)
        t_pid = self.time_steps[i : i + l].astype(np.float32)
        x_pid, t_pid = self.exclude_tokens(x_pid, t_pid)
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
    def participants(self):

        have_data = self.p2i.obj["seq_len"] > 0
        participants_with_data = self.p2i.obj.loc[have_data, "pid"].unique()

        return participants_with_data

    def __getitem__(
        self, pid: int
    ) -> tuple[None | list[np.ndarray], None | np.ndarray]:

        pid_grp = self.p2i.get_group(pid)
        pid_time = pid_grp["time"].to_numpy()  # type: ignore
        if (pid_time == -1e4).all():
            return None, None
        s = np.argsort(pid_time)
        pid_time = pid_time[s]
        pid_seq_len = pid_grp["seq_len"].to_numpy()[s]  # type: ignore
        pid_start_pos = pid_grp["start_pos"].to_numpy()[s]  # type: ignore
        pid_data = list()
        for i, l in zip(pid_start_pos, pid_seq_len):
            if l > 0:
                x = self.data[i : i + l]
                pid_data.append(x)
            if self.first_time_only:
                pid_time = pid_time[[0]]
                break
        return pid_data, pid_time


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

        tokenizer_path = os.path.join(path, "tokenizer.yaml")
        with open(tokenizer_path, "r") as f:
            self.tokenizer = yaml.safe_load(f)

    def __getitem__(self, pid: int) -> tuple[np.ndarray, np.ndarray]:

        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l] + self.offset
        t_pid = self.time_steps[i : i + l]

        return x_pid, t_pid


def update_tokenizer(base_tokenizer: dict, add_tokenizer: dict) -> tuple[dict, int]:

    assert min(base_tokenizer.values()) == 0, "base tokenizer must start with 0"
    assert min(add_tokenizer.values()) == 1, "additional tokenizer must start with 1"
    offset = len(base_tokenizer) - 1
    for key, value in add_tokenizer.items():
        if key not in base_tokenizer:
            base_tokenizer[key] = value + offset
        else:
            raise ValueError(f"{key} already exists in base tokenizer")
    return base_tokenizer, offset


def pad_trailing_biomarkers(
    X: torch.Tensor,
    T: torch.Tensor,
    M: torch.Tensor,
):

    trailing_M = M[:, -1:]
    if trailing_M.max() > 1:
        M = torch.cat([M, trailing_M * 0], dim=1)
        X = torch.cat([X, torch.zeros_like(trailing_M, dtype=X.dtype)], dim=1)
        T = torch.cat([T, T[:, -1:]], dim=1)

    return X, T, M


class MultimodalUKBDataset:

    def __init__(
        self,
        data_dir: str = "ukb_real_data",
        expansion_pack_dir: str = "expansion_packs",
        expansion_packs: None | list = None,
        biomarker_dir: str = "biomarkers",
        biomarkers: None | list = None,
        first_time_only: bool = True,
        must_have_biomarkers: None | list = None,
        subject_list: str = "participants/train_fold.bin",
        no_event_interval: None | float = 5 * 365.25,
        no_event_mode: str = "legacy-random",
        perturb: bool = True,
        perturb_list: None | list = None,
        block_size: None | int = None,
        crop_mode: Literal["left", "right", "random"] = "right",
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

        self.expansion_packs = []
        self.expansion_pack_tokenizers = []
        if expansion_packs is not None:
            expansion_packs.sort()
            for pack in expansion_packs:
                pack_path = os.path.join(
                    DELPHI_DATA_DIR, data_dir, expansion_pack_dir, pack
                )
                assert os.path.exists(pack_path), FileNotFoundError(
                    f"expansion pack {pack_path} not found"
                )
                tokenizer_path = os.path.join(pack_path, "tokenizer.yaml")
                with open(tokenizer_path, "r") as f:
                    add_tokenizer = yaml.safe_load(f)

                self.tokenizer, offset = update_tokenizer(
                    base_tokenizer=self.tokenizer, add_tokenizer=add_tokenizer  # type: ignore
                )
                self.expansion_pack_tokenizers.append(add_tokenizer)
                self.expansion_packs.append(
                    ExpansionPack(path=pack_path, offset=offset, memmap=memmap)
                )

        self.biomarker_dir = os.path.join(DELPHI_DATA_DIR, data_dir, biomarker_dir)
        self.mod_ds = {}
        if biomarkers is not None:
            for modality in biomarkers:
                modality = Modality[modality.upper()]
                biomarker_path = os.path.join(self.biomarker_dir, modality.name.lower())
                dataset = Biomarker(
                    path=biomarker_path, memmap=memmap, first_time_only=first_time_only
                )
                self.mod_ds[modality] = dataset

        if must_have_biomarkers is not None:
            print(f"keeping participants with biomarkers: {must_have_biomarkers}")
            old_n = self.participants.size
            for modality in must_have_biomarkers:
                modality = Modality[modality.upper()]
                data_participants = self.mod_ds[modality].data_participants
                self.participants = self.participants[
                    np.isin(self.participants, data_participants)
                ]
            print(f"{self.participants.size}/{old_n} remaining")

        if no_event_interval is not None:
            self.append_no_event = functools.partial(
                append_no_event,
                rng=self.rng,
                interval=no_event_interval,
                token=self.tokenizer["no_event"],
                mode=no_event_mode,
            )
        else:
            self.append_no_event = identity_transform

        if perturb:
            if perturb_list is None:
                perturb_list = lifestyle
            tokens_to_perturb = np.array(
                [self.tokenizer[event] for event in perturb_list]
            )
            self.perturb_time = functools.partial(
                perturb_time, tokens=tokens_to_perturb, rng=self.rng
            )
        else:
            self.perturb_time = identity_transform

        if block_size is not None:
            self.crop_block_size = functools.partial(
                crop_contiguous_multimodal,
                block_size=block_size,
                rng=self.rng,
                mode=crop_mode,
            )
        else:
            self.crop_block_size = identity_transform

    def __len__(self):
        return self.participants.size

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def expansion_tokens(self):
        tokens = list()
        for exp_pack in self.expansion_packs:
            tokens.extend([v + exp_pack.offset for v in exp_pack.tokenizer.values()])
        return tokens

    def __getitem__(self, idx: int):

        pid = self.participants[idx]
        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x = self.tokens[i : i + l].astype(np.uint32)
        t = self.time_steps[i : i + l].astype(np.float32)
        for expansion_pack in self.expansion_packs:
            exp_x, exp_t = expansion_pack[pid]
            x = np.concatenate((x, exp_x))
            t = np.concatenate((t, exp_t))
        x, t = self.append_no_event(x, t)
        m = np.ones_like(x, dtype=np.int32)

        biomarker = dict()
        for modality, ds in self.mod_ds.items():
            bio_x, mod_t = ds[pid]
            if bio_x is None:
                continue
            biomarker[modality] = bio_x
            mod_m = np.full_like(mod_t, fill_value=modality.value)
            x = np.concatenate((x, np.zeros_like(mod_m)))
            t = np.concatenate((t, mod_t))
            m = np.concatenate((m, mod_m))

        x, t = self.perturb_time(x, t)
        t, x, m = sort_by_time(t, x, m)
        x, biomarker, t, m = self.crop_block_size(x, biomarker, t, m)
        return x, biomarker, t, m

    def get_batch(self, batch_idx: Iterable, cut: bool = True):

        X, T, M = list(), list(), list()
        biomarker_X = defaultdict(list)
        for idx in batch_idx:
            x, biomarker, t, m = self[idx]
            X.append(x)
            T.append(t)
            M.append(m)
            for modality, bio_x in biomarker.items():
                biomarker_X[modality].extend(bio_x)

        X = collate_batch(X)
        T = collate_batch(T, fill_value=-1e4)
        M = collate_batch(M)
        X = torch.tensor(X, dtype=torch.long)
        T = torch.tensor(T, dtype=torch.float32)
        M = torch.tensor(M, dtype=torch.long)

        for modality, bio_x_lst in biomarker_X.items():
            biomarker_X[modality] = torch.from_numpy(np.stack(bio_x_lst))  # type: ignore

        if cut:
            X, T, M = pad_trailing_biomarkers(X, T, M)
            return X[:, :-1], T[:, :-1], M[:, :-1], biomarker_X, X[:, 1:], T[:, 1:]
        else:
            return X, T, M, biomarker_X


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
