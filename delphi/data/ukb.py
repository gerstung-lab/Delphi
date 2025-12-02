import functools
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch
import yaml

from delphi.data.utils import (
    append_no_event,
    crop_contiguous,
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
        if bd.size > 0:
            if pad_left:
                collated_batch[i, -bd.size :] = bd
            else:
                collated_batch[i, : bd.size] = bd

    return collated_batch


def collate_batches(
    batch_data: list[np.ndarray], fill_value: int | float = 0, pad_left: bool = True
) -> np.ndarray:

    max_len = max([bd.shape[1] for bd in batch_data])
    n_lst = np.array([bd.shape[0] for bd in batch_data])
    collated_batch = np.full(
        shape=(n_lst.sum(), max_len),
        fill_value=fill_value,
        dtype=batch_data[0].dtype,
    )

    s = 0
    for i, bd in enumerate(batch_data):
        l = bd.shape[0]
        if pad_left:
            collated_batch[s : s + l, -bd.shape[1] :] = bd
        else:
            collated_batch[s : s + l, : bd.shape[1]] = bd
        s += l

    return collated_batch


LIFESTYLE = [
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


def cut_batch_for_prompt(idx, age, prompt_age, append_no_event):

    idx = idx.clone()
    age = age.clone()

    idx[age > prompt_age] = 0
    age[age > prompt_age] = -10000.0

    if prompt_age > 0 and append_no_event:
        idx = torch.nn.functional.pad(idx, (0, 1), "constant", 1)
        age = torch.nn.functional.pad(age, (0, 1), "constant", prompt_age)

    age_sort = age.argsort(1)
    idx = idx.gather(1, age_sort)
    age = age.gather(1, age_sort)

    trim_margin = torch.min(torch.sum(idx == 0, dim=1)).item()
    idx, age = idx[:, trim_margin:], age[:, trim_margin:]

    return idx, age


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
                exclude_list = LIFESTYLE
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
                perturb_list = LIFESTYLE
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

    def subset_participants_for_prompt(
        self, prompt_age: float, must_have_lifestyle: bool = False
    ):
        lifestyle_tokens = np.array(self.tokenizer[i] for i in LIFESTYLE)
        keep_lst = list()
        for i in range(self.participants.size):
            x_pid, t_pid = self[i]
            have_before = t_pid.min() <= prompt_age
            have_after = t_pid.max() >= prompt_age
            if must_have_lifestyle:
                have_lifestyle = np.isin(
                    x_pid[t_pid <= prompt_age], lifestyle_tokens
                ).any()
                if not have_lifestyle:
                    continue
            if have_before and have_after:
                keep_lst.append(i)
        print(
            f"prompt age {prompt_age}: {len(keep_lst)}/{self.participants.size} participants remaining"
        )
        self.participants = self.participants[keep_lst]

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

        p2i = pd.read_csv(os.path.join(path, "p2i.csv")).sort_values(by=["pid", "time"])
        self.start_pos = p2i["start_pos"].to_numpy()
        self.seq_len = p2i["seq_len"].to_numpy()
        self.time_steps = p2i["time"].to_numpy()
        self.pids, ct = np.unique(p2i["pid"].to_numpy(), return_counts=True)
        cumul_ct = np.insert(np.cumsum(ct), 0, 0, axis=0)
        self.pid2idx = dict(zip(self.pids, cumul_ct))
        self.pid2cnt = dict(zip(self.pids, ct))

        self.first_time_only = first_time_only

    def __repr__(self):
        return f"Biomarker(path={self.path})"

    @property
    def participants(self):

        have_data = np.array([self.pid2cnt[pid] > 0 for pid in self.pids])
        participants_with_data = self.pids[have_data]

        return participants_with_data

    def __getitem__(
        self, pid: int
    ) -> tuple[None | list[np.ndarray], None | np.ndarray]:

        pid_i = self.pid2idx[pid]
        pid_l = self.pid2cnt[pid]
        pid_slice = slice(pid_i, pid_i + pid_l)
        pid_time = self.time_steps[pid_slice]
        if (pid_time == -1e4).all():
            return None, None
        pid_seq_len = self.seq_len[pid_slice]
        pid_start_pos = self.start_pos[pid_slice]
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


def remove_biomarkers_after_time(biomarker, biomarker_t, m, final_time):
    drop_mask = biomarker_t >= final_time
    m_to_drop = m[drop_mask]
    if m_to_drop.size == 0:
        return biomarker, biomarker_t, m
    else:
        uniq_val, uniq_ct = np.unique(m_to_drop, return_counts=True)
        for m_val, m_ct in zip(uniq_val, uniq_ct):
            modality = Modality(m_val)
            del biomarker[modality][-m_ct:]
        biomarker_t = biomarker_t[~drop_mask]
        m = m[~drop_mask]
        return biomarker, biomarker_t, m


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
        perturb: bool = False,
        perturb_list: None | list = None,
        block_size: None | int = None,
        crop_mode: Literal["left", "right", "random"] = "left",
        seed: int = 42,
        memmap: bool = False,
    ):
        """
        note: the following defaults differ from UKBDataset
            - crop_mode
            - perturb
        """

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
                data_participants = self.mod_ds[modality].participants
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
                perturb_list = LIFESTYLE
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
                crop_contiguous,
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
        x_lst, t_lst = [x], [t]
        for expansion_pack in self.expansion_packs:
            exp_x, exp_t = expansion_pack[pid]
            x_lst.append(exp_x)
            t_lst.append(exp_t)
        x = np.concatenate(x_lst)
        t = np.concatenate(t_lst)
        x, t = self.append_no_event(x, t)
        x, t = self.perturb_time(x, t)
        t, x = sort_by_time(t, x)
        x, t = self.crop_block_size(x, t)

        bio_x_dict = dict()
        bio_t_lst = list()
        bio_m_lst = list()
        for modality, ds in self.mod_ds.items():
            bio_x, mod_t = ds[pid]
            if bio_x is None:
                continue
            bio_x_dict[modality] = bio_x
            mod_m = np.full_like(mod_t, fill_value=modality.value)
            bio_t_lst.append(mod_t)
            bio_m_lst.append(mod_m)

        if len(bio_x_dict) == 0:
            assert len(bio_t_lst) == 0
            assert len(bio_m_lst) == 0
            bio_t = np.array([])
            bio_m = np.array([])
        else:
            bio_t = np.concatenate(bio_t_lst)
            bio_m = np.concatenate(bio_m_lst)

            bio_t, bio_m = sort_by_time(bio_t, bio_m)
            bio_x_dict, bio_t, bio_m = remove_biomarkers_after_time(
                bio_x_dict, bio_t, bio_m, t.max()
            )

        return x, t, bio_x_dict, bio_t, bio_m

    def get_batch(self, batch_idx: Iterable, cut: bool = True):

        X, T, bio_M = list(), list(), list()
        bio_X = defaultdict(list)
        bio_T = list()
        for idx in batch_idx:
            x, t, bio_x, bio_t, m = self[idx]
            X.append(x)
            T.append(t)

            bio_T.append(bio_t)
            for modality in bio_x.keys():
                bio_X[modality].extend(bio_x[modality])
            bio_M.append(m)

        X = collate_batch(X)
        T = collate_batch(T, fill_value=-1e4)
        bio_M = collate_batch(bio_M)
        X = torch.tensor(X, dtype=torch.long)
        T = torch.tensor(T, dtype=torch.float32)
        bio_M = torch.tensor(bio_M, dtype=torch.long)

        bio_T = collate_batch(bio_T, fill_value=-1e4)
        bio_T = torch.tensor(bio_T, dtype=torch.float32)
        for modality, bio_x_lst in bio_X.items():
            bio_X[modality] = torch.from_numpy(np.stack(bio_x_lst))  # type: ignore

        return X[:, :-1], T[:, :-1], bio_M, bio_X, bio_T, X[:, 1:], T[:, 1:]


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
