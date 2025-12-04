import functools
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import shap
import torch
import yaml

from delphi.data.utils import (
    append_no_event,
    collate_batch,
    crop_contiguous,
    exclude_tokens,
    identity_transform,
    perturb_time,
    remove_biomarkers_after_time,
    sort_by_time,
    update_tokenizer,
)
from delphi.env import DELPHI_DATA_DIR
from delphi.multimodal import Modality

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
        deterministic: bool = False,
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

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.deterministic = deterministic

        self.no_event_interval = no_event_interval
        if no_event_interval is not None:
            self.append_no_event = functools.partial(
                append_no_event,
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
                perturb_time,
                tokens=tokens_to_perturb,
            )
        else:
            self.perturb_time = identity_transform

        if block_size is not None:
            self.crop_block_size = functools.partial(
                crop_contiguous,
                block_size=block_size,
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
    def detokenizer(self):
        return {v: k for k, v in self.tokenizer.items()}

    def __getitem__(self, idx: int):

        pid = self.participants[idx]
        if self.deterministic:
            rng = np.random.default_rng(pid + self.seed)
        else:
            rng = self.rng
        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l].astype(np.uint32)
        t_pid = self.time_steps[i : i + l].astype(np.float32)
        x_pid, t_pid = self.exclude_tokens(x_pid, t_pid)
        x_pid, t_pid = self.append_no_event(x_pid, t_pid, rng=rng)
        x_pid, t_pid = self.perturb_time(x_pid, t_pid, rng=rng)
        t_pid, x_pid = sort_by_time(t_pid, x_pid, stable=self.deterministic)
        x_pid, t_pid = self.crop_block_size(x_pid, t_pid, rng=rng)

        return x_pid[:-1], t_pid[:-1], x_pid[1:], t_pid[1:]

    def subset_participants_for_prompt(
        self, prompt_age: float, must_have_lifestyle: bool = False
    ):
        lifestyle_tokens = np.array(self.tokenizer[i] for i in LIFESTYLE)
        keep_lst = list()
        for i in range(self.participants.size):
            x_pid, t_pid, _, _ = self[i]
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

    def get_batch(self, batch_idx: Iterable):

        X0, T0, X1, T1 = list(), list(), list(), list()
        for idx in batch_idx:
            x0, t0, x1, t1 = self[idx]
            X0.append(x0)
            X1.append(x1)
            T0.append(t0)
            T1.append(t1)

        X0 = collate_batch(X0)
        T0 = collate_batch(T0, fill_value=-1e4)
        X1 = collate_batch(X1)
        T1 = collate_batch(T1, fill_value=-1e4)

        X0 = torch.tensor(X0, dtype=torch.long)
        T0 = torch.tensor(T0, dtype=torch.float32)
        X1 = torch.tensor(X1, dtype=torch.long)
        T1 = torch.tensor(T1, dtype=torch.float32)

        return X0, T0, X1, T1


class Biomarker:

    def __init__(
        self,
        path: str,
        stats_subjects: None | np.ndarray = None,
        memmap: bool = False,
        first_time_only: bool = True,
        z_score: bool = False,
    ):

        self.path = path
        data_path = os.path.join(path, "data.bin")
        if memmap:
            self.data = np.memmap(data_path, dtype=np.float32, mode="r")
        else:
            self.data = np.fromfile(data_path, dtype=np.float32)

        with open(os.path.join(path, "features.yaml"), "r") as f:
            self.features = yaml.safe_load(f)
        self.n_features = len(self.features)

        p2i = pd.read_csv(os.path.join(path, "p2i.csv")).sort_values(by=["pid", "time"])
        self.start_pos = p2i["start_pos"].to_numpy()
        self.seq_len = p2i["seq_len"].to_numpy()
        self.time_steps = p2i["time"].to_numpy()
        self.pids = p2i["pid"].to_numpy()
        self.uniq_pids, ct = np.unique(p2i["pid"].to_numpy(), return_counts=True)
        cumul_ct = np.insert(np.cumsum(ct), 0, 0, axis=0)
        self.pid2idx = dict(zip(self.uniq_pids, cumul_ct))
        self.pid2cnt = dict(zip(self.uniq_pids, ct))

        self.first_time_only = first_time_only
        self.z_score = z_score
        if stats_subjects is None:
            stats_subjects = self.uniq_pids
        self.mean, self.std = self.stats(stats_subjects)

    def __repr__(self):
        return f"Biomarker(path={self.path}, n_features={self.n_features})"

    @property
    def mask(self):
        if self.z_score:
            if self.n_features > 1:
                return np.zeros((self.n_features,))
            else:
                return 0
        else:
            return self.mean

    def stats(self, subjects: np.ndarray):
        data = list()
        start_pos = self.start_pos[np.isin(self.pids, subjects)]
        seq_len = self.seq_len[np.isin(self.pids, subjects)]
        for i, l in zip(start_pos, seq_len):
            pid_data = self.data[i : i + l]
            if pid_data.size > 0:
                data.append(pid_data)
        data = np.stack(data, axis=0)
        if data.shape[1] > 1:
            return np.mean(data, axis=0), np.std(data, axis=0)
        else:
            return np.mean(data), np.std(data)

    def transform(self, x):
        if self.z_score:
            return (x - self.mean) / self.std
        else:
            return x

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
                x = self.transform(x)
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


class MultimodalUKBDataset:

    def __init__(
        self,
        data_dir: str = "ukb_real_data",
        expansion_pack_dir: str = "expansion_packs",
        expansion_packs: None | list = None,
        biomarker_datasets: None | dict = None,
        biomarker_dir: str = "biomarkers",
        biomarkers: None | list = None,
        z_score_biomarkers: bool = False,
        first_time_only: bool = True,
        must_have_biomarkers: None | list = None,
        stats_subject_list: None | str = None,
        subject_list: str = "participants/train_fold.bin",
        no_event_interval: None | float = 5 * 365.25,
        no_event_mode: str = "legacy-random",
        perturb: bool = False,
        perturb_list: None | list = None,
        block_size: None | int = None,
        crop_mode: Literal["left", "right", "random"] = "left",
        seed: int = 42,
        deterministic: bool = False,
        memmap: bool = False,
    ):
        """
        args:
        - biomarker_datasets(optional): a list of pre-initialized Biomarkers
        - stats_subject_list(optional): path to an array of subjects for compputing Biomarker stats
        note: the following defaults differ from UKBDataset
            - crop_mode
            - perturb
        """

        self._init_args = locals().copy()
        self._init_args.pop("self")  # Remove 'self' reference

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

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.deterministic = deterministic

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

        if biomarker_datasets is not None:
            assert biomarkers is None
            self.mod_ds = biomarker_datasets
        else:
            biomarker_dir = os.path.join(DELPHI_DATA_DIR, data_dir, biomarker_dir)
            if stats_subject_list is None:
                stats_subjects = self.participants
            else:
                stats_subjects = np.fromfile(
                    os.path.join(DELPHI_DATA_DIR, data_dir, stats_subject_list),
                    dtype=np.uint32,
                )

            self.mod_ds = {}
            if biomarkers is not None:
                for modality in biomarkers:
                    modality = Modality[modality.upper()]
                    biomarker_path = os.path.join(biomarker_dir, modality.name.lower())
                    dataset = Biomarker(
                        path=biomarker_path,
                        stats_subjects=stats_subjects,
                        memmap=memmap,
                        first_time_only=first_time_only,
                        z_score=z_score_biomarkers,
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
                perturb_time, tokens=tokens_to_perturb
            )
        else:
            self.perturb_time = identity_transform

        if block_size is not None:
            self.crop_block_size = functools.partial(
                crop_contiguous,
                block_size=block_size,
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
    def detokenizer(self):
        return {v: k for k, v in self.tokenizer.items()}

    @property
    def expansion_tokens(self):
        tokens = list()
        for exp_pack in self.expansion_packs:
            tokens.extend([v + exp_pack.offset for v in exp_pack.tokenizer.values()])
        return tokens

    def __getitem__(self, idx: int):

        pid = self.participants[idx]
        if self.deterministic:
            rng = np.random.default_rng(pid + self.seed)
        else:
            rng = self.rng
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
        x, t = self.append_no_event(x, t, rng=rng)
        x, t = self.perturb_time(x, t, rng=rng)
        t, x = sort_by_time(t, x)
        x, t = self.crop_block_size(x, t, rng=rng)

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

        x0, x1 = x[:-1].copy(), x[1:].copy()
        t0, t1 = t[:-1].copy(), t[1:].copy()

        return x0, t0, bio_x_dict, bio_t, bio_m, x1, t1

    def get_batch(self, batch_idx: Iterable):

        X0, T0, X1, T1 = list(), list(), list(), list()
        bio_X_dict, bio_T, bio_M = defaultdict(list), list(), list()
        for idx in batch_idx:
            x0, t0, bio_x_dict, bio_t, bio_m, x1, t1 = self[idx]
            X0.append(x0)
            T0.append(t0)
            X1.append(x1)
            T1.append(t1)

            for modality in bio_x_dict.keys():
                bio_X_dict[modality].extend(bio_x_dict[modality])
            bio_T.append(bio_t)
            bio_M.append(bio_m)

        X0 = collate_batch(X0)
        X0 = torch.tensor(X0, dtype=torch.long)
        T0 = collate_batch(T0, fill_value=-1e4)
        T0 = torch.tensor(T0, dtype=torch.float32)
        X1 = collate_batch(X1)
        X1 = torch.tensor(X1, dtype=torch.long)
        T1 = collate_batch(T1, fill_value=-1e4)
        T1 = torch.tensor(T1, dtype=torch.float32)

        for modality, bio_x_lst in bio_X_dict.items():
            bio_X_dict[modality] = torch.from_numpy(np.stack(bio_x_lst))  # type: ignore
        bio_T = collate_batch(bio_T, fill_value=-1e4)
        bio_T = torch.tensor(bio_T, dtype=torch.float32)
        bio_M = collate_batch(bio_M)
        bio_M = torch.tensor(bio_M, dtype=torch.long)

        return X0, T0, bio_X_dict, bio_T, bio_M, X1, T1


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
