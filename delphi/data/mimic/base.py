import json
import pickle
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import torch as th

from delphi.tokenizer import Tokenizer

from ._sharded_data import ShardedData
from .constants import STATIC_DATA_FN, SpecialToken
from .vocabulary import Vocabulary


def collate_batch_data(batch_data: list[th.Tensor]) -> th.Tensor:

    max_len = max([bd.numel() for bd in batch_data])
    collated_batch = th.full(
        size=(len(batch_data), max_len),
        fill_value=0,
        dtype=batch_data[0].dtype,
    )
    for i, bd in enumerate(batch_data):
        collated_batch[i, : bd.numel()] = bd

    return collated_batch


def collate_batch_time(batch_time: list[th.Tensor]) -> th.Tensor:

    max_len = max([bd.numel() for bd in batch_time])
    collated_batch = th.full(
        size=(len(batch_time), max_len),
        fill_value=-1e4,
        dtype=batch_time[0].dtype,
    )
    for i, bd in enumerate(batch_time):
        collated_batch[i, : bd.numel()] = bd

    return collated_batch


class MIMICDataset:
    def __init__(
        self,
        input_dir: str | Path,
        n_positions: int = 2048,
        is_encoder_decoder: bool = False,
        sep_time_tokens: bool = True,
    ):
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        self._data = ShardedData(input_dir)

        self.vocab = Vocabulary.from_path(input_dir)
        self._num_quantiles = len(self.vocab.quantile_stokens)
        self.static_data = pickle.load((input_dir / STATIC_DATA_FN).open("rb"))

        # plus one, because DOB takes 2 spots
        self.context_size = len(next(iter(self.static_data.values()))) + 1
        self.timeline_size = n_positions - self.context_size

        self.is_encoder_decoder = is_encoder_decoder
        if is_encoder_decoder:
            self.timeline_size = n_positions

        with open(input_dir / "interval_estimates.json", "r") as f:
            interval_stats = json.load(f)
        time_intervals = list(interval_stats["min"].keys())
        self.time_tokens = th.tensor(self.vocab.encode(time_intervals))
        self.sep_time_tokens = sep_time_tokens

    @property
    def tokenizer(self):
        return Tokenizer(name2id=self.vocab.stoi)

    @property
    def tokens(self):
        return self._data.tokens

    @property
    def times(self):
        return self._data.times

    @property
    def patient_ids(self) -> th.Tensor:
        return th.cat([shard["patient_ids"] for shard in self._data.shards])

    @property
    def patient_id_at_idx(self):
        return self._data.patient_id_at_idx

    @property
    def patient_offsets(self) -> th.Tensor:
        return th.cat(
            [shard["patient_offsets"] + shard["offset"] for shard in self._data.shards]
        )

    @property
    def patient_offset_at_idx(self):
        """Aka patient data start at idx."""
        return self._data.patient_offset_at_idx

    @property
    def patient_data_end_at_idx(self):
        return self._data.patient_data_end_at_idx

    @property
    def is_mimic(self):
        return "hadm_id" in self._data.shards[0]

    @property
    def hadm_id(self):
        if not self.is_mimic:
            raise AttributeError("It's not MIMIC, no 'hadm_id' available.")
        return self._data.hadm_id

    @property
    def icu_stay_id(self):
        if not self.is_mimic:
            raise AttributeError("It's not MIMIC, no 'icustay_id' available.")
        return self._data.icu_stay_id

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(len={len(self):,}, "
            f"patient_num={len(self.patient_ids):,}, "
            f"vocab_size={len(self.vocab):,})"
        )

    def __len__(self) -> int:
        return len(self.tokens) - self.timeline_size

    @staticmethod
    def convert_time(timesteps: th.Tensor):
        return timesteps / 1e6 / 60

    def __getitem__(self, idx: int) -> tuple:
        pt_ctx = self._get_patient_context(idx)
        end = min(idx + self.timeline_size + 1, self.patient_data_end_at_idx[idx])  # type: ignore
        tokens = self.tokens[idx:end]
        timesteps = self.times[idx:end]
        timesteps = self.convert_time(timesteps)
        pt_ctx_timesteps = th.full(size=pt_ctx.shape, fill_value=timesteps[0].item())

        if self.sep_time_tokens:
            is_time_token = th.isin(tokens, self.time_tokens)
            tokens = tokens[~is_time_token]
            timesteps = timesteps[~is_time_token]

        if self.is_encoder_decoder:
            return (pt_ctx, tokens[:-1]), tokens[1:]

        x = th.cat((pt_ctx, tokens[:-1]))
        x_t = th.cat((pt_ctx_timesteps, timesteps[:-1]))
        y = th.cat((pt_ctx, tokens[1:]))
        y_t = th.cat((pt_ctx_timesteps, timesteps[1:]))
        y[: self.context_size] = 0

        return x, x_t, y, y_t

    def get_batch(self, batch_idx: list[int]):

        X, X_t, Y, Y_t = [], [], [], []
        for idx in batch_idx:
            x, x_t, y, y_t = self[idx]
            X.append(x)
            X_t.append(x_t)
            Y.append(y)
            Y_t.append(y_t)

        X = collate_batch_data(X)
        X_t = collate_batch_time(X_t)
        Y = collate_batch_data(Y)
        Y_t = collate_batch_time(Y_t)

        if self.sep_time_tokens:
            return X, X_t, Y, Y_t
        else:
            return X, Y

    def _get_patient_context(self, idx: int) -> th.Tensor:
        patient_id = self.patient_id_at_idx[idx].item()
        time_at_start = self.times[idx].item()

        static_tokens = []
        for token in self.static_data[patient_id].values():
            if token["code"][0] == SpecialToken.DOB:
                age = timedelta(microseconds=time_at_start - token["time"][0])
                static_tokens.extend(self._age_to_tokens(age.days / 365.25))
            elif len(token["code"]) == 1:
                static_tokens.append(token["code"][0])
            else:
                time_idx = self._find_idx_of_last_smaller_or_equal(
                    token["time"], time_at_start
                )
                code = (
                    token["code"][0].split("//")[0] + "//UNKNOWN"
                    if time_idx == -1
                    else token["code"][time_idx]
                )
                static_tokens.append(code)
        return th.tensor(self.vocab.encode(static_tokens))

    def _age_to_tokens(self, age_years: float) -> tuple[str, str]:
        age_scaled = age_years * self._num_quantiles**2 / 100
        age_scaled = min(age_scaled, self._num_quantiles**2 - 1)

        age_t1 = int(age_scaled // self._num_quantiles)
        age_t2 = round(age_scaled % self._num_quantiles)
        if age_t2 == self._num_quantiles:
            age_t1 += 1
            age_t2 = 0

        return f"Q{age_t1 + 1}", f"Q{age_t2 + 1}"

    @staticmethod
    def _find_idx_of_last_smaller_or_equal(ll: Sequence, value: float) -> int:
        """Assumes ll is sorted in ascending order."""
        indices = [i for i, v in enumerate(ll) if v <= value]
        if indices:
            return indices[-1]
        return -1
