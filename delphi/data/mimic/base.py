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
        collated_batch[i, -bd.numel() :] = bd

    return collated_batch


class MIMICDataset:
    def __init__(
        self,
        input_dir: str | Path,
        n_positions: int = 2048,
        is_encoder_decoder: bool = False,
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

    def __getitem__(self, idx: int) -> tuple[th.Tensor | tuple, th.Tensor]:
        pt_ctx = self._get_patient_context(idx)
        end = min(idx + self.timeline_size + 1, self.patient_data_end_at_idx[idx])
        timeline = self.tokens[idx:end]

        if self.is_encoder_decoder:
            return (pt_ctx, timeline[:-1]), timeline[1:]

        x = th.cat((pt_ctx, timeline[:-1]))
        y = th.cat((pt_ctx, timeline[1:]))
        y[: self.context_size] = 0
        return x, y

    def get_batch(self, batch_idx: list[int]):

        X = []
        Y = []
        for idx in batch_idx:
            x, y = self[idx]
            X.append(x)
            Y.append(y)

        X = collate_batch_data(X)
        Y = collate_batch_data(Y)

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
