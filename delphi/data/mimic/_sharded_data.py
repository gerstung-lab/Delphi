import abc
from bisect import bisect_right
from collections.abc import Callable, Iterable
from pathlib import Path

import torch as th
from safetensors import safe_open


class ShardedData:
    def __init__(self, data_fp: Path):
        shard_fps = sorted(data_fp.glob("[0-9]*.safetensors"))
        if not shard_fps:
            raise FileNotFoundError(
                f"No files matching '[0-9]*.safetensors' found in: {data_fp}"
            )

        self.shards = []
        shard_offset = 0
        for shard_fp in shard_fps:
            with safe_open(shard_fp, framework="pt") as f:
                self.shards.append(
                    {
                        "tokens": f.get_slice("tokens"),
                        "times": f.get_slice("times"),
                        "offset": shard_offset,
                        # TODO: should these two be slices as well?
                        "patient_ids": f.get_tensor("patient_ids"),
                        "patient_offsets": f.get_tensor("patient_offsets"),
                    }
                )
                for mimic_col in ["hadm_id", "icustay_id", "dicom_id"]:
                    if mimic_col in f.keys():
                        self.shards[-1][mimic_col] = f.get_slice(mimic_col)
            shard_offset += self.shards[-1]["tokens"].get_shape()[0]

    @property
    def tokens(self) -> "SliceableData":
        return SliceableData("tokens", self.shards)

    @property
    def times(self) -> "SliceableData":
        return SliceableData("times", self.shards)

    @property
    def patient_id_at_idx(self) -> "LookupData":
        def func(shard, shard_idx):
            pt_idx = (
                th.searchsorted(shard["patient_offsets"], shard_idx, right=True) - 1
            )
            return shard["patient_ids"][pt_idx]

        return LookupData("patient_ids", self.shards, access_func=func)

    @property
    def patient_offset_at_idx(self) -> "LookupData":
        def func(shard, shard_idx):
            pt_idx = (
                th.searchsorted(shard["patient_offsets"], shard_idx, right=True) - 1
            )
            return shard["offset"] + shard["patient_offsets"][pt_idx]

        return LookupData("patient_offsets", self.shards, access_func=func)

    @property
    def patient_data_end_at_idx(self) -> "LookupData":
        def func(shard, shard_idx):
            pt_idx = th.searchsorted(shard["patient_offsets"], shard_idx, right=True)
            if pt_idx == len(shard["patient_offsets"]):
                data_idx = shard["tokens"].get_shape()[0]
            else:
                data_idx = shard["patient_offsets"][pt_idx]

            return shard["offset"] + data_idx

        return LookupData("patient_offsets", self.shards, access_func=func)

    @property
    def hadm_id(self) -> "LookupData":
        return LookupData("hadm_id", self.shards)

    @property
    def icu_stay_id(self) -> "LookupData":
        return LookupData("icustay_id", self.shards)

    @property
    def dicom_id(self) -> "LookupData":
        return LookupData("dicom_id", self.shards)


class _DataBase(abc.ABC):
    def __init__(self, data_name: str, shards: list[dict]):
        self.data_name = data_name
        self.shards = shards

    def _get_shard_no_and_idx(self, g_idx: int) -> tuple[th.Tensor | int, th.Tensor]:
        shard_no = (
            bisect_right(self.shards, g_idx, key=lambda shard: shard["offset"]) - 1
        )
        return shard_no, g_idx - self.shards[shard_no]["offset"]


class SliceableData(_DataBase):
    def __getitem__(self, idx: int | Iterable[int] | slice) -> th.Tensor:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError(f"Step is not supported, got: {step} != 1")
            requested_length = stop - start
        elif isinstance(idx, Iterable) and not (
            isinstance(idx, th.Tensor) and idx.ndim == 0
        ):
            raise NotImplementedError("Indexing with an iterable is not implemented")
        else:
            start = idx
            requested_length = None

        shard_no, shard_idx = self._get_shard_no_and_idx(start)
        if requested_length is not None:
            shard_idx = slice(shard_idx, shard_idx + requested_length)
        value = self.shards[shard_no][self.data_name][shard_idx]

        # get the remaining values from the next shard if the requested length is not satisfied
        # assert requested_length is not None
        if value.ndim != 0 and len(value) < requested_length:
            next_chunk = self.shards[shard_no + 1][self.data_name]
            value = th.cat((value, next_chunk[: requested_length - len(value)]))

        return value

    def __len__(self) -> int:
        return sum(shard[self.data_name].get_shape()[0] for shard in self.shards)

    def __iter__(self):
        for shard in self.shards:
            yield shard[self.data_name][:]


class LookupData(_DataBase):
    def __init__(self, data_name: str, shards, access_func: Callable | None = None):
        super().__init__(data_name, shards)

        access_func = access_func or self.direct_access

        def access_func_wrapper(idx: int) -> th.Tensor:
            shard_no, shard_idx = self._get_shard_no_and_idx(idx)
            return access_func(shards[shard_no], shard_idx)

        self.access_func = access_func_wrapper

    def __len__(self):
        return sum(len(shard[self.data_name]) for shard in self.shards)

    def __getitem__(self, idx: int | Iterable[int]) -> th.Tensor:
        if isinstance(idx, Iterable) and not (
            isinstance(idx, th.Tensor) and idx.ndim == 0
        ):
            return th.stack([self.access_func(i) for i in idx])
        return self.access_func(idx)

    def direct_access(self, shard: dict, shard_idx: th.Tensor) -> th.Tensor:
        return shard[self.data_name][shard_idx]
