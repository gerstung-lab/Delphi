import io
import os
from typing import Optional, Union

import lmdb
import numpy as np


def estimate_write_size(
    db: dict[bytes, np.ndarray],
    safety_margin: float = 0.5,
    metadata_size: int = 16,
    n_samples: Optional[int] = None,
) -> int:

    total_entries = len(db)

    if n_samples is None or n_samples >= total_entries:
        sample_keys = list(db.keys())
        sample_multiplier = 1
    else:
        import random

        sample_keys = random.sample(list(db.keys()), n_samples)
        sample_multiplier = total_entries / n_samples

    sampled_size = 0
    for key in sample_keys:
        assert isinstance(key, bytes), "keys must be of type bytes"
        val = db[key]
        key_size = len(key)
        buffer = io.BytesIO()
        np.save(buffer, val)
        value_size = len(buffer.getvalue())
        sampled_size += key_size + value_size + metadata_size

    estimated_size = int(sampled_size * sample_multiplier)
    map_size = int(estimated_size * (1 + safety_margin))

    print(f"estimated map_size: {map_size / (1024**3):.2f} GB")
    return map_size


def write_lmdb(
    db: dict[bytes, np.ndarray],
    db_path: str,
    map_size: int,
) -> None:

    if os.path.exists(db_path):
        import shutil

        shutil.rmtree(db_path)
        print(f"deleted existing lmdb directory: {db_path}")

    env = lmdb.open(db_path, map_size=map_size)
    with env.begin(write=True) as txn:
        for key, val in db.items():
            assert isinstance(key, bytes), "keys must be of type bytes"
            buffer = io.BytesIO()
            np.save(buffer, val)
            value_bytes = buffer.getvalue()
            txn.put(key, value_bytes)
    env.close()
    print(f"lmdb written to {db_path}")


def data_key(pid: Union[int, str]) -> bytes:
    return f"{pid}.data".encode("utf-8")


def time_key(pid: Union[int, str]) -> bytes:
    return f"{pid}.time".encode("utf-8")


def get_all_pids(db_path: str) -> np.ndarray:

    assert os.path.exists(db_path), f"LMDB database not found at {db_path}"

    env = lmdb.open(db_path, readonly=True, lock=False)
    pids = []
    with env.begin() as txn:
        with txn.cursor() as cursor:
            for pid, _ in cursor:
                pids.append(int(pid.decode("utf-8").split(".")[0]))

    return np.unique(np.array(pids))


def get_token_count(db_path: str) -> int:

    assert os.path.exists(db_path), f"LMDB database not found at {db_path}"

    env = lmdb.open(db_path, readonly=True, lock=False)
    token_count = 0
    with env.begin() as txn:
        with txn.cursor() as cursor:
            for _, value in cursor:
                buffer = io.BytesIO(value)
                data = np.load(buffer)
                token_count += data.size

    return token_count


class BiomarkerLMDB:
    def __init__(self, lmdb_path: str, n_token: Optional[int] = None) -> None:

        assert os.path.exists(lmdb_path), f"lmdb {lmdb_path} does not exist."
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.n_token = n_token

    def get_all_pids(self) -> np.ndarray:
        return get_all_pids(self.env.path())

    def get_raw_batch(
        self, pids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self.env.begin() as txn:
            batch_data = []
            batch_time = []
            batch_count = []
            for pid in pids:
                data_bytes = txn.get(data_key(pid))
                time_bytes = txn.get(time_key(pid))
                # if data_bytes is not None and time_bytes is not None:
                biomarker_data = np.load(io.BytesIO(data_bytes))
                batch_data.append(biomarker_data)
                if biomarker_data.size > 0:
                    n_token = (
                        biomarker_data.shape[1]
                        if self.n_token is None
                        else self.n_token
                    )
                    biomarker_time = np.load(io.BytesIO(time_bytes))
                    biomarker_time = np.repeat(biomarker_time, n_token)
                else:
                    biomarker_time = np.array([-1e4])
                batch_time.append(biomarker_time)
                batch_count.append(biomarker_data.shape[0])
                # else:
                #     batch_time.append(np.array([-1e4]))
                #     batch_count.append(0)

        batch_data = collate_batch_data(batch_data)
        batch_time = collate_batch_time(batch_time)
        batch_count = np.array(batch_count, dtype=np.int32)

        return batch_data, batch_time, batch_count


def collate_batch_data(
    batch_data: list[np.ndarray],
) -> np.ndarray:

    n_measurement = [data.shape[0] for data in batch_data]
    n_feature = [data.shape[1] for data in batch_data]
    max_n_feature = max(n_feature) if n_feature else 0
    collated_batch = np.full(
        shape=(sum(n_measurement), max_n_feature),
        fill_value=0,
        dtype=batch_data[0].dtype,
    )
    for i, data in enumerate(batch_data):
        start_idx = sum(n_measurement[:i])
        end_idx = start_idx + n_measurement[i]
        collated_batch[start_idx:end_idx, : data.shape[1]] = data

    return collated_batch


def collate_batch_time(batch_time: list[np.ndarray]) -> np.ndarray:
    """
    collate batch time into a single array with padding
    """
    seq_len = [len(bt) for bt in batch_time]
    collated_batch = np.full(
        shape=(len(batch_time), max(seq_len)), fill_value=-1e4, dtype=np.float32
    )
    for i, bt in enumerate(batch_time):
        collated_batch[i, : len(bt)] = bt

    return collated_batch
