import io
from dataclasses import dataclass

import lmdb
import numpy as np
import yaml


@dataclass
class FamilyHxConfig:
    lmdb_fname: str = "family_hx.lmdb"
    map_yaml: str = "config/familly_hx/map.yaml"
    include: bool = False
    must: bool = False


class FamilyHxDataset:
    def __init__(self, db_path: str, map_yaml: str):

        self.db_path = db_path
        self.env = lmdb.open(db_path, readonly=True, lock=False)

        with open(map_yaml, "r") as f:
            map_config = yaml.safe_load(f)
        self.lookup_max = max(map_config.keys())
        self.lookup_min = min(map_config.keys())
        lookup_size = self.lookup_max - self.lookup_min + 1
        self.lookup = np.zeros((lookup_size,), dtype=np.int32)
        for key, value in map_config.items():
            self.lookup[int(key) - self.lookup_min] = int(value)

    def get_all_pids(self):

        pids = []

        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                for pid, _ in cursor:
                    pids.append(int(pid.decode("utf-8")))

        return pids

    def get_raw_batch(self, pids):

        with self.env.begin() as txn:
            batch_data = []
            batch_time = []
            for pid in pids:
                pid_bytes = str(pid).encode("utf-8")
                value_bytes = txn.get(pid_bytes)
                if value_bytes is not None:
                    buffer = io.BytesIO(value_bytes)
                    hx_fam = np.load(buffer)
                    hx = hx_fam[0, :].astype(int)
                    hx = self.lookup[hx - self.lookup_min]
                    hx = np.unique(hx)
                    if hx.sum() > 0:
                        hx = hx[hx > 0]
                    batch_data.extend(hx)
                    batch_time.append(np.zeros_like(hx))
                else:
                    batch_time.append(np.array([-1e4]))

        batch_data = np.array(batch_data)
        batch_time = pack_family_hx_time(batch_time)

        return batch_data, batch_time


def pack_family_hx_time(batch_time: list[np.ndarray]):
    """
    pack a list of family hx time into a 2D array of shape (batch_size, max_hx_len)

    args:
        batch_time: list of family hx time, each element is a 1D array of shape (hx_len,)
    returns:
        batch_time_padded: 2D array of shape (batch_size, max_hx_len),
            where max_hx_len is the maximum length of hx in the batch
            and -1e4 is used to pad the empty space
    """

    hx_len = [len(bt) for bt in batch_time]
    max_hx_len = max(hx_len)

    batch_time_padded = np.full(
        shape=(len(batch_time), max_hx_len), fill_value=-1e4, dtype=np.float32
    )
    for i, bt in enumerate(batch_time):
        batch_time_padded[i, : len(bt)] = bt

    return batch_time_padded
