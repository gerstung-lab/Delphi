import io
from dataclasses import dataclass, field

import lmdb
import numpy as np


@dataclass
class PRSConfig:
    lmdb_fname: str = "prs.lmdb"
    include: bool = False
    must: bool = False


class PRSDataset:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.env = lmdb.open(db_path, readonly=True, lock=False)

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
                    prs_scores = np.load(buffer)
                    batch_data.append(prs_scores)
                    batch_time.append(0)
                else:
                    batch_time.append(-1e4)

        batch_data = np.stack(batch_data)
        batch_time = np.expand_dims(np.array(batch_time), axis=1)

        return batch_data, batch_time


def get_from_lmdb(db_path, key):

    env = lmdb.open(db_path, readonly=True)

    with env.begin() as txn:
        key_bytes = str(key).encode("utf-8")
        value_bytes = txn.get(key_bytes)

        if value_bytes is not None:
            buffer = io.BytesIO(value_bytes)
            prs_scores = np.load(buffer)
            return prs_scores
        else:
            return None  # Return None if key is not found


# all_keys = get_all_keys(db_path)

# # Optionally decode the byte keys if they were stored as strings
# decoded_keys = [key.decode('utf-8') for key in all_keys]
# print("All keys:", decoded_keys)
