from dataclasses import dataclass
import io
import numpy as np
import lmdb


@dataclass
class FamilyHxConfig:
    lmdb_fname: str = "family_hx.lmdb"
    include: bool = False
    must: bool = False


class FamilyHxDataset:
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
                    hx_fam = np.load(buffer)
                    hx = np.unique(hx_fam[0, :])
                    batch_data.append(hx)
                    batch_time.append(np.zeros_like(hx))
                else:
                    batch_time.append(np.array([-1e4]))

        batch_data = np.stack(batch_data)
        batch_time = np.expand_dims(np.array(batch_time), axis=1)

        return batch_data, batch_time


def pack_family_hx(
    batch_data: list[np.array],
    batch_time: list[np.array]
):

    hx_len = [len(hx) for hx in batch_data]
    max_hx_len = max(max_len)
    

    pass
