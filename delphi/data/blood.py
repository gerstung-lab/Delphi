import io
from dataclasses import dataclass

import lmdb
import numpy as np


@dataclass
class BloodConfig:
    lmdb_fname: str = "blood_mice.lmdb"
    include: bool = False
    must: bool = False


class BloodDataset:
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
