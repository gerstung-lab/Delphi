import io
import os

import lmdb
import numpy as np


def estimate_size_by_first_record(
    keys: np.ndarray,
    values: np.ndarray,
    safety_margin: float = 0.5,
    metadata_size: int = 16,
) -> int:

    assert (
        keys.shape[0] == values.shape[0]
    ), "keys and values must have the same number of records"

    test_buffer = io.BytesIO()
    np.save(test_buffer, values[0])
    array_size = len(test_buffer.getvalue())
    record_size = len(str(int(keys[0])).encode("utf-8")) + array_size + metadata_size
    estimated_size = record_size * keys.shape[0]
    map_size = int(estimated_size * (1 + safety_margin))

    print(f"estimated database size: {estimated_size / (1024**3):.2f} GB")
    print(f"using map_size: {map_size / (1024**3):.2f} GB")

    return map_size


def get_all_pids(db_path: str) -> np.ndarray:

    assert os.path.exists(db_path), f"LMDB database not found at {db_path}"

    env = lmdb.open(db_path, readonly=True, lock=False)
    pids = []
    with env.begin() as txn:
        with txn.cursor() as cursor:
            for pid, _ in cursor:
                pids.append(int(pid.decode("utf-8")))

    return np.array(pids)
