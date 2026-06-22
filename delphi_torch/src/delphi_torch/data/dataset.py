from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def build_p2i(data: np.ndarray) -> np.ndarray:
    px = data[:, 0].astype("int")
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
        if i == len(px) - 1:
            p2i.append([j, i - j + 1])
    return np.array(p2i)


class TrajectoryDataset(Dataset[int]):
    def __init__(self, data_path: Path, *, data_fraction: float = 1.0) -> None:
        self.data = np.memmap(data_path, dtype=np.uint32, mode="r").reshape(-1, 3)
        self.p2i = build_p2i(self.data)
        if data_fraction < 1.0:
            keep = int(len(self.p2i) * data_fraction)
            self.p2i = self.p2i[:keep]

    def __len__(self) -> int:
        return len(self.p2i)

    def __getitem__(self, index: int) -> int:
        return int(index)
