from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from delphi_torch.config import DataConfig
from delphi_torch.data.batch import make_batch
from delphi_torch.data.dataset import TrajectoryDataset


Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class DelphiCollator:
    data: object
    p2i: object
    cfg: DataConfig

    def __call__(self, batch: list[int]) -> Batch:
        ix = torch.tensor(batch, dtype=torch.int64)
        return make_batch(
            ix,
            self.data,
            self.p2i,
            block_size=self.cfg.block_size,
            select=self.cfg.select,
            padding=self.cfg.padding,
            lifestyle_augmentations=self.cfg.lifestyle_augmentations,
            no_event_token_rate=self.cfg.no_event_token_rate,
            cut_batch=True,
            device="cpu",
        )


def build_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    data_root = Path(cfg.data_dir) / cfg.dataset
    train_ds = TrajectoryDataset(data_root / "train.bin", data_fraction=cfg.data_fraction)
    val_ds = TrajectoryDataset(data_root / "val.bin", data_fraction=1.0)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=DelphiCollator(train_ds.data, train_ds.p2i, cfg),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=DelphiCollator(val_ds.data, val_ds.p2i, cfg),
        drop_last=False,
    )

    return train_loader, val_loader
