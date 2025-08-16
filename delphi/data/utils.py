from typing import Iterable, Iterator

import numpy as np
import torch


def move_batch_to_device(args: Iterable, device: str):

    return tuple([arg.to(device) for arg in args])


def eval_iter(total_size: int, batch_size: int) -> Iterator[np.ndarray]:

    batch_start_pos = np.arange(0, total_size, batch_size)
    batch_end_pos = batch_start_pos + batch_size
    batch_end_pos[-1] = total_size

    for start, end in zip(batch_start_pos, batch_end_pos):
        yield np.arange(start, end)


def train_iter(
    seed: int,
    total_size: int,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    step: int = 0,
) -> Iterator[np.ndarray]:

    while True:
        seed_with_offset = seed + step * world_size + rank
        rng = np.random.default_rng(seed_with_offset)
        batch_idx = rng.integers(total_size, size=(batch_size,))
        step += 1

        yield batch_idx


def duplicate_participants(args: Iterable[torch.Tensor], n_repeat: int):

    return tuple(
        [torch.repeat_interleave(arg, repeats=n_repeat, dim=0) for arg in args]
    )
