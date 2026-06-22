from __future__ import annotations

import numpy as np
import torch


def make_batch(
    ix: torch.Tensor,
    data: np.ndarray,
    p2i: np.ndarray,
    *,
    block_size: int,
    select: str = "left",
    index: str = "patient",
    padding: str | None = "regular",
    lifestyle_augmentations: bool = False,
    no_event_token_rate: int = 5,
    cut_batch: bool = False,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a batch from patient indices.

    Returns:
        x: input tokens
        a: input ages
        y: target tokens
        b: target ages
    """
    mask_time = -10000.0

    if not isinstance(ix, torch.Tensor):
        ix = torch.tensor(np.array(ix), dtype=torch.int64)

    x = torch.tensor(np.array([p2i[int(i)] for i in ix]))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(ix.sum().item())

    if index == "patient":
        if select == "left":
            traj_start_idx = x[:, 0]
        elif select == "right":
            traj_start_idx = torch.clamp(x[:, 0] + x[:, 1] - block_size - 1, 0, data.shape[0])
        elif select == "random":
            traj_start_idx = x[:, 0] + (
                torch.randint(2**63 - 1, (len(ix),), generator=gen)
                % torch.clamp(x[:, 1] - block_size, 1)
            )
            traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1)
    traj_start_idx = traj_start_idx.numpy()

    batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    mask = torch.from_numpy(data[:, 0][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(data[p2i[ix.numpy()][:, 0], 0][:, None].astype(np.int64)).to(mask.dtype)

    tokens = torch.from_numpy(data[:, 2][batch_idx].astype(np.int64))
    ages = torch.from_numpy(data[:, 1][batch_idx].astype(np.float32))

    if lifestyle_augmentations:
        lifestyle_idx = (tokens >= 3) * (tokens <= 11)
        if lifestyle_idx.sum():
            ages[lifestyle_idx] += torch.randint(
                -20 * 365, 365 * 40, (lifestyle_idx.sum(),), generator=gen
            ).float()

    tokens = tokens.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)

    if padding is None or padding.lower() == "none" or no_event_token_rate == 0:
        pad = torch.ones(len(ix), 0)
    elif padding == "regular":
        pad = torch.arange(0, 36525, 365.25 * no_event_token_rate) * torch.ones(len(ix), 1) + 1
    elif padding == "random":
        pad = torch.randint(1, 36525, (len(ix), int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError

    m = ages.max(1, keepdim=True).values

    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
    ages = torch.hstack([ages, pad])

    tokens = tokens.masked_fill(ages > m, -1)
    ages = ages.masked_fill(ages > m, mask_time)

    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)

    tokens = tokens + 1

    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    x = tokens[:, :-1]
    a = ages[:, :-1]
    y = tokens[:, 1:]
    b = ages[:, 1:]

    x = x.masked_fill((x == 0) * (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, mask_time)

    if device == "cuda":
        x, a, y, b = [t.pin_memory().to(device, non_blocking=True) for t in (x, a, y, b)]
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)

    return x, a, y, b
