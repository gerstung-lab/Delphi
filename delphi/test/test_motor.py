import torch

from delphi import DAYS_PER_YEAR
from delphi.model.loss import broadcast_delta_t, time_to_event

seed = 42
batch_size = 2
seq_len = 11
vocab_size = 5
max_age = int(100 * DAYS_PER_YEAR)
max_delta = int(5 * DAYS_PER_YEAR)
max_pad = 5

torch.manual_seed(seed)
test_shape = (batch_size, seq_len)
timesteps = torch.randint(0, max_age, test_shape)
timesteps, s = torch.sort(timesteps, dim=1)
tokens = torch.randint(1, vocab_size, test_shape)

pad_idx = torch.randint(0, max_pad, (batch_size,))
pad_mask = torch.arange(seq_len).reshape(1, -1) <= pad_idx.reshape(-1, 1)

tokens[pad_mask] = 0
timesteps[pad_mask] = -1e4

age, targets_age = timesteps[:, :-1], timesteps[:, 1:]
_, targets = tokens[:, :-1], tokens[:, 1:]


def test_broadcast_delta_t() -> None:

    B = age.shape[0]
    L = age.shape[1]
    delta_age = torch.zeros((B, L, L), dtype=torch.int32)
    for i in range(L):
        delta_age[:, i, :] = targets_age - age[:, i].unsqueeze(-1)

    delta_age_broadcasted = broadcast_delta_t(age, targets_age)
    assert torch.equal(
        delta_age, delta_age_broadcasted
    ), "broadcast_delta_t did not produce expected output"


def no_out_of_range_idx(token_index: torch.Tensor, seq_len: int):
    return token_index.max() <= seq_len


def test_time_to_event() -> None:

    tte, no_event, token_index = time_to_event(
        age=age, targets_age=targets_age, targets=targets, vocab_size=vocab_size
    )
    assert no_out_of_range_idx(token_index=token_index, seq_len=targets.shape[1])
    B = age.shape[0]
    L = age.shape[1]
    V = vocab_size
    assert tte.shape == no_event.shape == token_index.shape == (B, L, V)

    for i in range(B):
        for j in range(L):
            future_tokens = targets[i, :].clone()
            future_tokens[:j] = 0
            for k in range(1, V):
                if k in future_tokens:
                    pos = torch.argwhere(future_tokens == k).min()
                    assert token_index[i, j, k] == pos
                    dt = targets_age[i, pos] - age[i, j]
                    assert tte[i, j, k] == dt
                else:
                    assert no_event[i, j, k] == 1
