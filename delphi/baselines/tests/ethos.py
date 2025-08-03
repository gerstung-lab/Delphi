import numpy as np
import torch

from delphi import DAYS_PER_YEAR
from delphi.baselines.ethos import _estimate_time_bins, create_ethos_sequence

batch_size = 10
seq_len = 60
vocab_size = 100
n_time_tokens = 10

max_age = 100 * DAYS_PER_YEAR
torch.manual_seed(42)

sample_shape = (batch_size, seq_len)
timesteps = torch.randint(0, int(max_age), sample_shape)
timesteps, _ = torch.sort(timesteps, dim=1)
tokens = torch.randint(1, vocab_size, sample_shape)

max_pad = int(seq_len / 2)
pad_idx = torch.randint(0, max_pad, (batch_size,))
pad_mask = torch.arange(seq_len).reshape(1, -1) <= pad_idx.reshape(-1, 1)

tokens[pad_mask] = 0
timesteps[pad_mask] = -1e4

timesteps_flat = timesteps.ravel().numpy()
timesteps_flat = timesteps_flat[timesteps_flat != -1e4]


def test_time_bin_estimates():

    time_bins = _estimate_time_bins(sample_t=timesteps_flat, n_tokens=n_time_tokens)

    assert len(time_bins) == n_time_tokens
    assert time_bins[0] > 0
    assert time_bins[1] < timesteps.max()


def no_missing_event_tokens(tokens: np.ndarray, ethos_tokens: np.ndarray):

    _, uniq_count = np.unique(tokens, return_counts=True)
    _, ethos_uniq_count = np.unique(ethos_tokens, return_counts=True)

    uniq_count = uniq_count[1:]
    ethos_uniq_count = ethos_uniq_count[1 : 1 + len(uniq_count)]

    return np.array_equal(uniq_count, ethos_uniq_count)


def correct_num_of_time_tokens(
    tokens: np.ndarray, timesteps: np.ndarray, ethos_tokens: np.ndarray
):

    for i in range(tokens.shape[0]):
        x = tokens[i, :]
        x = x[x > 0]
        n_event_tokens = len(x)

        t = timesteps[i, :]
        t = t[t != -1e4]
        delta_t = np.diff(t)
        n_time_tokens = (delta_t > 0).sum()

        n_ethos_tokens = (ethos_tokens[i, :] > 0).sum()

        if n_ethos_tokens == n_event_tokens + n_time_tokens:
            continue
        else:
            return False

    return True


def test_ethos_sequence():

    time_bins = _estimate_time_bins(sample_t=timesteps_flat, n_tokens=n_time_tokens)

    ethos_tokens = create_ethos_sequence(
        X=tokens.numpy(),
        T=timesteps.numpy(),
        offset=vocab_size - 1,
        time_bins=time_bins,
    )

    max_token = vocab_size - 1 + n_time_tokens
    assert ethos_tokens.max() == max_token

    assert no_missing_event_tokens(tokens=tokens.numpy(), ethos_tokens=ethos_tokens)

    assert correct_num_of_time_tokens(
        tokens=tokens.numpy(), ethos_tokens=ethos_tokens, timesteps=timesteps.numpy()
    )
