import numpy as np


def no_missing_event_tokens(tokens: np.ndarray, ethos_tokens: np.ndarray):

    _, uniq_count = np.unique(tokens, return_counts=True)
    _, ethos_uniq_count = np.unique(ethos_tokens, return_counts=True)

    uniq_count = uniq_count[1:]
    ethos_uniq_count = ethos_uniq_count[1 : 1 + len(uniq_count)]

    return np.array_equal(uniq_count, ethos_uniq_count)


def correct_num_of_time_tokens(
    tokens: np.ndarray, timesteps: np.ndarray, ethos_tokens: np.ndarray
):

    n_event_tokens = len(tokens)
    delta_t = np.diff(timesteps)
    n_time_tokens = (delta_t > 0).sum()
    n_ethos_tokens = len(ethos_tokens)

    return n_ethos_tokens == n_event_tokens + n_time_tokens
