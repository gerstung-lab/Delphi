import torch

from delphi.baselines.ethos import create_ethos_sequence, estimate_time_bins


def no_missing_event_tokens(tokens: torch.Tensor, ethos_tokens: torch.Tensor):

    _, uniq_count = torch.unique(tokens, return_counts=True)
    _, ethos_uniq_count = torch.unique(ethos_tokens, return_counts=True)

    uniq_count = uniq_count[1:]
    ethos_uniq_count = ethos_uniq_count[1 : 1 + len(uniq_count)]

    return torch.equal(uniq_count, ethos_uniq_count)


def correct_num_of_time_tokens(
    tokens: torch.Tensor, timesteps: torch.Tensor, ethos_tokens: torch.Tensor
):

    for i in range(tokens.shape[0]):
        x = tokens[i, :]
        x = x[x > 0]
        n_event_tokens = len(x)

        t = timesteps[i, :]
        t = t[t != -1e4]
        delta_t = torch.diff(t)
        n_time_tokens = (delta_t > 0).sum()

        n_ethos_tokens = (ethos_tokens[i, :] > 0).sum()

        if n_ethos_tokens == n_event_tokens + n_time_tokens:
            continue
        else:
            return False

    return True


class TestEthos:

    def test_ethos_sequence(
        self,
        tokens: torch.Tensor,
        timesteps: torch.Tensor,
        n_time_tokens: int,
        vocab_size: int,
    ):

        timesteps_flat = timesteps[timesteps != -1e4].ravel()
        time_bins = estimate_time_bins(
            sample_t=timesteps_flat.numpy(), n_tokens=n_time_tokens
        )
        assert time_bins[0] > 0
        if n_time_tokens > 1:
            assert time_bins[1] < timesteps.max()

        ethos_tokens, _ = create_ethos_sequence(
            X=tokens,
            T=timesteps,
            offset=vocab_size - 1,
            time_bins=torch.from_numpy(time_bins),
        )

        max_token = vocab_size - 1 + n_time_tokens
        assert ethos_tokens.max() == max_token

        assert no_missing_event_tokens(tokens=tokens, ethos_tokens=ethos_tokens)

        assert correct_num_of_time_tokens(
            tokens=tokens, ethos_tokens=ethos_tokens, timesteps=timesteps
        )
