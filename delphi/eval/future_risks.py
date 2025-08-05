import torch

from delphi.data.core import BaseDataConfig, MiniDataset
from delphi.model.transformer import Delphi, load_ckpt
from delphi.sampler import CausalSamplerConfig
from delphi.tokenizer import load_tokenizer_from_ckpt


def integrate_risk(x: torch.Tensor, t: torch.Tensor, start: float, end: float):
    r"""
    Aggregate values x over time intervals t within a specified time window [start, end].
    As per the theory of non-homogeneous exponential distribution, the probability
    an event occurs in the time window [start, end] is given by:
    P(event in [start, end]) = 1 - exp(- \int_{start}^{end} \lambda(t) dt)
    where \lambda(t) is the disease rate at time t.
    This this function calculates the integral of the disease rate over the time window
    under that piecewise constant disease rate assumption, using the tokens that
    fall in the time window.

    Args:
        x: Disease rate to integrate, lambda_0, ...., lambda_n, [batch, block_size, disease]
        t: Time points, days since birth, t_0, ...., t_n, t_(n+1) [batch, block_size]
            (the last time point is needed to calculate the duration of the last event)
        start: Start of time window
        end: End of time window

    Returns:
        Aggregated risk values, normalized by time exposure
    """

    # Clamp time values to the end of the window
    t_clamped = t.clamp(None, end)

    # Create usage mask for each time interval
    # If there are no time points in the window, the use mask will be all zeros
    # and the risk will be NaN
    use = ((t_clamped >= start) * (t_clamped < end)) + 0.0
    dt = t_clamped.diff(1)

    # Apply masks to get effective time exposure within the window
    dt_masked = dt * use[:, :-1]

    # Normalize time weights to sum to the length of the window
    dt_norm = dt_masked / (dt_masked.sum(1).unsqueeze(-1) + 1e-6) * (end - start)

    # Calculate risk by weighting x values with normalized time exposure
    # print(x.shape, dt_norm.shape)
    risk = x * dt_norm.unsqueeze(-1)
    risk = risk.sum(-2)  # Sum over the time dimension

    # Set zero risks to NaN (indicates no exposure in the time window)
    risk[risk == 0] = torch.nan

    return risk


ckpt = "checkpoints/finetune/baseline"
model, _ = load_ckpt(ckpt_path=ckpt)
tokenizer = load_tokenizer_from_ckpt(ckpth_path=ckpt)

sampler_cfg = CausalSamplerConfig()

data_cfg = BaseDataConfig(subject_list="")
