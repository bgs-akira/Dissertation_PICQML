"""Evaluation: TVD, plots, diagnostics."""

from __future__ import annotations

import torch


def tvd(p: torch.Tensor, p_hat: torch.Tensor) -> torch.Tensor:
    """Total variation distance per sample.

    Args:
        p, p_hat : (batch, m) real, each row sums to 1.

    Returns:
        (batch,) real in [0, 1].
    """
    raise NotImplementedError


def mse(p: torch.Tensor, p_hat: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance per sample, summed over modes.

    Same convention as the training loss (CLAUDE.md §5): sum over modes,
    mean over batch returned as a scalar; this helper returns per-sample
    values so callers can choose the reduction.
    """
    raise NotImplementedError
