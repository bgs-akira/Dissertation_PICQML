"""Evaluation: per-sample metrics and aggregate stats over a DataLoader.

Metric conventions (CLAUDE.md §5, §7):
    MSE = sum over modes of (p - p_hat)^2, per sample. Mean over batch yields
          the training loss; std over batch is reported by evaluate() for
          uncertainty quantification.
    TVD = 0.5 * sum over modes of |p - p_hat|, per sample. Bounded in [0, 1].
          Standard distributional distance reported in the paper (~2.9% on
          the 12-mode chip, paper §7.C).

Argument order is (p, p_hat) -- ground truth first, prediction second --
matching the scaffold contract in CLAUDE.md §9.

The training loop imports `mse` for its per-sample loss term and `evaluate`
for the per-epoch test pass.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


def mse(p: torch.Tensor, p_hat: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE = sum over modes of (p - p_hat)^2. Shape (batch,)."""
    return ((p - p_hat) ** 2).sum(dim=-1)


def tvd(p: torch.Tensor, p_hat: torch.Tensor) -> torch.Tensor:
    """Per-sample total variation distance = 0.5 * sum |p - p_hat|. Shape (batch,)."""
    return 0.5 * (p - p_hat).abs().sum(dim=-1)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run `model` over `loader` and return aggregate MSE / TVD stats.

    Args:
        model:  any module with `forward(V, input_ports) -> p_hat`.
        loader: DataLoader yielding (V, port, p) batches.
        device: "cpu" or "cuda".

    Returns:
        Dict with keys:
            "mse_mean"  : float, mean per-sample MSE
            "mse_std"   : float, unbiased std of per-sample MSE
            "tvd_mean"  : float, mean per-sample TVD
            "tvd_std"   : float, unbiased std of per-sample TVD
            "n_samples" : int,   total samples evaluated
    """
    model.eval()
    mses: list[torch.Tensor] = []
    tvds: list[torch.Tensor] = []
    with torch.no_grad():
        for V, port, p in loader:
            V = V.to(device)
            port = port.to(device)
            p = p.to(device)
            p_hat = model(V, port)
            mses.append(mse(p, p_hat).cpu())
            tvds.append(tvd(p, p_hat).cpu())
    mse_all = torch.cat(mses) if mses else torch.zeros(0)
    tvd_all = torch.cat(tvds) if tvds else torch.zeros(0)
    n = int(mse_all.numel())
    return {
        "mse_mean": mse_all.mean().item() if n > 0 else 0.0,
        "mse_std": mse_all.std(unbiased=True).item() if n > 1 else 0.0,
        "tvd_mean": tvd_all.mean().item() if n > 0 else 0.0,
        "tvd_std": tvd_all.std(unbiased=True).item() if n > 1 else 0.0,
        "n_samples": n,
    }
