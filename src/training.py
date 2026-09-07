"""Training loop for one ML stage.

Per CLAUDE.md §6: all three trainable blocks (C_2_raw, R_logit, T_logit)
update simultaneously via Adam with per-parameter learning rates. No freeze
alternation within a stage; c_0 stays a buffer and T_in is not in the model
(CLAUDE.md §10 pitfalls 1-2).

Loss convention (CLAUDE.md §5, §6.2): squared L2 distance per sample, summed
over modes, then averaged over the batch. The paper's per-parameter learning
rates are tuned to this convention.

Within-stage LR scheduling (``lr_schedule``)
-------------------------------------------
At the paper LRs, m>=8 chips oscillate around the minimum: training MSE
swings by ~10x once the model is near a good basin, and the best test MSE
is reached very early (epoch ~30 of 500) after which the model gets
progressively worse. Lowering the LR uniformly cures the oscillation but
also slows down the early descent. ``lr_schedule="cosine"`` (recommended
for m>=8) anneals each parameter group's LR from its initial value down
to ``eta_min`` over the stage, giving fast initial descent and a smooth
landing at the basin. Default ``None`` keeps the historical constant-LR
behaviour for backward compatibility.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.evaluate import evaluate, mse
from src.model import DigitalTwin


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    kind: str | None,
    *,
    epochs: int,
    eta_min_frac: float = 0.0,
):
    """Build a within-stage LR scheduler. ``kind=None`` -> no schedule.

    ``"cosine"`` anneals every parameter group from its initial LR down to
    ``eta_min_frac * initial`` over ``epochs`` epochs. ``eta_min_frac=0``
    uses ``CosineAnnealingLR`` directly; otherwise a ``LambdaLR`` applies
    the same cosine multiplier to every group, scaled per-group by its
    base LR so each group anneals to the same fraction of its own start.
    """
    if kind is None:
        return None
    if kind != "cosine":
        raise ValueError(
            f"lr_schedule must be None or 'cosine', got {kind!r}"
        )

    if eta_min_frac == 0.0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.0,
        )

    def lr_fn(step: int) -> float:
        return (
            eta_min_frac
            + 0.5 * (1.0 - eta_min_frac)
            * (1.0 + math.cos(math.pi * step / max(1, epochs)))
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lr_fn for _ in optimizer.param_groups],
    )


def _build_optimizer(
    model: DigitalTwin,
    *,
    lr_C2: float,
    lr_R: float,
    lr_Tout: float,
) -> torch.optim.Adam:
    """Build the Adam optimiser with one parameter group per trainable block.

    Group order is fixed: 0 -> C_2_raw, 1 -> R_logit, 2 -> T_logit. Tests rely
    on this ordering.
    """
    return torch.optim.Adam(
        [
            {"params": [model.C_2_raw], "lr": lr_C2},
            {"params": [model.R_logit], "lr": lr_R},
            {"params": [model.T_logit], "lr": lr_Tout},
        ]
    )


def train(
    model: DigitalTwin,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int = 500,
    lr_C2: float = 1e-5,
    lr_R: float = 1e-3,
    lr_Tout: float = 1e-3,
    lr_schedule: str | None = None,
    lr_eta_min_frac: float = 0.0,
    device: str = "cpu",
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one ML stage.

    Optimises C_2_raw, R_logit, T_logit jointly with Adam using per-parameter
    learning rates from CLAUDE.md §6.2. The model is updated in place.

    Args:
        model:        a DigitalTwin to optimise.
        train_loader: DataLoader over the training PICDataset.
        test_loader:  DataLoader over the test PICDataset.
        epochs:       number of passes over train_loader.
        lr_C2, lr_R, lr_Tout: per-parameter learning rates.
        lr_schedule:  optional within-stage LR scheduler. ``None`` (default)
                      keeps the historical constant-LR behaviour; ``"cosine"``
                      anneals every group's LR from its initial value to
                      ``lr_eta_min_frac * initial`` over ``epochs`` epochs.
                      Recommended for m>=8 to remove the late-stage
                      oscillation that caps the post-ML TVD around 2 %.
        lr_eta_min_frac: floor for the cosine schedule, expressed as a
                      fraction of each group's initial LR. ``0.0`` (default)
                      anneals all the way to zero — the final epochs do
                      essentially no update, which is what we want when the
                      goal is to land at the minimum reached by the
                      schedule's bulk. Set to e.g. ``0.1`` to keep a small
                      residual LR throughout.
        device:       "cpu" or "cuda".

    Returns:
        Dict with keys:
            "train_mse"       : list[float], per-epoch training MSE.
            "test_mse"        : list[float], per-epoch test MSE.
            "test_tvd"        : list[float], per-epoch test TVD.
            "best_epoch"      : int,   epoch with the lowest test MSE.
            "best_test_mse"   : float, the lowest test MSE observed.
            "best_state_dict" : dict,  deep-copied state_dict at best_epoch.
            "elapsed"         : float, wall-clock seconds for this stage.
                                Also printed at the end of the stage when
                                ``verbose=True`` so it appears in any
                                tee'd log file.

    After train() returns, `model` holds the FINAL-epoch state (not the best).
    To restore the best-epoch state, call:
        model.load_state_dict(history["best_state_dict"])
    """
    model = model.to(device)
    optimizer = _build_optimizer(model, lr_C2=lr_C2, lr_R=lr_R, lr_Tout=lr_Tout)
    scheduler = _build_scheduler(
        optimizer, lr_schedule, epochs=epochs, eta_min_frac=lr_eta_min_frac,
    )

    train_mse_history: list[float] = []
    test_mse_history: list[float] = []
    test_tvd_history: list[float] = []

    best_test_mse = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    log_every = max(1, epochs // 5)  # ~5 progress lines per ML stage

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        train_mse_total = 0.0
        train_count = 0
        for V, port, p in train_loader:
            V = V.to(device)
            port = port.to(device)
            p = p.to(device)

            optimizer.zero_grad()
            p_hat = model(V, port)
            loss = mse(p, p_hat).mean()
            loss.backward()
            optimizer.step()

            batch_size = V.shape[0]
            train_mse_total += loss.item() * batch_size
            train_count += batch_size

        train_mse = train_mse_total / train_count
        test_stats = evaluate(model, test_loader, device)

        train_mse_history.append(train_mse)
        test_mse_history.append(test_stats["mse_mean"])
        test_tvd_history.append(test_stats["tvd_mean"])

        if test_stats["mse_mean"] < best_test_mse:
            best_test_mse = test_stats["mse_mean"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler.step()

        if verbose and ((epoch + 1) % log_every == 0 or epoch == epochs - 1):
            print(
                f"    epoch {epoch + 1:>{len(str(epochs))}}/{epochs}  "
                f"train_mse={train_mse:.3e}  "
                f"test_mse={test_stats['mse_mean']:.3e}  "
                f"test_tvd={test_stats['tvd_mean'] * 100:.3f}%",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"    ML stage elapsed: {elapsed:.1f} s", flush=True)

    return {
        "train_mse": train_mse_history,
        "test_mse": test_mse_history,
        "test_tvd": test_tvd_history,
        "best_epoch": best_epoch,
        "best_test_mse": best_test_mse,
        "best_state_dict": best_state,
        "elapsed": elapsed,
    }
