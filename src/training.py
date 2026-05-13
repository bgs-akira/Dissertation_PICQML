"""Training loop for one ML stage.

Per CLAUDE.md §6: all three trainable blocks (C_2, R, T_out) update
simultaneously with per-parameter learning rates. No freeze alternation
within a stage. c_0 is a buffer and is not optimised here.
"""

from __future__ import annotations

from .model import DigitalTwin


def train(
    model: DigitalTwin,
    train_loader,
    test_loader,
    *,
    epochs: int = 500,
    lr_C2: float = 1e-5,
    lr_R: float = 1e-3,
    lr_Tout: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """Run one ML stage.

    Returns:
        dict with per-epoch train/test MSE and TVD, plus the best test-MSE
        epoch and the model state at that epoch.
    """
    raise NotImplementedError
