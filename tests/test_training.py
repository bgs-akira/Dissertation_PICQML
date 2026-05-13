"""Tests for the training loop."""

import pytest


def test_loss_decreases_on_synthetic_data():
    """On data generated from a known-ground-truth DigitalTwin, loss should
    drop monotonically (in expectation) over 10 epochs."""
    pytest.skip("not implemented")


def test_parameter_groups_have_correct_lrs():
    """Adam optimiser has three param groups with lrs lr_C2, lr_R, lr_Tout."""
    pytest.skip("not implemented")


def test_returns_best_test_mse_epoch():
    """train() returns the epoch index of the lowest test MSE and the
    corresponding model state."""
    pytest.skip("not implemented")
