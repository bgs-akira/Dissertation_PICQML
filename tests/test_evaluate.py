"""Tests for src/evaluate.py."""

import math

import pytest
import torch
from torch.utils.data import DataLoader

from src.chip_mesh import ChipMesh
from src.data import make_synthetic_dataset
from src.evaluate import evaluate, mse, tvd
from src.model import DigitalTwin


def _toy_model(m: int = 4) -> DigitalTwin:
    mesh = ChipMesh.clements(m)
    return DigitalTwin(mesh, torch.zeros(mesh.n_PS), torch.full((mesh.n_PS,), 0.034))


def test_mse_zero_when_equal():
    """mse(p, p) == 0 for every sample."""
    p = torch.rand(4, 6)
    p = p / p.sum(dim=-1, keepdim=True)
    out = mse(p, p)
    assert out.shape == (4,)
    assert torch.allclose(out, torch.zeros_like(out))


def test_mse_value_matches_manual():
    """mse matches a hand-computed sum-of-squares per sample."""
    p = torch.tensor([[0.1, 0.9], [0.5, 0.5]])
    p_hat = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
    expected = torch.tensor([0.01 + 0.01, 0.01 + 0.01])
    assert torch.allclose(mse(p, p_hat), expected)


def test_mse_symmetric_in_args():
    """mse(p, p_hat) == mse(p_hat, p): the difference is squared."""
    p = torch.rand(3, 4)
    p_hat = torch.rand(3, 4)
    assert torch.allclose(mse(p, p_hat), mse(p_hat, p))


def test_tvd_zero_when_equal():
    """tvd(p, p) == 0 for every sample."""
    p = torch.rand(4, 6)
    p = p / p.sum(dim=-1, keepdim=True)
    out = tvd(p, p)
    assert out.shape == (4,)
    assert torch.allclose(out, torch.zeros_like(out))


def test_tvd_bounded_0_1():
    """For valid probability distributions, TVD lies in [0, 1]."""
    torch.manual_seed(0)
    p = torch.rand(20, 8)
    p = p / p.sum(dim=-1, keepdim=True)
    p_hat = torch.rand(20, 8)
    p_hat = p_hat / p_hat.sum(dim=-1, keepdim=True)
    out = tvd(p, p_hat)
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_tvd_max_value_for_disjoint_distributions():
    """TVD = 1 when distributions have disjoint support."""
    p = torch.tensor([[1.0, 0.0]])
    p_hat = torch.tensor([[0.0, 1.0]])
    assert math.isclose(tvd(p, p_hat).item(), 1.0)


def test_tvd_symmetric_in_args():
    """tvd is symmetric in its arguments."""
    p = torch.rand(3, 4)
    p_hat = torch.rand(3, 4)
    assert torch.allclose(tvd(p, p_hat), tvd(p_hat, p))


def test_evaluate_returns_expected_keys():
    """evaluate() returns the five documented keys."""
    model = _toy_model(m=4)
    ds = make_synthetic_dataset(model, n_samples=16, seed=0)
    loader = DataLoader(ds, batch_size=4)
    out = evaluate(model, loader)
    assert set(out.keys()) == {"mse_mean", "mse_std", "tvd_mean", "tvd_std", "n_samples"}


def test_evaluate_n_samples_matches_loader():
    """n_samples equals total dataset length, regardless of batch size."""
    model = _toy_model(m=4)
    ds = make_synthetic_dataset(model, n_samples=15, seed=0)
    loader = DataLoader(ds, batch_size=4)
    assert evaluate(model, loader)["n_samples"] == 15


def test_evaluate_self_distance_is_zero():
    """Evaluating a model against data it generated yields MSE = TVD = 0."""
    model = _toy_model(m=4)
    ds = make_synthetic_dataset(model, n_samples=8, seed=0)
    loader = DataLoader(ds, batch_size=4)
    out = evaluate(model, loader)
    assert out["mse_mean"] == pytest.approx(0.0, abs=1e-10)
    assert out["tvd_mean"] == pytest.approx(0.0, abs=1e-6)
