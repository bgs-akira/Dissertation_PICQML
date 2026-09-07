"""Tests for src/data.py."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.chip_mesh import ChipMesh
from src.data import (
    PICDataset,
    load_dataset,
    load_seeds,
    make_synthetic_dataset,
    save_dataset,
    train_test_split,
)
from src.model import DigitalTwin


def _toy_dataset(N: int = 10, n_PS: int = 12, m: int = 4) -> PICDataset:
    x = torch.rand(N, n_PS, dtype=torch.float64)
    port = torch.randint(0, m, (N,))
    p = torch.rand(N, m, dtype=torch.float64)
    p = p / p.sum(dim=-1, keepdim=True)
    return PICDataset(x, port, p)


def _toy_model(m: int = 4) -> DigitalTwin:
    mesh = ChipMesh.clements(m)
    return DigitalTwin(mesh, torch.zeros(mesh.n_PS), torch.full((mesh.n_PS,), 0.034))


def test_picdataset_basic():
    """Length, indexing, n_PS, and m all behave as expected."""
    ds = _toy_dataset(N=10, n_PS=12, m=4)
    assert len(ds) == 10
    assert ds.n_PS == 12
    assert ds.m == 4
    V_i, port_i, p_i = ds[3]
    assert V_i.shape == (12,)
    assert port_i.shape == ()
    assert p_i.shape == (4,)


def test_picdataset_rejects_mismatched_batch():
    """V, port, p must agree on batch dim."""
    with pytest.raises(ValueError):
        PICDataset(
            torch.rand(10, 5),
            torch.zeros(5, dtype=torch.long),
            torch.rand(10, 4),
        )


def test_picdataset_rejects_wrong_dims():
    """V is 2-D, port is 1-D, p is 2-D."""
    with pytest.raises(ValueError):
        PICDataset(torch.rand(5), torch.zeros(5, dtype=torch.long), torch.rand(5, 4))


def test_load_save_roundtrip(tmp_path: Path):
    """save_dataset then load_dataset preserves values."""
    ds = _toy_dataset()
    save_dataset(ds, tmp_path)
    loaded = load_dataset(tmp_path)
    assert torch.allclose(loaded.x, ds.x)
    assert torch.equal(loaded.port, ds.port)
    assert torch.allclose(loaded.p, ds.p)


def test_load_seeds_roundtrip(tmp_path: Path):
    """load_seeds reads the same values that were written."""
    c_0 = torch.randn(12, dtype=torch.float64)
    c2_diag = torch.randn(12, dtype=torch.float64)
    np.save(tmp_path / "c_0_seed.npy", c_0.numpy())
    np.save(tmp_path / "c2_diag_seed.npy", c2_diag.numpy())
    loaded_c0, loaded_c2 = load_seeds(tmp_path)
    assert torch.allclose(loaded_c0, c_0)
    assert torch.allclose(loaded_c2, c2_diag)


def test_train_test_split_sizes_and_disjoint():
    """Sizes are correct and no sample appears in both splits."""
    ds = _toy_dataset(N=20)
    train, test = train_test_split(ds, test_frac=0.25, seed=42)
    assert len(train) + len(test) == 20
    assert len(test) == 5
    train_x = {tuple(v.tolist()) for v in train.x}
    test_x = {tuple(v.tolist()) for v in test.x}
    assert train_x.isdisjoint(test_x)


def test_train_test_split_reproducible():
    """Same seed -> same split."""
    ds = _toy_dataset(N=20)
    t1, _ = train_test_split(ds, test_frac=0.2, seed=7)
    t2, _ = train_test_split(ds, test_frac=0.2, seed=7)
    assert torch.equal(t1.x, t2.x)


def test_train_test_split_rejects_extreme_fracs():
    """test_frac outside (0, 1) raises."""
    ds = _toy_dataset(N=10)
    with pytest.raises(ValueError):
        train_test_split(ds, test_frac=0.0)
    with pytest.raises(ValueError):
        train_test_split(ds, test_frac=1.0)


def test_make_synthetic_dataset_shape_and_normalisation():
    """Synthetic dataset has correct shapes; p sums to 1 within tolerance."""
    model = _toy_model(m=4)
    ds = make_synthetic_dataset(model, n_samples=8, x_max=1.0, seed=0)
    assert len(ds) == 8
    assert ds.x.shape == (8, model.n_PS)
    assert ds.port.shape == (8,)
    assert ds.p.shape == (8, model.m)
    assert (ds.x >= 0).all() and (ds.x < 1.0).all()
    assert (ds.port >= 0).all() and (ds.port < model.m).all()
    sums = ds.p.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_make_synthetic_dataset_reproducible():
    """Same seed -> identical dataset."""
    model = _toy_model(m=4)
    a = make_synthetic_dataset(model, n_samples=4, seed=11)
    b = make_synthetic_dataset(model, n_samples=4, seed=11)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.port, b.port)
    assert torch.allclose(a.p, b.p)


def test_make_synthetic_dataset_with_noise_still_normalised():
    """Even with Gaussian noise + clamp + renormalise, p sums to 1."""
    model = _toy_model(m=4)
    ds = make_synthetic_dataset(model, n_samples=8, noise_std=0.05, seed=0)
    sums = ds.p.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
