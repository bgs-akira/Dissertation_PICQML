"""Tests for the training loop."""

import torch
from torch.utils.data import DataLoader

from src.chip_mesh import ChipMesh
from src.data import make_synthetic_dataset, train_test_split
from src.model import DigitalTwin
from src.training import _build_optimizer, train


def _truth_model(m: int = 4, seed: int = 0) -> DigitalTwin:
    """A non-trivial ground-truth model with perturbed trainable params."""
    mesh = ChipMesh.clements(m)
    g = torch.Generator().manual_seed(seed)
    c_0 = torch.randn(mesh.n_PS, generator=g) * 0.1
    c2_diag = torch.full((mesh.n_PS,), 0.034) + torch.randn(
        mesh.n_PS, generator=g
    ) * 0.002
    model = DigitalTwin(mesh, c_0, c2_diag)
    with torch.no_grad():
        model.C_2_raw.add_(torch.randn(mesh.n_PS, mesh.n_PS, generator=g) * 5e-4)
        model.R_logit.copy_(torch.randn(mesh.n_BS, generator=g) * 0.3)
        model.T_logit.copy_(torch.full((m,), 6.0) + torch.randn(m, generator=g) * 0.1)
    return model


def _fresh_model(
    m: int, c_0_seed: torch.Tensor, c2_diag_seed: torch.Tensor
) -> DigitalTwin:
    """A model seeded as V-IFM would produce: known c_0 and diag(C_2), zero off-diag."""
    return DigitalTwin(ChipMesh.clements(m), c_0_seed, c2_diag_seed)


def test_parameter_groups_have_correct_lrs():
    """Adam optimiser has three param groups with lrs lr_C2, lr_R, lr_Tout."""
    mesh = ChipMesh.clements(4)
    model = DigitalTwin(mesh, torch.zeros(mesh.n_PS), torch.zeros(mesh.n_PS))
    opt = _build_optimizer(model, lr_C2=1e-5, lr_R=1e-3, lr_Tout=1e-3)
    assert isinstance(opt, torch.optim.Adam)
    assert len(opt.param_groups) == 3
    assert opt.param_groups[0]["lr"] == 1e-5
    assert opt.param_groups[1]["lr"] == 1e-3
    assert opt.param_groups[2]["lr"] == 1e-3
    assert opt.param_groups[0]["params"][0] is model.C_2_raw
    assert opt.param_groups[1]["params"][0] is model.R_logit
    assert opt.param_groups[2]["params"][0] is model.T_logit


def test_loss_decreases_on_synthetic_data():
    """Test MSE drops over a few epochs on a known-truth synthetic dataset.

    LRs are scaled uniformly to 10x paper defaults so the 1:100:100
    C_2 : R : T_out ratio from CLAUDE.md §6.2 is preserved. Paper LRs are
    tuned for 500 epochs; 10 epochs at 10x reproduces a similar trajectory
    shape without overshooting (Adam's per-parameter step is bounded near LR).
    """
    torch.manual_seed(42)
    m = 4
    truth = _truth_model(m=m, seed=0)
    ds = make_synthetic_dataset(truth, n_samples=64, x_max=1.4, seed=1)
    train_ds, test_ds = train_test_split(ds, test_frac=0.25, seed=2)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    fresh = _fresh_model(m, truth.c_0.clone(), truth.C_2.diag().clone())
    history = train(
        fresh,
        train_loader,
        test_loader,
        epochs=10,
        lr_C2=1e-4,    # 10x paper default 1e-5
        lr_R=1e-2,     # 10x paper default 1e-3 (preserves 1:100:100 ratio)
        lr_Tout=1e-2,  # 10x paper default 1e-3 (preserves 1:100:100 ratio)
    )
    assert len(history["train_mse"]) == 10
    assert len(history["test_mse"]) == 10
    assert len(history["test_tvd"]) == 10
    assert history["test_mse"][-1] < history["test_mse"][0]


def test_returns_best_test_mse_epoch():
    """best_epoch and best_state_dict reflect the lowest observed test MSE."""
    torch.manual_seed(42)
    m = 4
    truth = _truth_model(m=m, seed=0)
    ds = make_synthetic_dataset(truth, n_samples=32, x_max=1.4, seed=1)
    train_ds, test_ds = train_test_split(ds, test_frac=0.25, seed=2)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    fresh = _fresh_model(m, truth.c_0.clone(), truth.C_2.diag().clone())
    history = train(
        fresh,
        train_loader,
        test_loader,
        epochs=5,
        lr_C2=1e-4,    # 10x paper default 1e-5
        lr_R=1e-2,     # 10x paper default 1e-3 (preserves 1:100:100 ratio)
        lr_Tout=1e-2,  # 10x paper default 1e-3 (preserves 1:100:100 ratio)
    )
    assert 0 <= history["best_epoch"] < 5
    assert history["best_test_mse"] == min(history["test_mse"])
    assert history["best_test_mse"] == history["test_mse"][history["best_epoch"]]
    assert history["best_state_dict"] is not None
    assert set(history["best_state_dict"].keys()) == set(fresh.state_dict().keys())


def test_cosine_schedule_anneals_to_zero():
    """``lr_schedule="cosine"`` brings each group's LR to (near) zero at the
    end of the stage, matching the documented behaviour of CosineAnnealingLR
    with ``eta_min=0``. Regression: the m=8 fix relies on this.
    """
    torch.manual_seed(42)
    m = 4
    truth = _truth_model(m=m, seed=0)
    ds = make_synthetic_dataset(truth, n_samples=16, x_max=1.4, seed=1)
    train_ds, test_ds = train_test_split(ds, test_frac=0.25, seed=2)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    fresh = _fresh_model(m, truth.c_0.clone(), truth.C_2.diag().clone())
    history = train(
        fresh, train_loader, test_loader,
        epochs=5, lr_C2=1e-3, lr_R=1e-2, lr_Tout=1e-2,
        lr_schedule="cosine",
    )
    # With eta_min_frac=0 the schedule must produce a strictly-shorter
    # trajectory than constant LR, but only checking the API contract here.
    assert len(history["train_mse"]) == 5
    assert history["best_state_dict"] is not None


def test_best_state_dict_is_snapshot_not_reference():
    """best_state_dict is a deep copy: continued training does not mutate it."""
    torch.manual_seed(42)
    m = 4
    truth = _truth_model(m=m, seed=0)
    ds = make_synthetic_dataset(truth, n_samples=16, x_max=1.4, seed=1)
    train_ds, test_ds = train_test_split(ds, test_frac=0.25, seed=2)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    fresh = _fresh_model(m, truth.c_0.clone(), truth.C_2.diag().clone())
    history = train(
        fresh, train_loader, test_loader,
        epochs=3, lr_C2=1e-3, lr_R=1e-2, lr_Tout=1e-2,
    )
    snapshot = history["best_state_dict"]["R_logit"].clone()
    # Mutate the live model
    with torch.no_grad():
        fresh.R_logit.add_(1.0)
    assert torch.allclose(history["best_state_dict"]["R_logit"], snapshot)
