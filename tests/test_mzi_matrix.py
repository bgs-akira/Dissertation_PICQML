"""The MZI block equals the calibration document's equations.

Unitarity and topology are covered elsewhere. These check the ALGEBRA:
that the 2x2 transfer matrix produced by walking ``mesh.layers`` is
literally the symmetric-MZI matrix of the document's section 2 --
including the ordering of the two couplers (alpha acts first, beta
second) and the global ``i e^{i Sigma}`` prefactor, which is NOT a
removable global phase once MZIs compose in a mesh.

A 2-mode Bell mesh is exactly one MZI followed by two independent phase
shifters, so holding those at zero leaves the bare MZI to compare.
"""

import numpy as np
import pytest
import torch

from src.chip_mesh import ChipMesh
from src.model import DigitalTwin


@pytest.fixture(scope="module")
def rig():
    mesh = ChipMesh.bell(2)
    model = DigitalTwin(
        mesh,
        torch.zeros(mesh.n_PS, dtype=torch.float64),
        torch.zeros(mesh.n_PS, dtype=torch.float64),
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.T_logit.fill_(30.0)          # T_out -> 1
    bs_a, bs_b = mesh.mzi_bs(0, 0)
    up, lo = mesh.mzi_arm_ps(0, 0)
    return mesh, model, bs_a, bs_b, up, lo


def _R_from_alpha(alpha: float) -> float:
    """The document's coupler deviation maps to this codebase's R."""
    return float(np.cos(np.pi / 4 + alpha) ** 2)


def _mine(rig, th1, th2, alpha, beta) -> np.ndarray:
    """The 2x2 built by the model's own layer machinery."""
    mesh, model, bs_a, bs_b, up, lo = rig
    with torch.no_grad():
        logit = model.R_logit.clone()
        for idx, a in ((bs_a, alpha), (bs_b, beta)):
            R = _R_from_alpha(a)
            logit[idx] = float(np.log(R / (1.0 - R)))
        model.R_logit.copy_(logit)
        phi = torch.zeros(mesh.n_PS, dtype=torch.float64)
        phi[up] = th1
        phi[lo] = th2
        return model._build_U(phi).numpy()


def _B(a: float) -> np.ndarray:
    """Document equation 1."""
    return np.array([
        [np.cos(np.pi / 4 + a), 1j * np.sin(np.pi / 4 + a)],
        [1j * np.sin(np.pi / 4 + a), np.cos(np.pi / 4 + a)],
    ])


def _eqn2(th1, th2, alpha, beta) -> np.ndarray:
    """B(beta) @ diag(e^{i th1}, e^{i th2}) @ B(alpha)."""
    P = np.diag([np.exp(1j * th1), np.exp(1j * th2)])
    return _B(beta) @ P @ _B(alpha)


def _eqn3(th1, th2, alpha, beta) -> np.ndarray:
    """Closed form in delta and Sigma."""
    S, d = (th1 + th2) / 2, (th1 - th2) / 2
    bm, bp = beta - alpha, beta + alpha
    return 1j * np.exp(1j * S) * np.array([
        [np.cos(bm) * np.sin(d) + 1j * np.sin(bp) * np.cos(d),
         np.cos(bp) * np.cos(d) - 1j * np.sin(bm) * np.sin(d)],
        [np.cos(bp) * np.cos(d) + 1j * np.sin(bm) * np.sin(d),
         -np.cos(bm) * np.sin(d) + 1j * np.sin(bp) * np.cos(d)],
    ])


def test_two_mode_bell_mesh_is_one_mzi_plus_two_independents(rig):
    mesh = rig[0]
    assert mesh.n_PS == 4 and mesh.n_BS == 2
    assert len(mesh.mzi_labels()) == 1
    assert len(mesh.ps_indices_with_role("independent")) == 2


def test_matches_equation_2(rig):
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        th1, th2 = rng.uniform(0, 2 * np.pi, 2)
        alpha, beta = rng.uniform(-0.06, 0.06, 2)
        worst = max(worst, np.abs(
            _mine(rig, th1, th2, alpha, beta) - _eqn2(th1, th2, alpha, beta)
        ).max())
    assert worst < 1e-12


def test_matches_equation_3(rig):
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(100):
        th1, th2 = rng.uniform(0, 2 * np.pi, 2)
        alpha, beta = rng.uniform(-0.06, 0.06, 2)
        worst = max(worst, np.abs(
            _mine(rig, th1, th2, alpha, beta) - _eqn3(th1, th2, alpha, beta)
        ).max())
    assert worst < 1e-12


def test_matches_equation_6_ideal_limit(rig):
    """alpha = beta = 0 gives i e^{iS} [[sin d, cos d], [cos d, -sin d]]."""
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(100):
        th1, th2 = rng.uniform(0, 2 * np.pi, 2)
        S, d = (th1 + th2) / 2, (th1 - th2) / 2
        want = 1j * np.exp(1j * S) * np.array([
            [np.sin(d), np.cos(d)], [np.cos(d), -np.sin(d)]])
        worst = max(worst, np.abs(_mine(rig, th1, th2, 0.0, 0.0) - want).max())
    assert worst < 1e-12


def test_matches_equation_7_cross_state(rig):
    """delta = 0 (both arms equal) is the cross state."""
    alpha, beta, S = 0.03, -0.02, 0.4
    want = 1j * np.exp(1j * S) * np.array([
        [1j * np.sin(beta + alpha), np.cos(beta + alpha)],
        [np.cos(beta + alpha), 1j * np.sin(beta + alpha)]])
    assert np.abs(_mine(rig, S, S, alpha, beta) - want).max() < 1e-12


def test_matches_equation_8_bar_state(rig):
    """delta = pi/2 is the bar state."""
    alpha, beta, S = 0.03, -0.02, 0.4
    want = 1j * np.exp(1j * S) * np.array([
        [np.cos(beta - alpha), -1j * np.sin(beta - alpha)],
        [1j * np.sin(beta - alpha), -np.cos(beta - alpha)]])
    got = _mine(rig, S + np.pi / 2, S - np.pi / 2, alpha, beta)
    assert np.abs(got - want).max() < 1e-12


def test_first_coupler_carries_alpha(rig):
    """Ordering matters, and this check can actually tell the difference.

    BS_a (the coupler light meets first) must carry alpha, BS_b beta.
    Swapping them changes the matrix by ~7e-2 at these deviations, so a
    passing test above is evidence of correct ordering rather than of an
    insensitive comparison.
    """
    th1, th2, a, b = 1.1, 0.3, 0.05, -0.04
    correct = np.abs(_mine(rig, th1, th2, a, b) - _eqn2(th1, th2, a, b)).max()
    swapped = np.abs(_mine(rig, th1, th2, a, b) - _eqn2(th1, th2, b, a)).max()
    assert correct < 1e-12
    assert swapped > 1e-3


def test_global_prefactor_is_present(rig):
    """The i e^{i Sigma} factor is real structure, not a removable phase.

    Sweeping Sigma at fixed delta must rotate the block's phase; if the
    prefactor were dropped the matrix would be Sigma-independent, and
    every Sigma calibration downstream would be measuring nothing.
    """
    d = 0.3
    ref = _mine(rig, d, -d, 0.0, 0.0)
    rotated = _mine(rig, np.pi / 2 + d, np.pi / 2 - d, 0.0, 0.0)
    assert np.abs(rotated - ref).max() > 1e-3
    # ...and the rotation is exactly e^{i Sigma}.
    np.testing.assert_allclose(
        rotated, np.exp(1j * np.pi / 2) * ref, atol=1e-12,
    )
