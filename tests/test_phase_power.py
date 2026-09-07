"""Tests for the iterative phase-power solver (Supplement H).

The solver inverts phi = C_2 @ x + c_0 for x in [0, x_max], working modulo
2pi via wrap_to_pi on the residual. Tests cover:

  - the trivial fixed point (x = 0 when phi_target == c_0),
  - diagonal-only C_2 in 1-D and N-D (analytical reachable target),
  - full crosstalk matrix at realistic paper magnitudes,
  - 2pi-periodic targets resolved through wrap,
  - returned x always inside [0, x_max] and threshold respected,
  - non-convergence emits a RuntimeWarning,
  - shape / positivity validation,
  - reshuffle seed reproducibility,
  - the _wrap_to_pi helper itself (pitfall 14 is easy to get wrong).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.power_lookup import K_NOMINAL
from src.phase_power import X_MAX, _wrap_to_pi, solve_phase_power


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _residual(phi_target: np.ndarray, C_2: np.ndarray, c_0: np.ndarray,
              x: np.ndarray) -> np.ndarray:
    """Wrapped phase residual phi_now - phi_target, in (-pi, pi]."""
    return _wrap_to_pi(C_2 @ x + c_0 - phi_target)


def _diag_C2(n_PS: int, value: float = K_NOMINAL) -> np.ndarray:
    return np.eye(n_PS) * value


# ---------------------------------------------------------------------------
# _wrap_to_pi
# ---------------------------------------------------------------------------


def test_wrap_to_pi_identity_in_range():
    """Inputs already in [-pi, pi) survive wrapping unchanged."""
    x = np.array([-np.pi + 1e-3, -1.0, 0.0, 1.0, np.pi - 1e-3])
    np.testing.assert_allclose(_wrap_to_pi(x), x, atol=1e-12)


def test_wrap_to_pi_shifts_by_2pi():
    """Adding 2pi to any input does not change the wrapped value."""
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    np.testing.assert_allclose(
        _wrap_to_pi(x + 2 * np.pi), _wrap_to_pi(x), atol=1e-12
    )
    np.testing.assert_allclose(
        _wrap_to_pi(x - 2 * np.pi), _wrap_to_pi(x), atol=1e-12
    )


def test_wrap_to_pi_known_values():
    """Spot-check a few representative angles."""
    x = np.array([3 * np.pi / 2, -3 * np.pi / 2, 5 * np.pi, 0.0])
    expected = np.array([-np.pi / 2, np.pi / 2, -np.pi, 0.0])
    # 5*pi mod 2pi = pi; (pi + pi) % 2pi - pi = 0 - pi = -pi (lower edge).
    np.testing.assert_allclose(_wrap_to_pi(x), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Trivial fixed point
# ---------------------------------------------------------------------------


def test_zero_target_zero_c0_returns_zero():
    """At x=0 phi_now = c_0; if c_0 = phi_target, the solver returns x = 0."""
    n_PS = 6
    x = solve_phase_power(
        phi_target=np.zeros(n_PS),
        C_2=_diag_C2(n_PS),
        c_0=np.zeros(n_PS),
    )
    np.testing.assert_array_equal(x, np.zeros(n_PS))


def test_phi_target_equals_c_0_returns_zero():
    """Same as above but with a non-zero c_0 seed."""
    n_PS = 6
    rng = np.random.default_rng(0)
    c_0 = rng.uniform(-1.0, 1.0, size=n_PS)
    x = solve_phase_power(
        phi_target=c_0.copy(),
        C_2=_diag_C2(n_PS),
        c_0=c_0,
    )
    np.testing.assert_array_equal(x, np.zeros(n_PS))


# ---------------------------------------------------------------------------
# Diagonal-only convergence
# ---------------------------------------------------------------------------


def test_diagonal_only_single_shifter_analytical_v():
    """n_PS = 1: the law is linear, so x = target / k exactly."""
    C_2 = np.array([[K_NOMINAL]])
    c_0 = np.array([0.0])
    phi_target = np.array([1.0])
    x = solve_phase_power(
        phi_target=phi_target,
        C_2=C_2,
        c_0=c_0,
        threshold=1e-6,
    )
    expected = 1.0 / K_NOMINAL
    assert x[0] == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("n_PS", [4, 16, 64])
def test_diagonal_only_converges_within_threshold(n_PS):
    """Diagonal C_2 with paper-magnitude self-heating, reachable targets."""
    rng = np.random.default_rng(42)
    C_2 = _diag_C2(n_PS)
    c_0 = rng.uniform(-np.pi, np.pi, size=n_PS)
    # Reachable within X_MAX: max phase swing = K_NOMINAL * 1.4 W = 4*pi rad.
    phi_target = c_0 + rng.uniform(0.0, 4.0 * np.pi, size=n_PS)

    x = solve_phase_power(phi_target, C_2, c_0, threshold=1e-4)
    delta = _residual(phi_target, C_2, c_0, x)
    assert np.max(np.abs(delta)) < 1e-4
    assert np.all(x >= 0.0)
    assert np.all(x <= X_MAX + 1e-12)


# ---------------------------------------------------------------------------
# Full crosstalk matrix
# ---------------------------------------------------------------------------


def test_full_crosstalk_matrix_converges():
    """Realistic magnitudes: diag ~ k, off-diag ~ 1.5 % of k."""
    n_PS = 12
    rng = np.random.default_rng(1)
    C_2 = _diag_C2(n_PS)
    off = rng.uniform(-0.015 * K_NOMINAL, 0.015 * K_NOMINAL, size=(n_PS, n_PS))
    np.fill_diagonal(off, 0.0)
    C_2 = C_2 + off
    c_0 = rng.uniform(-np.pi, np.pi, size=n_PS)
    phi_target = c_0 + rng.uniform(0.0, 3.0, size=n_PS)

    x = solve_phase_power(
        phi_target, C_2, c_0,
        threshold=1e-4, max_iter=20000, seed=0,
    )
    delta = _residual(phi_target, C_2, c_0, x)
    assert np.max(np.abs(delta)) < 1e-4
    assert np.all((x >= 0.0) & (x <= X_MAX + 1e-12))


# ---------------------------------------------------------------------------
# 2pi periodicity (wrap_to_pi is what makes out-of-range targets reachable)
# ---------------------------------------------------------------------------


def test_target_plus_2pi_converges_to_same_phase_mod_2pi():
    """phi_target and phi_target + 2pi are physically equivalent; the solver
    must converge in both cases and the residuals must agree mod 2pi."""
    n_PS = 4
    C_2 = _diag_C2(n_PS)
    c_0 = np.zeros(n_PS)
    base = np.array([0.7, 1.3, 2.1, 0.9])

    V_a = solve_phase_power(base.copy(), C_2, c_0, seed=0)
    V_b = solve_phase_power(base + 2 * np.pi, C_2, c_0, seed=0)

    # Both solutions must satisfy the wrapped residual.
    for x, target in ((V_a, base), (V_b, base + 2 * np.pi)):
        delta = _residual(target, C_2, c_0, x)
        assert np.max(np.abs(delta)) < 1e-4

    # And the induced phases differ from the targets only by multiples of 2pi.
    phi_a = C_2 @ V_a + c_0
    phi_b = C_2 @ V_b + c_0
    np.testing.assert_allclose(_wrap_to_pi(phi_a - phi_b), 0.0, atol=1e-4)


def test_target_just_above_direct_reach_uses_wrap():
    """Direct reach is k * X_MAX = 4*pi ~ 12.57 rad. A target above that is
    out of direct reach but reachable mod 2pi."""
    C_2 = np.array([[K_NOMINAL]])
    c_0 = np.array([0.0])
    phi_target = np.array([4.0 * np.pi + 1.0])

    x = solve_phase_power(phi_target, C_2, c_0, threshold=1e-6, seed=0)
    delta = _residual(phi_target, C_2, c_0, x)
    assert np.max(np.abs(delta)) < 1e-6
    assert 0.0 <= x[0] <= X_MAX

    # The budget spans 4*pi, so a target above direct reach has SEVERAL
    # valid pre-images (wrap down by 2*pi, by 4*pi, ...). Asserting a
    # particular branch would over-specify the solver -- it is free to
    # land on whichever the descent reaches first, and in fact prefers
    # the 4*pi wrap here because it costs less power. What must hold is
    # that the delivered phase matches the target modulo 2*pi, and that
    # the solution really is a wrapped one rather than direct.
    delivered = float((C_2 @ x + c_0)[0])
    assert _wrap_to_pi(np.array([delivered - phi_target[0]]))[0] == (
        pytest.approx(0.0, abs=1e-6)
    )
    assert delivered < phi_target[0] - np.pi


# ---------------------------------------------------------------------------
# Bounds, threshold, dtype
# ---------------------------------------------------------------------------


def test_returns_float64():
    """The solver promotes to float64 internally and returns float64."""
    n_PS = 3
    x = solve_phase_power(
        phi_target=np.zeros(n_PS, dtype=np.float32),
        C_2=_diag_C2(n_PS).astype(np.float32),
        c_0=np.zeros(n_PS, dtype=np.float32),
    )
    assert x.dtype == np.float64


def test_v_is_strictly_in_bounds_after_random_solve():
    """No iteration ever leaves x outside [0, x_max] (clip after every step).

    Targets are constrained to be reachable inside x_max so that the
    converged x is in the open interior of [0, x_max], not just the
    boundary (which would trivially pass).
    """
    n_PS = 8
    rng = np.random.default_rng(7)
    x_max = 0.5 * X_MAX
    C_2 = _diag_C2(n_PS)
    c_0 = rng.uniform(-1.0, 1.0, size=n_PS)
    # Direct-reach phase swing at half budget is k * 0.7 W = 2*pi rad.
    phi_target = c_0 + rng.uniform(0.2, 2.0 * np.pi - 0.2, size=n_PS)

    x = solve_phase_power(phi_target, C_2, c_0, x_max=x_max, seed=0)
    assert np.all(x >= 0.0)
    assert np.all(x <= x_max)
    # Sanity: actually reached the interior.
    assert np.all(x > 0.0)
    assert np.all(x < x_max)


# ---------------------------------------------------------------------------
# Non-convergence: RuntimeWarning instead of silent failure
# ---------------------------------------------------------------------------


def test_non_convergence_emits_runtime_warning():
    """With max_iter = 1 and a non-trivial target, the solver warns."""
    C_2 = np.array([[K_NOMINAL]])
    c_0 = np.array([0.0])
    phi_target = np.array([1.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = solve_phase_power(
            phi_target, C_2, c_0,
            max_iter=1, reshuffle_every=0,
        )
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert x.shape == (1,)


def test_convergence_does_not_warn():
    """A trivially-converged problem emits no warning."""
    n_PS = 3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solve_phase_power(
            phi_target=np.zeros(n_PS),
            C_2=_diag_C2(n_PS),
            c_0=np.zeros(n_PS),
        )
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_phi_target_must_be_1d():
    with pytest.raises(ValueError, match="phi_target must be 1-D"):
        solve_phase_power(
            phi_target=np.zeros((3, 3)),
            C_2=np.zeros((3, 3)),
            c_0=np.zeros(3),
        )


def test_C_2_shape_mismatch():
    with pytest.raises(ValueError, match="C_2.shape"):
        solve_phase_power(
            phi_target=np.zeros(4),
            C_2=np.zeros((4, 5)),
            c_0=np.zeros(4),
        )


def test_c_0_shape_mismatch():
    with pytest.raises(ValueError, match="c_0.shape"):
        solve_phase_power(
            phi_target=np.zeros(4),
            C_2=np.zeros((4, 4)),
            c_0=np.zeros(3),
        )


@pytest.mark.parametrize("x_max", [0.0, -1.0])
def test_V_max_must_be_positive(x_max):
    with pytest.raises(ValueError, match="x_max"):
        solve_phase_power(
            phi_target=np.zeros(2),
            C_2=np.zeros((2, 2)),
            c_0=np.zeros(2),
            x_max=x_max,
        )


def test_threshold_must_be_positive():
    with pytest.raises(ValueError, match="threshold"):
        solve_phase_power(
            phi_target=np.zeros(2),
            C_2=np.zeros((2, 2)),
            c_0=np.zeros(2),
            threshold=0.0,
        )


def test_max_iter_must_be_positive():
    with pytest.raises(ValueError, match="max_iter"):
        solve_phase_power(
            phi_target=np.zeros(2),
            C_2=np.zeros((2, 2)),
            c_0=np.zeros(2),
            max_iter=0,
        )


def test_reshuffle_every_must_be_non_negative():
    with pytest.raises(ValueError, match="reshuffle_every"):
        solve_phase_power(
            phi_target=np.zeros(2),
            C_2=np.zeros((2, 2)),
            c_0=np.zeros(2),
            reshuffle_every=-1,
        )


def test_reshuffle_disabled_still_converges_on_easy_problem():
    """reshuffle_every = 0 must not crash and must still converge on a
    problem where reshuffles aren't needed."""
    n_PS = 4
    rng = np.random.default_rng(0)
    C_2 = _diag_C2(n_PS)
    c_0 = np.zeros(n_PS)
    phi_target = rng.uniform(0.2, 3.0, size=n_PS)
    x = solve_phase_power(
        phi_target, C_2, c_0,
        reshuffle_every=0, threshold=1e-4,
    )
    delta = _residual(phi_target, C_2, c_0, x)
    assert np.max(np.abs(delta)) < 1e-4


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_seed_reproducibility():
    """Same seed -> identical x on a problem whose path depends on reshuffles."""
    n_PS = 8
    rng = np.random.default_rng(123)
    C_2 = _diag_C2(n_PS)
    off = rng.uniform(-0.015 * K_NOMINAL, 0.015 * K_NOMINAL, size=(n_PS, n_PS))
    np.fill_diagonal(off, 0.0)
    C_2 = C_2 + off
    c_0 = rng.uniform(-np.pi, np.pi, size=n_PS)
    phi_target = c_0 + rng.uniform(-5.0, 5.0, size=n_PS)

    V_a = solve_phase_power(
        phi_target, C_2, c_0,
        max_iter=5000, reshuffle_every=100, seed=2024,
    )
    V_b = solve_phase_power(
        phi_target, C_2, c_0,
        max_iter=5000, reshuffle_every=100, seed=2024,
    )
    np.testing.assert_array_equal(V_a, V_b)


# ---------------------------------------------------------------------------
# Scale: full Clements m=12 size
# ---------------------------------------------------------------------------


def test_clements_m12_scale_converges():
    """n_PS = 132 matches the m=12 Clements mesh. With diagonal-only C_2
    and reachable targets the solver must still converge."""
    n_PS = 132
    rng = np.random.default_rng(2026)
    C_2 = _diag_C2(n_PS)
    c_0 = rng.uniform(-np.pi, np.pi, size=n_PS)
    phi_target = c_0 + rng.uniform(0.0, 4.0 * np.pi, size=n_PS)

    x = solve_phase_power(
        phi_target, C_2, c_0,
        threshold=1e-4, max_iter=20000, seed=0,
    )
    delta = _residual(phi_target, C_2, c_0, x)
    assert np.max(np.abs(delta)) < 1e-4
