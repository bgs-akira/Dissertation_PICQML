"""Tests for the phi-IFM stage.

Covers:
  - _wrap_to_pi_scalar (the per-PS update is wrapped before being added),
  - model_fringe periodicity and consistency with the synthetic generator,
  - generate_synthetic_phi_ifm: data structure, noise behaviour, and the
    eq. 8 raised-cosine shape (paper supplement),
  - fit_fringe_fast / fit_fringe_precise: recover delta_phi = 0 on
    clean matched data, and recover a known phase shift,
  - run_phi_ifm_stage: returns a fresh tensor, does NOT mutate model.c_0
    or the trained parameters (pitfall 16), recovers per-PS phase offsets,
  - rejection of invalid arguments.

The mesh is kept small (m = 4) so the tests run in seconds; the algorithm
is mesh-size-agnostic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.chip_mesh import ChipMesh
from src.model import DigitalTwin
from src.phi_ifm import (
    _wrap_to_pi_scalar,
    fit_fringe_fast,
    fit_fringe_precise,
    generate_synthetic_phi_ifm,
    model_fringe,
    run_phi_ifm_stage,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_ground_truth(
    m: int = 4,
    c_0_value: float = 0.5,
    c2_diag_value: float = 0.034,
) -> DigitalTwin:
    """Construct a small DigitalTwin to play the role of ground truth.

    The default R = 0.5 and uniform T_out keep the fringe a pure raised
    cosine after normalisation (a non-uniform T_out makes the normalised
    intensity a ratio of raised cosines and would defeat the eq. 8 sanity
    check).
    """
    mesh = ChipMesh.clements(m)
    c_0 = torch.full((mesh.n_PS,), c_0_value, dtype=torch.float64)
    c2_diag = torch.full((mesh.n_PS,), c2_diag_value, dtype=torch.float64)
    return DigitalTwin(mesh, c_0, c2_diag, dtype=torch.float64)


def _fit_raised_cosine(
    phi: np.ndarray, intensity: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit I = A + B*cos(phi - phi_0) by linear regression on (1, cos, sin).

    Returns (A, B, phi_0, residual_rms).
    """
    X = np.column_stack([
        np.ones_like(phi), np.cos(phi), np.sin(phi),
    ])
    sol, *_ = np.linalg.lstsq(X, intensity, rcond=None)
    A, alpha, beta = sol
    B = float(np.hypot(alpha, beta))
    phi_0 = float(np.arctan2(beta, alpha))
    residual_rms = float(
        np.sqrt(((intensity - X @ sol) ** 2).mean())
    )
    return float(A), B, phi_0, residual_rms


# ---------------------------------------------------------------------------
# _wrap_to_pi_scalar
# ---------------------------------------------------------------------------


def test_wrap_to_pi_scalar_in_range():
    """Values in (-pi, pi) are unchanged."""
    for x in [-1.0, 0.0, 1.0, np.pi - 1e-3]:
        assert _wrap_to_pi_scalar(x) == pytest.approx(x, abs=1e-12)


def test_wrap_to_pi_scalar_2pi_shift_invariant():
    """f(x + 2pi) == f(x) within float tolerance."""
    for x in [-2.5, -1.0, 0.0, 1.0, 2.5]:
        assert _wrap_to_pi_scalar(x) == pytest.approx(
            _wrap_to_pi_scalar(x + 2 * np.pi), abs=1e-12,
        )


def test_wrap_to_pi_scalar_endpoint():
    """+pi is wrapped to the lower edge -pi by the standard formula."""
    assert _wrap_to_pi_scalar(np.pi) == pytest.approx(-np.pi, abs=1e-12)


# ---------------------------------------------------------------------------
# model_fringe
# ---------------------------------------------------------------------------


def test_model_fringe_shape_and_dtype():
    """Returns (n_points,) float64."""
    model = _make_ground_truth()
    f = model_fringe(model, ps_index=0, phi_sweep=np.linspace(0, 2 * np.pi, 11),
                     input_port=0, output_port=1)
    assert f.shape == (11,)
    assert f.dtype == np.float64


def test_model_fringe_periodic():
    """Intensity at phi = 0 and phi = 2pi must match (chip is exp(i*phi))."""
    model = _make_ground_truth()
    f = model_fringe(model, 0, np.array([0.0, 2 * np.pi]), 0, 1)
    assert f[0] == pytest.approx(f[1], abs=1e-10)


def test_model_fringe_intensity_in_unit_interval():
    """Normalised intensity at one output port is in [0, 1]."""
    model = _make_ground_truth()
    f = model_fringe(model, 0, np.linspace(0, 2 * np.pi, 21), 0, 1)
    assert np.all(f >= -1e-12)
    assert np.all(f <= 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# generate_synthetic_phi_ifm
# ---------------------------------------------------------------------------


def test_synthetic_dict_structure():
    """Keys = [0, n_PS), each value has phi_sweep, intensity_sweep, ports."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, n_points=15, noise_std=0.0)
    assert set(data.keys()) == set(range(gt.n_PS))
    for fringe in data.values():
        assert set(fringe.keys()) == {
            "phi_sweep", "intensity_sweep", "input_port", "output_port",
        }
        assert fringe["phi_sweep"].shape == (15,)
        assert fringe["intensity_sweep"].shape == (15,)
        assert 0 <= fringe["input_port"] < gt.m
        assert 0 <= fringe["output_port"] < gt.m


def test_synthetic_phi_sweep_endpoints():
    """phi_sweep spans [0, 2pi] inclusive (n_points samples)."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, n_points=15, noise_std=0.0)
    fringe = data[0]
    assert fringe["phi_sweep"][0] == pytest.approx(0.0)
    assert fringe["phi_sweep"][-1] == pytest.approx(2 * np.pi)


def test_synthetic_periodic_when_noise_zero():
    """Intensity at the matching endpoints agrees (modulo numerical noise)."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, n_points=15, noise_std=0.0)
    for fringe in data.values():
        assert fringe["intensity_sweep"][0] == pytest.approx(
            fringe["intensity_sweep"][-1], abs=1e-10,
        )


def test_synthetic_noise_zero_is_deterministic():
    """noise_std = 0 should produce reproducible data without an RNG seed."""
    gt = _make_ground_truth()
    a = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    b = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    for i in a:
        np.testing.assert_array_equal(
            a[i]["intensity_sweep"], b[i]["intensity_sweep"]
        )


def test_synthetic_noise_nonzero_changes_data_but_seed_is_reproducible():
    """noise_std > 0 produces different fringes; same seed is reproducible."""
    gt = _make_ground_truth()
    clean = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    noisy_a = generate_synthetic_phi_ifm(gt, noise_std=1e-2, seed=42)
    noisy_b = generate_synthetic_phi_ifm(gt, noise_std=1e-2, seed=42)
    noisy_c = generate_synthetic_phi_ifm(gt, noise_std=1e-2, seed=43)
    # Reproducibility under fixed seed.
    np.testing.assert_array_equal(
        noisy_a[0]["intensity_sweep"], noisy_b[0]["intensity_sweep"]
    )
    # Different from clean (noise added).
    assert not np.allclose(
        clean[0]["intensity_sweep"], noisy_a[0]["intensity_sweep"]
    )
    # Different seed -> different realisation.
    assert not np.allclose(
        noisy_a[0]["intensity_sweep"], noisy_c[0]["intensity_sweep"]
    )


def test_synthetic_fringe_is_raised_cosine():
    """Each PS's fringe satisfies I = A + B*cos(phi - phi_0) (supplement
    eq. 8). Verified by linear regression on (1, cos phi, sin phi)."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, n_points=21, noise_std=0.0)
    informative = 0
    for fringe in data.values():
        _, B, _, rms = _fit_raised_cosine(
            fringe["phi_sweep"], fringe["intensity_sweep"]
        )
        # Residual RMS should be tiny relative to fringe amplitude.
        # Skip near-flat fringes (port choice without information).
        if B < 1e-3:
            continue
        informative += 1
        assert rms < 1e-6 * max(B, 1.0), (
            f"raised-cosine fit residual {rms:.2e} too large for B={B:.3e}"
        )
    # The output-port picker should produce at least one informative
    # fringe; in practice it picks the most informative for every PS.
    assert informative >= 1


def test_synthetic_matches_model_fringe_when_noise_zero():
    """At noise_std = 0 the synthetic intensities match model_fringe
    exactly when evaluated on the same ground-truth model."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    for i, fringe in data.items():
        f = model_fringe(
            gt, i, fringe["phi_sweep"],
            fringe["input_port"], fringe["output_port"],
        )
        np.testing.assert_allclose(
            fringe["intensity_sweep"], f, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# fit_fringe_fast
# ---------------------------------------------------------------------------


def test_fit_fast_zero_when_measured_equals_model():
    """measured == model_f -> dphi very close to 0."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, n_points=21, noise_std=0.0)
    for i, fringe in data.items():
        model_f = model_fringe(
            gt, i, fringe["phi_sweep"],
            fringe["input_port"], fringe["output_port"],
        )
        # Skip flat fringes -- the fit is ill-conditioned without amplitude.
        if model_f.max() - model_f.min() < 1e-3:
            continue
        dphi = fit_fringe_fast(fringe["phi_sweep"], model_f, model_f)
        assert abs(dphi) < 1e-4, f"PS {i}: dphi = {dphi:.3e}, expected ~0"


@pytest.mark.parametrize("true_offset", [-1.0, -0.3, 0.2, 0.7, 1.2])
def test_fit_fast_recovers_known_offset(true_offset):
    """measured = f(phi + true_offset) -> fit recovers true_offset.

    Uses the per-PS informative port pair chosen by
    generate_synthetic_phi_ifm so the fringe has visible amplitude. (A
    hardcoded (input=0, output=1) gives a flat fringe for PS 0 on a m=4
    Clements mesh: all amplitude flows to ports 2 and 3.)
    """
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    phi_sweep = np.linspace(0.0, 2 * np.pi, 21)
    ps = 0
    input_port = data[ps]["input_port"]
    output_port = data[ps]["output_port"]
    base_f = model_fringe(gt, ps, phi_sweep, input_port, output_port)
    assert base_f.max() - base_f.min() > 1e-3
    shifted_f = model_fringe(
        gt, ps, phi_sweep + true_offset, input_port, output_port,
    )
    dphi = fit_fringe_fast(phi_sweep, shifted_f, base_f)
    assert -np.pi <= dphi <= np.pi
    expected = _wrap_to_pi_scalar(true_offset)
    assert dphi == pytest.approx(expected, abs=2e-3)


def test_fit_fast_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="share shape"):
        fit_fringe_fast(
            np.zeros(5), np.zeros(5), np.zeros(4),
        )


def test_fit_fast_rejects_too_few_samples():
    with pytest.raises(ValueError, match="at least 3"):
        fit_fringe_fast(
            np.zeros(2), np.zeros(2), np.zeros(2),
        )


def test_fit_fast_wrap_range():
    """Returned dphi must lie in [-pi, pi]."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    ps = 0
    input_port = data[ps]["input_port"]
    output_port = data[ps]["output_port"]
    phi_sweep = np.linspace(0.0, 2 * np.pi, 21)
    base_f = model_fringe(gt, ps, phi_sweep, input_port, output_port)
    # 5.5 wraps to 5.5 - 2*pi ~ -0.78
    shifted_f = model_fringe(
        gt, ps, phi_sweep + 5.5, input_port, output_port,
    )
    dphi = fit_fringe_fast(phi_sweep, shifted_f, base_f)
    assert -np.pi <= dphi <= np.pi


# ---------------------------------------------------------------------------
# fit_fringe_precise
# ---------------------------------------------------------------------------


def test_fit_precise_zero_when_measured_matches_model():
    """measured = model_fringe(model, ...) -> dphi very close to 0.

    Iterates over every PS using each PS's informative port pair (the
    pair chosen by generate_synthetic_phi_ifm to maximise fringe
    amplitude).
    """
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    phi_sweep = np.linspace(0.0, 2 * np.pi, 15)
    for ps in range(gt.n_PS):
        ip = data[ps]["input_port"]
        op = data[ps]["output_port"]
        base_f = model_fringe(gt, ps, phi_sweep, ip, op)
        if base_f.max() - base_f.min() < 1e-3:
            continue
        dphi = fit_fringe_precise(
            phi_sweep, base_f, gt, ps, input_port=ip, output_port=op,
        )
        assert abs(dphi) < 1e-4, f"PS {ps}: dphi = {dphi:.3e}"


@pytest.mark.parametrize("true_offset", [-0.7, -0.2, 0.4, 1.0])
def test_fit_precise_recovers_known_offset(true_offset):
    """measured = f(phi + true_offset) -> fit recovers true_offset."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    phi_sweep = np.linspace(0.0, 2 * np.pi, 21)
    ps = 0
    input_port = data[ps]["input_port"]
    output_port = data[ps]["output_port"]
    base_amplitude = model_fringe(
        gt, ps, phi_sweep, input_port, output_port,
    )
    assert base_amplitude.max() - base_amplitude.min() > 1e-3
    measured = model_fringe(
        gt, ps, phi_sweep + true_offset, input_port, output_port,
    )
    dphi = fit_fringe_precise(
        phi_sweep, measured, gt, ps, input_port, output_port,
    )
    assert dphi == pytest.approx(_wrap_to_pi_scalar(true_offset), abs=1e-3)


def test_fit_precise_rejects_shape_mismatch():
    gt = _make_ground_truth()
    with pytest.raises(ValueError, match="share shape"):
        fit_fringe_precise(
            np.zeros(5), np.zeros(4), gt, 0, 0, 0,
        )


# ---------------------------------------------------------------------------
# run_phi_ifm_stage
# ---------------------------------------------------------------------------


def test_stage_returns_tensor_with_right_shape_and_dtype():
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    c_0_new = run_phi_ifm_stage(gt, data)
    assert isinstance(c_0_new, torch.Tensor)
    assert c_0_new.shape == (gt.n_PS,)
    assert c_0_new.dtype == gt.c_0.dtype


def test_stage_does_not_mutate_model_c_0():
    """The stage must return a fresh tensor; model.c_0 stays as-is."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    c_0_before = gt.c_0.detach().clone()
    _ = run_phi_ifm_stage(gt, data)
    torch.testing.assert_close(gt.c_0, c_0_before)


def test_stage_does_not_mutate_c2_r_tout():
    """Pitfall 16: only c_0 may change. C_2, R, T_out are read-only."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    C_2_before = gt.C_2.detach().clone()
    R_before = gt.R.detach().clone()
    T_before = gt.T_out.detach().clone()
    _ = run_phi_ifm_stage(gt, data)
    torch.testing.assert_close(gt.C_2, C_2_before)
    torch.testing.assert_close(gt.R, R_before)
    torch.testing.assert_close(gt.T_out, T_before)


def test_stage_clean_data_yields_small_update():
    """measured matches the model -> recovered c_0 stays near original."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    c_0_new = run_phi_ifm_stage(gt, data, method="fast")
    delta = (c_0_new - gt.c_0).abs().max().item()
    assert delta < 1e-3


def test_stage_recovers_injected_offsets_fast():
    """Inject a known per-PS phase shift into the synthetic data, then
    verify run_phi_ifm_stage recovers it via the fast method.

    The mechanism: each PS i has phi_sweep -> intensity. If on the real
    experiment the user's V is computed with a c_0_model that is offset
    by `delta_i` from c_0_gt, then the actual phase achieved is
    phi_sweep + delta_i and the measured intensity follows
    f_gt(phi_sweep + delta_i). The fit then recovers `delta_i`.
    """
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)

    rng = np.random.default_rng(0)
    true_offsets = rng.uniform(-1.0, 1.0, size=gt.n_PS)

    for ps, fringe in data.items():
        shifted = model_fringe(
            gt, ps, fringe["phi_sweep"] + true_offsets[ps],
            fringe["input_port"], fringe["output_port"],
        )
        fringe["intensity_sweep"] = shifted

    c_0_new = run_phi_ifm_stage(gt, data, method="fast")
    recovered_delta = (c_0_new - gt.c_0).cpu().numpy()

    # Only PSs with informative fringes can be recovered; restrict the
    # assertion to those.
    for ps in range(gt.n_PS):
        amp = (
            data[ps]["intensity_sweep"].max()
            - data[ps]["intensity_sweep"].min()
        )
        if amp < 1e-2:
            continue
        expected = _wrap_to_pi_scalar(float(true_offsets[ps]))
        assert recovered_delta[ps] == pytest.approx(expected, abs=5e-3), (
            f"PS {ps}: recovered {recovered_delta[ps]:.4e}, "
            f"expected {expected:.4e}"
        )


def test_stage_recovers_injected_offsets_precise():
    """Same as the fast-method test but using method='precise'."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)

    rng = np.random.default_rng(1)
    true_offsets = rng.uniform(-0.8, 0.8, size=gt.n_PS)

    for ps, fringe in data.items():
        shifted = model_fringe(
            gt, ps, fringe["phi_sweep"] + true_offsets[ps],
            fringe["input_port"], fringe["output_port"],
        )
        fringe["intensity_sweep"] = shifted

    c_0_new = run_phi_ifm_stage(gt, data, method="precise")
    recovered_delta = (c_0_new - gt.c_0).cpu().numpy()

    for ps in range(gt.n_PS):
        amp = (
            data[ps]["intensity_sweep"].max()
            - data[ps]["intensity_sweep"].min()
        )
        if amp < 1e-2:
            continue
        expected = _wrap_to_pi_scalar(float(true_offsets[ps]))
        assert recovered_delta[ps] == pytest.approx(expected, abs=1e-3)


def test_stage_fast_and_precise_agree_on_clean_data():
    """On noiseless matched data, both methods should recover near-zero
    updates and agree with each other to high precision."""
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    c_0_fast = run_phi_ifm_stage(gt, data, method="fast")
    c_0_precise = run_phi_ifm_stage(gt, data, method="precise")
    torch.testing.assert_close(c_0_fast, c_0_precise, atol=2e-3, rtol=0.0)


def test_stage_rejects_invalid_method():
    gt = _make_ground_truth()
    data = generate_synthetic_phi_ifm(gt, noise_std=0.0)
    with pytest.raises(ValueError, match="method must be"):
        run_phi_ifm_stage(gt, data, method="medium")


def test_stage_with_noise_recovers_offsets_approximately():
    """Small Gaussian noise on the data still allows recovery within a
    looser tolerance set by noise_std and fringe amplitude."""
    gt = _make_ground_truth()
    rng = np.random.default_rng(7)
    true_offsets = rng.uniform(-0.5, 0.5, size=gt.n_PS)

    data = generate_synthetic_phi_ifm(gt, n_points=21, noise_std=0.0)
    for ps, fringe in data.items():
        shifted = model_fringe(
            gt, ps, fringe["phi_sweep"] + true_offsets[ps],
            fringe["input_port"], fringe["output_port"],
        )
        fringe["intensity_sweep"] = (
            shifted + rng.normal(scale=1e-3, size=shifted.shape)
        )

    c_0_new = run_phi_ifm_stage(gt, data, method="fast")
    recovered_delta = (c_0_new - gt.c_0).cpu().numpy()
    for ps in range(gt.n_PS):
        amp = (
            data[ps]["intensity_sweep"].max()
            - data[ps]["intensity_sweep"].min()
        )
        if amp < 1e-2:
            continue
        expected = _wrap_to_pi_scalar(float(true_offsets[ps]))
        assert recovered_delta[ps] == pytest.approx(expected, abs=2e-2)


# ---------------------------------------------------------------------------
# Argument validation on generate_synthetic_phi_ifm
# ---------------------------------------------------------------------------


def test_generate_rejects_n_points_too_small():
    gt = _make_ground_truth()
    with pytest.raises(ValueError, match="n_points"):
        generate_synthetic_phi_ifm(gt, n_points=1)


def test_generate_rejects_negative_noise_std():
    gt = _make_ground_truth()
    with pytest.raises(ValueError, match="noise_std"):
        generate_synthetic_phi_ifm(gt, noise_std=-1.0)


def test_generate_rejects_out_of_range_input_port():
    gt = _make_ground_truth()
    with pytest.raises(ValueError, match="input_port"):
        generate_synthetic_phi_ifm(gt, input_port=gt.m)
