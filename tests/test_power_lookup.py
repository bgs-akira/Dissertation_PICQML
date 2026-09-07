"""Tests for the current <-> power conversion.

The chip budget from the calibration document (section 4):
    100 mA, 15 V, R ~ 140 ohm cold, ~700 mW per 2*pi of phase.
"""

import numpy as np
import pytest

from src.power_lookup import (
    ALPHA_R,
    I_MAX,
    K_NOMINAL,
    POWER_PER_2PI,
    R_COLD,
    PowerLookup,
    max_power,
    power_from_current,
)


# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------


def test_k_nominal_matches_the_stated_budget():
    """~700 mW for 2*pi gives k ~ 8.98 rad/W."""
    assert K_NOMINAL == pytest.approx(2 * np.pi / POWER_PER_2PI)
    assert K_NOMINAL == pytest.approx(8.976, abs=1e-3)


def test_current_limit_binds_before_the_voltage_limit():
    """15 V across a ~140 ohm heater would need ~107 mA, above the 100 mA cap.

    So the usable power span is set by I_MAX, not V_MAX.
    """
    assert I_MAX * R_COLD < 15.0


def test_max_power_spans_about_two_fringes():
    """The budget reaches ~1.4 W, i.e. about 4*pi of phase."""
    x_max = max_power()
    assert x_max == pytest.approx(1.4, abs=0.15)
    assert K_NOMINAL * x_max == pytest.approx(4 * np.pi, rel=0.15)


# ---------------------------------------------------------------------------
# power_from_current
# ---------------------------------------------------------------------------


def test_zero_current_is_zero_power():
    assert power_from_current(np.array([0.0]))[0] == 0.0


def test_matches_i_squared_r_when_cold():
    """With no resistance drift the law collapses to x = I**2 R."""
    i = np.array([0.02, 0.05, 0.1])
    np.testing.assert_allclose(
        power_from_current(i, R_COLD, alpha_r=0.0), i ** 2 * R_COLD,
    )


def test_heating_raises_power_above_the_cold_estimate():
    """A warm heater has higher resistance, so dissipates more at fixed I."""
    i = np.array([I_MAX])
    cold = power_from_current(i, R_COLD, alpha_r=0.0)[0]
    warm = power_from_current(i, R_COLD, alpha_r=ALPHA_R)[0]
    assert warm > cold
    # The chosen ALPHA_R should be a modest correction, not a runaway.
    assert warm / cold == pytest.approx(1.0, abs=0.25)


def test_monotonic_in_current():
    i = np.linspace(0.0, I_MAX, 50)
    x = power_from_current(i)
    assert np.all(np.diff(x) > 0)


def test_self_consistency_of_the_heating_model():
    """x solves x = I**2 * R_0 * (1 + alpha_r * x), by construction."""
    i = np.array([0.03, 0.07, I_MAX])
    x = power_from_current(i, R_COLD, ALPHA_R)
    np.testing.assert_allclose(
        x, i ** 2 * R_COLD * (1.0 + ALPHA_R * x), rtol=1e-12,
    )


def test_negative_current_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        power_from_current(np.array([-0.01]))


def test_thermal_runaway_rejected():
    """alpha_r * I**2 * R_0 >= 1 has no steady state; say so, don't return junk."""
    with pytest.raises(ValueError, match="runaway"):
        power_from_current(np.array([I_MAX]), R_COLD, alpha_r=10.0)


# ---------------------------------------------------------------------------
# PowerLookup
# ---------------------------------------------------------------------------


def test_synthetic_shapes():
    lut = PowerLookup.synthetic(100, n_points=64, seed=0)
    assert lut.n_PS == 100
    assert lut.currents.shape == (100, 64)
    assert lut.powers.shape == (100, 64)


def test_round_trip_current_to_power_and_back():
    """The inverse map is the whole point: solver returns x, instrument needs I."""
    lut = PowerLookup.synthetic(20, seed=1)
    i = np.linspace(0.0, I_MAX, 20)
    x = lut.power(i)
    back = lut.current(x)
    np.testing.assert_allclose(back, i, atol=1e-6)


def test_round_trip_power_to_current_and_back():
    lut = PowerLookup.synthetic(8, seed=2)
    x = np.linspace(0.0, lut.max_power.min(), 8)
    i = lut.current(x)
    np.testing.assert_allclose(lut.power(i), x, atol=1e-6)


def test_batched_query():
    lut = PowerLookup.synthetic(5, seed=3)
    batch = np.tile(np.linspace(0.0, I_MAX, 5), (4, 1))
    out = lut.power(batch)
    assert out.shape == (4, 5)
    np.testing.assert_allclose(out[0], out[3])


def test_channels_differ_because_resistances_differ():
    """Per-channel scatter is real: the same current gives different powers."""
    lut = PowerLookup.synthetic(50, r_spread=0.05, seed=4)
    x = lut.power(np.full(50, I_MAX))
    assert x.std() > 0
    assert x.std() / x.mean() == pytest.approx(0.05, abs=0.03)


def test_uniform_channels_when_spread_is_zero():
    lut = PowerLookup.synthetic(10, r_spread=0.0, seed=5)
    x = lut.power(np.full(10, I_MAX))
    np.testing.assert_allclose(x, x[0])


def test_out_of_range_query_rejected():
    lut = PowerLookup.synthetic(4, seed=6)
    with pytest.raises(ValueError, match="out of tabulated range"):
        lut.power(np.full(4, 2.0 * I_MAX))
    with pytest.raises(ValueError, match="out of tabulated range"):
        lut.current(np.full(4, 100.0))


def test_wrong_width_rejected():
    lut = PowerLookup.synthetic(4, seed=7)
    with pytest.raises(ValueError, match=r"shape \(n_PS,\)"):
        lut.power(np.zeros(3))


def test_non_monotonic_table_rejected():
    currents = np.array([[0.0, 0.05, 0.02]])
    powers = np.array([[0.0, 0.1, 0.2]])
    with pytest.raises(ValueError, match="strictly ascending"):
        PowerLookup(currents, powers)


def test_mismatched_table_shapes_rejected():
    with pytest.raises(ValueError, match="same shape"):
        PowerLookup(np.zeros((2, 5)), np.zeros((2, 6)))


def test_single_point_table_rejected():
    with pytest.raises(ValueError, match="at least 2 points"):
        PowerLookup(np.zeros((2, 1)), np.zeros((2, 1)))
