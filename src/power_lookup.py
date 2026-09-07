"""Current <-> electric-power conversion for the thermo-optic heaters.

Why this module exists (Prakash calibration document, section 4)
---------------------------------------------------------------
The chip's phase law is linear in electric POWER, ``theta = k*x + b``, so
the digital twin is driven by ``x`` (watts). But the instrument sets a
CURRENT. Naively ``x = I**2 * R`` with ``R ~ 140 ohm``, except a metal
heater's resistance rises as it warms, which makes ``R`` a function of
``I`` and ``x`` a polynomial in ``I`` of degree greater than two. The
document notes that inverting that polynomial analytically is awkward,
and instead records ``(I, x)`` pairs during the calibration sweeps and
interpolates. This module is that lookup table.

It is deliberately OUTSIDE the autograd graph. The twin never sees
current; conversion happens at the instrument boundary, exactly as in the
document. Keeping it here means the nonlinearity is not smeared into
``C_2``, where it would be mistaken for crosstalk.

Chip budget (m = 10 Prakash)
----------------------------
    I_max = 100 mA          per-channel current limit
    V_max = 15 V            per-channel voltage limit
    R_0   ~ 140 ohm         cold resistance
    ~700 mW  -> 2*pi of phase shift

The current limit binds first: 15 V across 140 ohm would need 107 mA. So
the usable span is ``x`` in ``[0, I_max**2 * R] ~ [0, 1.4 W]``, about two
full fringes, and ``k = 2*pi / 0.7 ~ 8.98 rad/W``.
"""

from __future__ import annotations

import numpy as np


#: Per-channel current limit, amperes.
I_MAX = 0.100
#: Per-channel voltage limit, volts.
V_MAX = 15.0
#: Cold resistance of a heater, ohms (document section 4).
R_COLD = 140.0
#: Electric power for a 2*pi phase shift, watts (document section 4).
POWER_PER_2PI = 0.700
#: Phase-power slope k = 2*pi / POWER_PER_2PI, rad/W. Seeds diag(C_2).
K_NOMINAL = 2.0 * np.pi / POWER_PER_2PI
#: Fractional resistance rise per watt dissipated. Chosen so a channel at
#: I_MAX runs ~10 % above its cold resistance, the right order for a metal
#: heater; the document fits a 4th-degree polynomial to the real curve.
ALPHA_R = 0.07


def max_power(r_cold: float = R_COLD, i_max: float = I_MAX,
              v_max: float = V_MAX, alpha_r: float = ALPHA_R) -> float:
    """Largest power reachable within both the current and voltage limits."""
    x_i = power_from_current(np.array([i_max]), r_cold, alpha_r)[0]
    # V = I * R(x); the voltage ceiling caps current at v_max / R.
    r_hot = r_cold * (1.0 + alpha_r * x_i)
    i_from_v = v_max / r_hot
    if i_from_v < i_max:
        return float(power_from_current(np.array([i_from_v]), r_cold,
                                        alpha_r)[0])
    return float(x_i)


def power_from_current(
    current: np.ndarray,
    r_cold: float = R_COLD,
    alpha_r: float = ALPHA_R,
) -> np.ndarray:
    """Electric power dissipated at a given current, accounting for heating.

    Solves ``x = I**2 * R_0 * (1 + alpha_r * x)`` for ``x``, i.e. the
    self-consistent steady state in which the resistance has already risen
    by the power it is dissipating. Closed form::

        x = I**2 R_0 / (1 - alpha_r I**2 R_0)

    Args:
        current: amperes, any shape.
        r_cold:  cold resistance, ohms.
        alpha_r: fractional resistance rise per watt.

    Returns:
        Power in watts, same shape as ``current``.
    """
    current = np.asarray(current, dtype=np.float64)
    if np.any(current < 0):
        raise ValueError("current must be non-negative")
    cold_power = current ** 2 * r_cold
    denom = 1.0 - alpha_r * cold_power
    if np.any(denom <= 0):
        raise ValueError(
            "thermal runaway: alpha_r * I**2 * R_0 >= 1 for some current; "
            "reduce alpha_r or the current range"
        )
    return cold_power / denom


class PowerLookup:
    """Per-shifter current <-> power table with interpolated inversion.

    One table per phase shifter, because each heater has its own
    resistance. Built either from measured sweeps (the real case: the
    document records ``(I, x)`` pairs while calibrating ``k`` and ``b``)
    or synthetically via :meth:`synthetic`.

    Attributes:
        currents: ``(n_PS, n_points)`` ascending, amperes.
        powers:   ``(n_PS, n_points)`` ascending, watts.
    """

    def __init__(self, currents: np.ndarray, powers: np.ndarray) -> None:
        currents = np.asarray(currents, dtype=np.float64)
        powers = np.asarray(powers, dtype=np.float64)
        if currents.ndim != 2 or powers.ndim != 2:
            raise ValueError(
                f"currents and powers must be 2-D (n_PS, n_points); got "
                f"{currents.shape} and {powers.shape}"
            )
        if currents.shape != powers.shape:
            raise ValueError(
                f"currents {currents.shape} and powers {powers.shape} must "
                "have the same shape"
            )
        if currents.shape[1] < 2:
            raise ValueError("need at least 2 points per table to interpolate")
        if np.any(np.diff(currents, axis=1) <= 0):
            raise ValueError("each row of currents must be strictly ascending")
        if np.any(np.diff(powers, axis=1) <= 0):
            raise ValueError("each row of powers must be strictly ascending")
        self.currents = currents
        self.powers = powers

    @property
    def n_PS(self) -> int:
        return self.currents.shape[0]

    @property
    def max_power(self) -> np.ndarray:
        """(n_PS,) largest power each channel can reach."""
        return self.powers[:, -1]

    @classmethod
    def synthetic(
        cls,
        n_PS: int,
        *,
        n_points: int = 256,
        r_cold: float = R_COLD,
        r_spread: float = 0.05,
        alpha_r: float = ALPHA_R,
        i_max: float = I_MAX,
        seed: int | None = None,
    ) -> "PowerLookup":
        """Build synthetic tables with per-channel resistance scatter.

        Args:
            n_PS:     number of phase shifters.
            n_points: samples per table.
            r_cold:   nominal cold resistance, ohms.
            r_spread: fractional 1-sigma scatter of cold resistance across
                      channels; fabrication is not perfectly uniform.
            alpha_r:  fractional resistance rise per watt.
            i_max:    top of the current sweep, amperes.
            seed:     RNG seed for the resistance scatter.
        """
        if n_PS < 1:
            raise ValueError(f"n_PS must be >= 1, got {n_PS}")
        rng = np.random.default_rng(seed)
        r = r_cold * (1.0 + rng.normal(scale=r_spread, size=n_PS))
        grid = np.linspace(0.0, i_max, n_points)
        currents = np.tile(grid, (n_PS, 1))
        powers = np.stack(
            [power_from_current(grid, float(r_k), alpha_r) for r_k in r]
        )
        return cls(currents, powers)

    def power(self, current: np.ndarray) -> np.ndarray:
        """Interpolate power from current, per shifter.

        Args:
            current: ``(n_PS,)`` or ``(batch, n_PS)`` amperes.

        Returns:
            Power in watts, same shape.
        """
        return self._interp(current, self.currents, self.powers, "current")

    def current(self, power: np.ndarray) -> np.ndarray:
        """Interpolate current from power, per shifter -- the inverse map.

        This is the direction the instrument needs: the solver returns a
        target power vector, and this says what to actually set.

        Args:
            power: ``(n_PS,)`` or ``(batch, n_PS)`` watts.

        Returns:
            Current in amperes, same shape.
        """
        return self._interp(power, self.powers, self.currents, "power")

    def _interp(
        self,
        query: np.ndarray,
        xp: np.ndarray,
        fp: np.ndarray,
        what: str,
    ) -> np.ndarray:
        query = np.asarray(query, dtype=np.float64)
        single = query.ndim == 1
        q = query[None, :] if single else query
        if q.ndim != 2 or q.shape[1] != self.n_PS:
            raise ValueError(
                f"{what} must have shape (n_PS,) or (batch, n_PS) with "
                f"n_PS={self.n_PS}; got {query.shape}"
            )
        lo = xp[:, 0]
        hi = xp[:, -1]
        if np.any(q < lo[None, :] - 1e-12) or np.any(q > hi[None, :] + 1e-12):
            raise ValueError(
                f"{what} out of tabulated range; each channel supports "
                f"[{lo.min():.4g}, {hi.max():.4g}]"
            )
        out = np.empty_like(q)
        for k in range(self.n_PS):
            out[:, k] = np.interp(q[:, k], xp[k], fp[k])
        return out[0] if single else out
