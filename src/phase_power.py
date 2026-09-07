"""Iterative solver for the chip's phase-power law.

The chip obeys

    phi = C_2 @ x + c_0

where ``x`` is the electric power dissipated in each heater (watts),
``diag(C_2)`` is the per-shifter slope ``k`` (rad/W) and ``c_0`` is the
offset ``b`` (rad) -- the calibration document's ``theta = k*x + b``,
extended off-diagonal to carry thermal crosstalk.

Unlike the earlier voltage form (``phi = C_2 @ V**2 + c_0``) this is
LINEAR in the drive variable, so a direct solve ``x = solve(C_2, phi -
c_0)`` exists and is well conditioned. We still iterate, for three
reasons the direct solve cannot handle:

  1. ``x`` is bounded: every channel lives in ``[0, x_max]`` set by the
     100 mA / 15 V budget. A direct solve happily returns negative or
     out-of-range powers.
  2. Phases are 2*pi-periodic, so the target is a lattice of solutions,
     not a point. Wrapping the RESIDUAL lets the solver fall into
     whichever lattice point is reachable inside the power budget.
  3. Clipping at a bound can strand a channel in a local minimum that
     only a random restart escapes.

Linearity does buy a much better step than the original scheme's ``x -=
step * delta``: since ``dphi_k/dx_k = C_2[k, k]``, dividing the phase
residual by the diagonal gives a Jacobi step that is exact for an
uncoupled chip and converges in a handful of iterations with crosstalk.

NumPy only, and deliberately outside the autograd graph (CLAUDE.md
pitfall 13). Callers detach ``C_2`` and ``c_0`` from the trained
``DigitalTwin`` before passing them in.
"""

from __future__ import annotations

import warnings

import numpy as np

from src.power_lookup import max_power


#: Default power ceiling per channel, watts (src.power_lookup).
X_MAX = max_power()


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles element-wise into ``[-pi, pi)``.

    Applied to the phase *difference* ``phi_now - phi_target`` only, never
    to the phase vectors themselves (CLAUDE.md pitfall 14): wrapping the
    two phases separately and then subtracting reintroduces 2pi jumps
    inside the residual and the solver fails to converge.
    """
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def solve_phase_power(
    phi_target: np.ndarray,    # (n_PS,) target phases, rad
    C_2: np.ndarray,           # (n_PS, n_PS) fixed, from trained model
    c_0: np.ndarray,           # (n_PS,) current passive phases, rad
    x_max: float = X_MAX,
    threshold: float = 1e-4,   # 0.1 mrad
    max_iter: int = 10000,
    relax: float = 1.0,
    reshuffle_every: int = 500,
    *,
    seed: int | None = None,
) -> np.ndarray:               # (n_PS,) powers in [0, x_max]
    """Solve ``phi = C_2 @ x + c_0`` for the power vector ``x``.

    Returns powers in ``[0, x_max]`` whose induced phases match
    ``phi_target`` modulo 2*pi.

    Algorithm::

        x = zeros(n_PS)
        repeat up to max_iter times:
            phi_now = C_2 @ x + c_0
            delta   = wrap_to_pi(phi_now - phi_target)
            if max |delta| < threshold:  return x
            x       = x - relax * delta / diag(C_2)     # Jacobi step
            x       = clip(x, 0, x_max)
            every `reshuffle_every` iterations: randomise stuck channels

    Args:
        phi_target:       (n_PS,) target phases, rad.
        C_2:              (n_PS, n_PS) from the trained ``DigitalTwin``,
                          rad/W. Detach to NumPy first. Its diagonal must
                          be non-zero -- a zero-slope channel cannot be
                          driven to a phase at all.
        c_0:              (n_PS,) current passive phases, rad. Detach first.
        x_max:            per-channel power ceiling, watts.
        threshold:        convergence target on ``max |delta|``. The
                          document's precision is 0.1 mrad = ``1e-4`` rad.
        max_iter:         hard cap. A ``RuntimeWarning`` is emitted if
                          convergence is not reached.
        relax:            Jacobi relaxation factor. ``1.0`` is the exact
                          Newton step for an uncoupled chip; lower it
                          towards ~0.5 if strong crosstalk makes the
                          iteration ring.
        reshuffle_every:  every this many iterations, randomise channels
                          pinned at a bound with residual still pointing
                          outward. Pass ``0`` to disable.
        seed:             RNG seed for the reshuffles. Only affects which
                          lattice point is reached; the returned ``x``
                          still satisfies ``max |delta| < threshold``.

    Returns:
        ``x`` of shape ``(n_PS,)`` in ``[0, x_max]``, dtype float64.

    Raises:
        ValueError: on inconsistent shapes, non-positive bounds, or a
            zero on the diagonal of ``C_2``.
    """
    phi_target = np.asarray(phi_target, dtype=np.float64)
    C_2 = np.asarray(C_2, dtype=np.float64)
    c_0 = np.asarray(c_0, dtype=np.float64)

    if phi_target.ndim != 1:
        raise ValueError(
            f"phi_target must be 1-D, got shape {phi_target.shape}"
        )
    n_PS = phi_target.shape[0]
    if C_2.shape != (n_PS, n_PS):
        raise ValueError(
            f"C_2.shape must be ({n_PS}, {n_PS}), got {C_2.shape}"
        )
    if c_0.shape != (n_PS,):
        raise ValueError(f"c_0.shape must be ({n_PS},), got {c_0.shape}")
    if x_max <= 0.0:
        raise ValueError(f"x_max must be positive, got {x_max}")
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if relax <= 0.0:
        raise ValueError(f"relax must be positive, got {relax}")
    if reshuffle_every < 0:
        raise ValueError(
            f"reshuffle_every must be >= 0, got {reshuffle_every}"
        )

    slope = np.diag(C_2).copy()
    if np.any(slope == 0.0):
        dead = np.flatnonzero(slope == 0.0)
        raise ValueError(
            f"C_2 has a zero diagonal at indices {dead.tolist()}; those "
            "channels have no phase response and cannot be solved for"
        )

    rng = np.random.default_rng(seed)
    x = np.zeros(n_PS, dtype=np.float64)
    max_abs = float("inf")

    for it in range(1, max_iter + 1):
        phi_now = C_2 @ x + c_0
        delta = _wrap_to_pi(phi_now - phi_target)
        max_abs = float(np.max(np.abs(delta)))

        if max_abs < threshold:
            return x

        x = x - relax * delta / slope
        np.clip(x, 0.0, x_max, out=x)

        if reshuffle_every > 0 and it % reshuffle_every == 0:
            # A channel pinned at a bound whose residual still pushes it
            # further out cannot escape by descent: the step points
            # outward and the clip pins it back. Randomising it uniformly
            # across the range is the only kick wide enough to land it
            # past the next 2*pi wrap. Channels stuck in a shallow
            # interior minimum (from off-diagonal coupling) get a smaller
            # nudge instead.
            outward = np.sign(delta) * np.sign(slope)
            stuck = ((x <= 0.0) & (outward > 0)) | ((x >= x_max) & (outward < 0))
            if stuck.any():
                x = np.where(stuck, rng.uniform(0.0, x_max, size=n_PS), x)
            else:
                x = x + rng.normal(scale=0.05 * x_max, size=n_PS)
                np.clip(x, 0.0, x_max, out=x)

    warnings.warn(
        f"solve_phase_power did not converge: max |delta| = {max_abs:.3e} "
        f"after {max_iter} iterations (threshold = {threshold:.3e}).",
        RuntimeWarning,
        stacklevel=2,
    )
    return x
