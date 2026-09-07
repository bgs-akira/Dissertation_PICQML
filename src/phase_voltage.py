"""Iterative solver for the chip's phase-voltage law (paper Supplement H).

The chip obeys

    phi = C_2 @ (V ** 2) + c_0

which is affine in ``V ** 2``. A direct least-squares solve
``V = sqrt(solve(C_2, phi - c_0))`` exists mathematically but offers no recourse
when a component of ``V ** 2`` comes back negative or when a component of ``V``
exceeds ``V_max``. The iterative solver below handles both: it descends on the
phase residual while keeping ``V`` inside ``[0, V_max]``, and uses 2pi-wrapping
of the *residual* to exploit fringe periodicity when the target is out of the
direct reach of the diagonal of ``C_2``.

This module is intentionally NumPy-only and stands outside the autograd graph
(CLAUDE.md pitfall 13). Callers detach ``C_2`` and ``c_0`` from the trained
``DigitalTwin`` before passing them in.
"""

from __future__ import annotations

import warnings

import numpy as np


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles element-wise into ``[-pi, pi)``.

    Applied to the phase *difference* ``phi_now - phi_target`` only, never to
    the phase vectors themselves (CLAUDE.md pitfall 14): wrapping the two
    phases separately and then subtracting reintroduces 2pi jumps inside the
    residual and the solver fails to converge.
    """
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def solve_phase_voltage(
    phi_target: np.ndarray,    # (n_PS,) target phases
    C_2: np.ndarray,           # (n_PS, n_PS) fixed, from trained model
    c_0: np.ndarray,           # (n_PS,) current passive phases
    V_max: float = 15.0,
    threshold: float = 1e-4,   # 0.1 mrad
    max_iter: int = 10000,
    step_scale: float = 0.1,   # voltage step = delta * V_max * step_scale
    reshuffle_every: int = 500,
    *,
    seed: int | None = None,
) -> np.ndarray:               # (n_PS,) voltages in [0, V_max]
    """Iterative solver for ``phi = C_2 @ (V ** 2) + c_0`` (Supplement H).

    Given a target phase vector ``phi_target``, returns a voltage vector ``V``
    in ``[0, V_max]`` whose induced phases match ``phi_target`` modulo 2pi.

    Algorithm (Supplement H)::

        V = zeros(n_PS)
        repeat up to max_iter times:
            phi_now = C_2 @ V**2 + c_0
            delta   = wrap_to_pi(phi_now - phi_target)
            if max |delta| < threshold:  return V
            V       = V - step_scale * V_max * delta
            V       = clip(V, 0, V_max)
            every `reshuffle_every` iterations:  V += small_random_vector

    Args:
        phi_target:       (n_PS,) target phases, rad.
        C_2:              (n_PS, n_PS) crosstalk matrix from the trained
                          ``DigitalTwin``, rad / V**2. Detach to NumPy first.
        c_0:              (n_PS,) current passive phases, rad. Detach first.
        V_max:            voltage upper bound. Paper uses 14--15 V.
        threshold:        convergence target on ``max |delta|``. Paper uses
                          0.1 mrad = ``1e-4`` rad.
        max_iter:         hard cap on iterations. A ``RuntimeWarning`` is
                          emitted if convergence is not reached.
        step_scale:       voltage step per radian of phase error, expressed as
                          a fraction of ``V_max``. Paper uses ``V_max / 10``
                          (``step_scale = 0.1``).
        reshuffle_every:  every this many iterations, add a small Gaussian
                          perturbation to ``V`` to escape stagnated minima.
                          Pass ``0`` to disable.
        seed:             RNG seed for the reshuffle perturbations. Only
                          affects which local minimum is reached; the final
                          ``V`` still satisfies ``max |delta| < threshold``
                          on convergence.

    Returns:
        ``V`` of shape ``(n_PS,)`` in ``[0, V_max]``, dtype ``float64``.

    Raises:
        ValueError: if shapes are inconsistent or numerical bounds are
            non-positive.
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
    if V_max <= 0.0:
        raise ValueError(f"V_max must be positive, got {V_max}")
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if reshuffle_every < 0:
        raise ValueError(
            f"reshuffle_every must be >= 0, got {reshuffle_every}"
        )

    rng = np.random.default_rng(seed)
    step = V_max * step_scale
    gauss_scale = 0.05 * V_max

    V = np.zeros(n_PS, dtype=np.float64)
    max_abs = float("inf")

    for it in range(1, max_iter + 1):
        phi_now = C_2 @ (V * V) + c_0
        delta = _wrap_to_pi(phi_now - phi_target)
        max_abs = float(np.max(np.abs(delta)))

        if max_abs < threshold:
            return V

        V = V - step * delta
        np.clip(V, 0.0, V_max, out=V)

        if reshuffle_every > 0 and it % reshuffle_every == 0:
            # Wrap-induced local minima at V = 0 (or V = V_max) cannot be
            # escaped by gradient descent: the gradient pushes outward but
            # the clip pins V to the boundary. Randomize those components
            # uniformly into [0, V_max] -- this is the only kick wide enough
            # to land them past the next wrap boundary in a known fraction
            # of attempts. Non-stuck high-error components get a smaller
            # Gaussian nudge to escape shallow interior minima caused by
            # off-diagonal coupling.
            stuck = (
                ((V <= 0.0) & (delta > 0.0))
                | ((V >= V_max) & (delta < 0.0))
            )
            if stuck.any():
                V_rand = rng.uniform(0.0, V_max, size=n_PS)
                V = np.where(stuck, V_rand, V)
            else:
                V = V + rng.normal(scale=gauss_scale, size=n_PS)
                np.clip(V, 0.0, V_max, out=V)

    warnings.warn(
        f"solve_phase_voltage did not converge: max |delta| = {max_abs:.3e} "
        f"after {max_iter} iterations (threshold = {threshold:.3e}).",
        RuntimeWarning,
        stacklevel=2,
    )
    return V
