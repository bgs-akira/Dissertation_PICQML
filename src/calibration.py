"""Simulated execution of the Prakash calibration protocol (m = 10).

This is the stage that replaces the fabricated ML seed. Previously the
initial guess was ``ground_truth + noise``; here it is FITTED from
simulated measurements, so nothing downstream gets to peek at the answer.

Why it must be sequential
-------------------------
Setting a router MZI to bar or cross requires already knowing that MZI's
``k`` and offset difference. The main diagonal needs no routers, so it can
be measured knowing nothing at all; calibrating diagonal ``k`` then
unlocks routing for the next one out. The pass therefore walks the
diagonals in protocol order, carrying a growing set of estimates, and
uses those estimates to drive the routers for later steps. That
dependency is the whole reason the document specifies an order.

What crosstalk does here
------------------------
Every sweep is driven in POWER against the ground-truth twin, so the
truth's full ``C_2`` -- including the off-diagonal -- maps powers to
phases. Heating one shifter therefore perturbs its neighbours exactly as
on hardware, and the fitted ``k`` and ``b`` carry that contamination. The
ML stage then has real residual structure to learn, rather than the
crosstalk-free fringes the old phi-IFM produced.

Scope: m = 10 only, per the chip this targets. The delta pass below is
closed-form in m and would work more widely, but nothing here is
maintained or tested for other sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.calibration_order import (
    ROUTER_DELTA,
    DeltaStep,
    delta_steps,
)
from src.chip_mesh import ChipMesh
from src.model import DigitalTwin
from src.power_lookup import K_NOMINAL
from src.phase_power import X_MAX


#: The chip this module is written for.
M = 10


def _require_m10(mesh: ChipMesh) -> None:
    if mesh.m != M or mesh.scheme != "bell":
        raise ValueError(
            f"src.calibration targets the m = {M} Bell chip only; got "
            f"scheme={mesh.scheme!r}, m={mesh.m}."
        )


# ---------------------------------------------------------------------------
# Measurement simulation
# ---------------------------------------------------------------------------


@torch.no_grad()
def monitored_intensity(
    truth: DigitalTwin,
    x_batch: torch.Tensor,
    input_port: int,
    output_port: int,
    reverse: bool,
) -> np.ndarray:
    """Normalised monitored intensity for a batch of power vectors.

    Args:
        truth:       ground-truth twin; its C_2 (crosstalk included) maps
                     the applied powers to phases.
        x_batch:     ``(batch, n_PS)`` heater powers, watts.
        input_port:  injection mode.
        output_port: monitored mode.
        reverse:     inject from the output facet. By reciprocity that
                     amplitude is a ROW of the chip matrix rather than a
                     column -- getting this wrong makes every reversed
                     diagonal read flat.

    Note that for a reversed measurement the ``T_out`` factor is common
    to the whole row, so it cancels under normalisation: reversed sweeps
    carry no output-transmission information. That is fine here, since
    ``T_out`` is recovered by the ML stage, not by this protocol.
    """
    phi = x_batch @ truth.C_2.T + truth.c_0
    U_0 = truth._build_U(phi)
    sqrt_T = torch.sqrt(truth.T_out).to(U_0.dtype)
    U = sqrt_T.view(1, -1, 1) * U_0
    vec = U[:, input_port, :] if reverse else U[:, :, input_port]
    inten = vec.real ** 2 + vec.imag ** 2
    p = inten[:, output_port] / inten.sum(dim=-1)
    return p.detach().cpu().numpy().astype(np.float64)


def router_power(
    k_up: float, k_lo: float, b_diff: float, delta: float,
    x_max: float = X_MAX,
) -> tuple[float, float]:
    """Powers that put an MZI at a target ``delta``.

    ``delta = (theta_up - theta_lo) / 2``, and with the lower arm left at
    zero power ``theta_up - theta_lo = k_up * x_up + b_diff`` where
    ``b_diff = b_up - b_lo``. Only the DIFFERENCE of the offsets is
    needed, which is exactly what the delta pass measures -- the sums
    (Sigma) are not required for routing.

    Returns ``(x_up, x_lo)``. Picks the smallest non-negative power in
    range, walking 2*pi branches until one fits the budget.
    """
    if k_up <= 0:
        raise ValueError(f"k_up must be positive, got {k_up}")
    target = 2.0 * delta - b_diff
    # Bring into [0, 2*pi) then step up by 2*pi until the power is in range.
    target = target % (2.0 * np.pi)
    for _ in range(8):
        x_up = target / k_up
        if 0.0 <= x_up <= x_max:
            return float(x_up), 0.0
        target += 2.0 * np.pi
    # Unreachable within the budget: clamp and let the caller see the error
    # show up as fringe distortion rather than silently failing.
    return float(min(max(target / k_up, 0.0), x_max)), 0.0


# ---------------------------------------------------------------------------
# Fringe fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmFit:
    """Result of sweeping one arm of one MZI.

    Attributes:
        k:        fitted phase-power slope, rad/W.
        phase:    fitted phase at zero power (the offset difference,
                  modulo sign and 2*pi).
        offset:   fitted DC level ``a``.
        amp:      fitted fringe amplitude ``c``.
        residual: RMS of the fit residual, normalised by ``amp``.
    """

    k: float
    phase: float
    offset: float
    amp: float
    residual: float


def fit_arm_sweep(
    x_sweep: np.ndarray,
    intensity: np.ndarray,
    *,
    k_lo: float = 0.4 * K_NOMINAL,
    k_hi: float = 1.8 * K_NOMINAL,
    n_grid: int = 400,
) -> ArmFit:
    """Fit ``P = a + c cos(k x + phase)`` to one arm sweep.

    The document's form is ``P = a + c sin(theta_up - theta_lo + pi/2)``,
    which is the same curve: sweeping the upper arm with the lower at zero
    power gives ``theta_up - theta_lo = k x + b_diff``, and
    ``sin(u + pi/2) = cos(u)``.

    Only ``k`` enters nonlinearly, so for each candidate ``k`` on a grid
    the remaining three parameters follow from an ordinary least squares
    against ``[1, cos(kx), sin(kx)]``. The grid minimum is then refined by
    a local golden-section search. This is far more robust than handing
    the whole four-parameter problem to a generic optimiser, which tends
    to lock onto a harmonic when the sweep spans several fringes.
    """
    x_sweep = np.asarray(x_sweep, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if x_sweep.shape != intensity.shape:
        raise ValueError(
            f"x_sweep {x_sweep.shape} and intensity {intensity.shape} "
            "must have the same shape"
        )
    if x_sweep.size < 4:
        raise ValueError("need at least 4 samples to fit 4 parameters")

    def solve(k: float):
        A = np.column_stack([
            np.ones_like(x_sweep), np.cos(k * x_sweep), np.sin(k * x_sweep),
        ])
        sol, *_ = np.linalg.lstsq(A, intensity, rcond=None)
        resid = float(np.sum((intensity - A @ sol) ** 2))
        return resid, sol

    grid = np.linspace(k_lo, k_hi, n_grid)
    residuals = np.array([solve(float(k))[0] for k in grid])
    best = int(residuals.argmin())

    # Golden-section refine inside the bracketing grid cell.
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a_lo = float(grid[max(0, best - 1)])
    b_hi = float(grid[min(n_grid - 1, best + 1)])
    c = b_hi - (b_hi - a_lo) * inv_phi
    d = a_lo + (b_hi - a_lo) * inv_phi
    fc, fd = solve(c)[0], solve(d)[0]
    for _ in range(80):
        if fc < fd:
            b_hi, d, fd = d, c, fc
            c = b_hi - (b_hi - a_lo) * inv_phi
            fc = solve(c)[0]
        else:
            a_lo, c, fc = c, d, fd
            d = a_lo + (b_hi - a_lo) * inv_phi
            fd = solve(d)[0]
        if abs(b_hi - a_lo) < 1e-10:
            break
    k_fit = 0.5 * (a_lo + b_hi)

    resid, sol = solve(k_fit)
    a, c_cos, c_sin = sol
    # a + c_cos cos(kx) + c_sin sin(kx) == a + amp cos(kx + phase)
    # with amp = hypot(c_cos, c_sin) and phase = atan2(-c_sin, c_cos).
    amp = float(np.hypot(c_cos, c_sin))
    phase = float(np.arctan2(-c_sin, c_cos))
    rms = float(np.sqrt(resid / x_sweep.size))
    return ArmFit(
        k=float(k_fit),
        phase=phase,
        offset=float(a),
        amp=amp,
        residual=rms / amp if amp > 0 else float("inf"),
    )


# ---------------------------------------------------------------------------
# The delta pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeltaResult:
    """Per-MZI outcome of the delta pass."""

    mzi: tuple[int, int]
    ps_up: int
    ps_lo: int
    k_up: float
    k_lo: float
    b_diff: float          # b_up - b_lo, radians
    residual_up: float
    residual_lo: float


def run_delta_pass(
    truth: DigitalTwin,
    *,
    n_points: int = 15,
    noise_std: float = 0.0,
    x_max: float = X_MAX,
    seed: int | None = None,
    progress: bool = False,
) -> dict[tuple[int, int], DeltaResult]:
    """Execute every delta step against ``truth`` and fit the fringes.

    Walks the diagonals in protocol order. Routers for a step are driven
    using estimates fitted on EARLIER diagonals only -- never using
    ``truth`` -- so the pass is a faithful bootstrap rather than a lookup.

    Args:
        truth:     ground-truth twin standing in for the chip.
        n_points:  samples per fringe. The document uses 15.
        noise_std: Gaussian noise on each intensity sample (shot noise).
        x_max:     per-channel power ceiling, watts.
        seed:      RNG seed for the measurement noise.
        progress:  print per-diagonal progress.

    Returns:
        Dict keyed by MZI label with the fitted slopes and offset
        difference for both arms.
    """
    mesh = truth.mesh
    _require_m10(mesh)
    rng = np.random.default_rng(seed)
    dtype = truth.c_0.dtype
    device = truth.c_0.device

    results: dict[tuple[int, int], DeltaResult] = {}
    x_sweep = np.linspace(0.0, x_max, n_points)
    current_k = None

    for step in delta_steps(mesh):
        if progress and step.diagonal != current_k:
            current_k = step.diagonal
            print(f"  diagonal k={step.diagonal} "
                  f"(in {step.input_port} -> out {step.output_port}"
                  f"{', reverse' if step.reverse else ''})", flush=True)

        # Base power vector: every router held at its state using only
        # estimates from already-calibrated diagonals.
        base = torch.zeros(mesh.n_PS, dtype=dtype, device=device)
        for mzi, state in step.routers.items():
            prev = results.get(mzi)
            if prev is None:
                raise RuntimeError(
                    f"step on MZI {step.mzi} routes through {mzi}, which "
                    "has not been calibrated yet -- the protocol order is "
                    "inconsistent"
                )
            x_up, x_lo = router_power(
                prev.k_up, prev.k_lo, prev.b_diff,
                ROUTER_DELTA[state], x_max,
            )
            base[prev.ps_up] = x_up
            base[prev.ps_lo] = x_lo

        ps_up, ps_lo = step.swept_ps
        fits: list[ArmFit] = []
        for swept in (ps_up, ps_lo):
            batch = base.unsqueeze(0).repeat(n_points, 1)
            batch[:, swept] = torch.as_tensor(
                x_sweep, dtype=dtype, device=device
            )
            inten = monitored_intensity(
                truth, batch,
                step.input_port, step.output_port, step.reverse,
            )
            if noise_std > 0:
                inten = inten + rng.normal(scale=noise_std, size=inten.shape)
            fits.append(fit_arm_sweep(x_sweep, inten))

        fit_up, fit_lo = fits
        # Sweeping the upper arm gives phase = +b_diff; sweeping the lower
        # arm gives phase = -b_diff (the difference changes sign). Average
        # the two independent estimates.
        b_diff = float(
            np.arctan2(
                np.sin(fit_up.phase) - np.sin(fit_lo.phase),
                np.cos(fit_up.phase) + np.cos(fit_lo.phase),
            )
        )
        results[step.mzi] = DeltaResult(
            mzi=step.mzi,
            ps_up=ps_up,
            ps_lo=ps_lo,
            k_up=fit_up.k,
            k_lo=fit_lo.k,
            b_diff=b_diff,
            residual_up=fit_up.residual,
            residual_lo=fit_lo.residual,
        )

    return results
