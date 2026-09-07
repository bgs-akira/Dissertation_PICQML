"""phi-IFM stage: refine c_0 with single-shifter phase-sweep fringes.

Sits between two ML stages. While the ML stage holds c_0 frozen and tunes
C_2, R, T_out by gradient descent, the phi-IFM stage holds C_2, R, T_out
fixed (at the values learned by the preceding ML stage) and refines c_0
by fitting one phase-sweep fringe per phase shifter (CLAUDE.md §7.5,
pitfall 16).

Per phase shifter i:
    measured fringe  f_meas(phi)        <- driven on the real chip, or
                                           synthesised from a ground-truth
                                           DigitalTwin for development
    model fringe     f_model(phi)        <- DigitalTwin evaluated with
                                           phi[i] set to the swept value
                                           and all other PS phases at 0
                                           (pitfall 15)
    fit  f_meas vs f_model(phi + dphi)  ->  dphi
    c_0[i]  <-  c_0[i] + dphi

The fits run in NumPy. The forward evaluation runs on the torch
DigitalTwin but is fully detached from the autograd graph (every call is
inside torch.no_grad): the c_0 update is not a differentiable step.
"""

from __future__ import annotations

import numpy as np
import torch

from src.model import DigitalTwin


# ---------------------------------------------------------------------------
# Forward evaluation at a chosen phi vector
# ---------------------------------------------------------------------------


@torch.no_grad()
def _intensity_at_phi(
    model: DigitalTwin,
    phi: np.ndarray | torch.Tensor,
    input_port: int,
    output_port: int,
) -> float:
    """Normalised intensity at output_port for a given phi vector.

    Bypasses the V -> phi conversion: phi is applied directly to the chip
    matrix. Uses the model's R, T_out and mesh topology; c_0 is *not*
    consulted (it would be overwritten by phi anyway).

    Implementation note: calls DigitalTwin._build_U, which is private. The
    alternative -- mutating model.c_0 in a try/finally -- leaves model
    state inconsistent if the caller interleaves evaluations or if an
    exception escapes the context. Reusing _build_U keeps the model state
    untouched.
    """
    device = model.c_0.device
    dtype = model._real_dtype
    if isinstance(phi, torch.Tensor):
        phi_t = phi.to(device=device, dtype=dtype)
    else:
        phi_t = torch.as_tensor(phi, device=device, dtype=dtype)
    U_0 = model._build_U(phi_t)
    sqrt_T_out = torch.sqrt(model.T_out).to(U_0.dtype)
    U = sqrt_T_out.unsqueeze(1) * U_0
    column = U[:, input_port]
    intensities = column.real ** 2 + column.imag ** 2
    return float((intensities[output_port] / intensities.sum()).item())


@torch.no_grad()
def _fringe_sweep(
    model: DigitalTwin,
    ps_index: int,
    phi_sweep: np.ndarray,
    input_port: int,
    output_port: int,
) -> np.ndarray:
    """Predicted monitored-output intensity along a single-PS phase sweep.

    Other PS phases are held at 0 (synthetic, no realistic routing --
    pitfall 15).
    """
    device = model.c_0.device
    dtype = model._real_dtype
    phi_vec = torch.zeros(model.n_PS, device=device, dtype=dtype)
    out = np.empty(len(phi_sweep), dtype=np.float64)
    for k, phi_k in enumerate(phi_sweep):
        phi_vec[ps_index] = float(phi_k)
        out[k] = _intensity_at_phi(model, phi_vec, input_port, output_port)
    return out


def _pick_informative_output_port(
    model: DigitalTwin,
    ps_index: int,
    input_port: int,
    n_probe: int = 8,
) -> int:
    """Pick the output port whose fringe varies the most with PS ps_index.

    A fixed (input_port, output_port) pair collapses to a flat fringe for
    PSs that have no path to output_port from input_port. Selecting the
    port with the largest probe-amplitude ensures every per-PS fringe
    carries information; pitfall 15 lets us pick ports freely on synthetic
    data.
    """
    probe = np.linspace(0.0, 2 * np.pi, n_probe, endpoint=False)
    best_port = 0
    best_amplitude = -np.inf
    for port in range(model.m):
        f = _fringe_sweep(model, ps_index, probe, input_port, port)
        amp = float(f.max() - f.min())
        if amp > best_amplitude:
            best_amplitude = amp
            best_port = port
    return best_port


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def generate_synthetic_phi_ifm(
    ground_truth: DigitalTwin,
    n_points: int = 15,
    noise_std: float = 1e-3,
    *,
    seed: int | None = None,
    input_port: int = 0,
) -> dict:
    """One fringe per phase shifter from a ground-truth DigitalTwin.

    For each PS i:
        phi_sweep      = linspace(0, 2pi, n_points)
        output_port[i] = port with maximum probe-fringe amplitude (so each
                         per-PS fringe is informative; pitfall 15)
        intensity[k]   = ground_truth's normalised intensity at
                         output_port[i] when phi[i] = phi_sweep[k] and all
                         other PS phases are zero, plus Gaussian noise.

    The fringe shape is a raised cosine in phi (paper supplement eq. 8) --
    the test suite verifies this on every PS.

    Args:
        ground_truth: source DigitalTwin with the "true" C_2, R, T_out, c_0.
        n_points:     samples per fringe along [0, 2pi]. Paper uses 15.
        noise_std:    Gaussian noise standard deviation added to each
                      intensity sample; set to 0 for deterministic data.
        seed:         RNG seed for the additive noise.
        input_port:   common input port for every fringe.

    Returns:
        Dict keyed by PS index (int) in [0, n_PS). Each value is a dict
        with keys "phi_sweep" (n_points,), "intensity_sweep" (n_points,),
        "input_port" (int), "output_port" (int).
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}")
    if not (0 <= input_port < ground_truth.m):
        raise ValueError(
            f"input_port must be in [0, {ground_truth.m}), got {input_port}"
        )

    rng = np.random.default_rng(seed)
    phi_sweep = np.linspace(0.0, 2 * np.pi, n_points)
    data: dict[int, dict] = {}
    for ps in range(ground_truth.n_PS):
        output_port = _pick_informative_output_port(
            ground_truth, ps, input_port
        )
        f = _fringe_sweep(
            ground_truth, ps, phi_sweep, input_port, output_port
        )
        if noise_std > 0.0:
            f = f + rng.normal(scale=noise_std, size=f.shape)
        data[ps] = {
            "phi_sweep": phi_sweep.copy(),
            "intensity_sweep": f.astype(np.float64),
            "input_port": int(input_port),
            "output_port": int(output_port),
        }
    return data


def model_fringe(
    model: DigitalTwin,
    ps_index: int,
    phi_sweep: np.ndarray,
    input_port: int,
    output_port: int,
) -> np.ndarray:
    """Predicted monitored-output intensity along the phase sweep.

    All PS phases other than ps_index are held at 0. Uses the model's
    current R, T_out and mesh (pitfall 16 -- c_0 is not consulted here,
    the swept phi directly sets the chip phases).
    """
    return _fringe_sweep(
        model, ps_index, np.asarray(phi_sweep, dtype=np.float64),
        input_port, output_port,
    )


# ---------------------------------------------------------------------------
# Fringe fits
# ---------------------------------------------------------------------------


def _wrap_to_pi_scalar(x: float) -> float:
    """Wrap a scalar phase offset into [-pi, pi)."""
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def _periodic_interp(
    x: np.ndarray, xp: np.ndarray, fp: np.ndarray,
) -> np.ndarray:
    """Linear periodic interpolation; both x and xp are wrapped into [0, 2pi)."""
    return np.interp(
        x % (2 * np.pi), xp % (2 * np.pi), fp, period=2 * np.pi,
    )


def _golden_section_search(
    f, lo: float, hi: float, *, tol: float = 1e-8, max_iter: int = 200,
) -> float:
    """Minimise unimodal f on [lo, hi] by golden section. Returns argmin."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - (b - a) * inv_phi
    d = a + (b - a) * inv_phi
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * inv_phi
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * inv_phi
            fd = f(d)
        if abs(b - a) < tol:
            break
    return 0.5 * (a + b)


def _minimise_on_circle(loss, n_grid: int = 72, tol: float = 1e-8) -> float:
    """Minimise loss(delta_phi) over delta_phi in [-pi, pi].

    The loss is generally multimodal in delta_phi (wrap period 2pi, and a
    raised-cosine fringe contributes another period-pi mode for the fast
    fit since a + b*f and a - b*f differ only by sign of b). A coarse grid
    scan over [-pi, pi] picks the basin; golden-section refines.
    """
    grid = np.linspace(-np.pi, np.pi, n_grid + 1)
    values = np.array([loss(x) for x in grid])
    k = int(values.argmin())
    lo = grid[max(0, k - 1)]
    hi = grid[min(n_grid, k + 1)]
    return _golden_section_search(loss, lo, hi, tol=tol)


def fit_fringe_fast(
    phi_sweep: np.ndarray,
    measured: np.ndarray,
    model_f: np.ndarray,
) -> float:
    """Three-parameter fit a + b * f(phi + dphi). Returns dphi in (-pi, pi].

    For each candidate dphi, the optimum (a, b) is the closed-form OLS
    solution against the design matrix [1, f_shifted]; dphi is then
    optimised over by grid scan + golden-section. The model fringe between
    grid points is reconstructed by 2pi-periodic linear interpolation.
    """
    phi_sweep = np.asarray(phi_sweep, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    model_f = np.asarray(model_f, dtype=np.float64)
    if phi_sweep.shape != measured.shape or phi_sweep.shape != model_f.shape:
        raise ValueError(
            "phi_sweep, measured and model_f must share shape; got "
            f"{phi_sweep.shape}, {measured.shape}, {model_f.shape}"
        )
    if phi_sweep.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 samples for the 3-parameter fit, "
            f"got {phi_sweep.shape[0]}"
        )

    # If the model fringe has no amplitude at this port pair, the phase
    # offset is unidentifiable: f(phi + dphi) is constant in dphi and any
    # dphi gives the same residual. Returning 0.0 leaves c_0 unchanged for
    # this PS, which is the right thing -- the caller picked a port pair
    # that doesn't see this PS. (For m=4 Clements with input_port=0, e.g.
    # PS 1 and PS 3 of MZI(2,3) are unreachable, so every output port
    # gives a flat fringe.)
    if model_f.max() - model_f.min() < 1e-10:
        return 0.0

    def residual(delta_phi: float) -> float:
        f_shifted = _periodic_interp(
            phi_sweep + delta_phi, phi_sweep, model_f,
        )
        A = np.column_stack([np.ones_like(f_shifted), f_shifted])
        sol, *_ = np.linalg.lstsq(A, measured, rcond=None)
        # Constraint: b >= 0. The fast fit has a sign ambiguity for any
        # fringe of the form A + B*cos(phi - phi_0):
        #   (a, b, dphi)  and  (a + 2*b*<f>, -b, dphi + pi)
        # give identical predictions. The dphi + pi branch flips b's sign,
        # so b >= 0 breaks the symmetry and pins the fit to the physical
        # branch (model and data both measure intensity in the same units).
        if sol[1] < 0:
            a_only = float(measured.mean())
            pred = np.full_like(measured, a_only)
        else:
            pred = A @ sol
        return float(((measured - pred) ** 2).sum())

    delta_phi = _minimise_on_circle(residual)
    return _wrap_to_pi_scalar(delta_phi)


def fit_fringe_precise(
    phi_sweep: np.ndarray,
    measured: np.ndarray,
    model: DigitalTwin,
    ps_index: int,
    input_port: int,
    output_port: int,
    *,
    dphi_min: float = 1e-3,
) -> float:
    """Single-parameter optimisation over dphi only. Returns dphi in (-pi, pi].

    The model fringe is regenerated from scratch at phi_sweep + dphi for
    every evaluation (no a/b rescaling -- the only degree of freedom is
    dphi). Slower than fit_fringe_fast but more accurate once the
    fast-method residual has stagnated (CLAUDE.md §7.5.7).

    Two robustness tweaks beyond the paper's plain "single-parameter L2":
    1. The residual is computed on DC-subtracted fringes (mean removed
       from both measured and model). This protects against the small
       residual DC mismatch caused by imperfect R, T_out at the moment of
       switchover. Amplitude differences still affect the residual scale
       but not the location of its minimum (in the continuous limit), so
       no further rescaling is added -- staying close to the paper's
       "single-parameter" definition.
    2. After the fit, |dphi| < dphi_min is returned as 0. This stops
       below-noise updates from random-walking c_0 once the protocol has
       essentially converged.
    """
    phi_sweep = np.asarray(phi_sweep, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if phi_sweep.shape != measured.shape:
        raise ValueError(
            "phi_sweep and measured must share shape; got "
            f"{phi_sweep.shape}, {measured.shape}"
        )

    # Same unidentifiability guard as fit_fringe_fast: if the model fringe
    # at this port pair is flat for all phases, no dphi is preferred over
    # any other and the grid scan returns a numerical artifact (often
    # ~+-pi). Returning 0.0 leaves c_0 unchanged for the PS.
    base_f = model_fringe(
        model, ps_index, phi_sweep, input_port, output_port,
    )
    if base_f.max() - base_f.min() < 1e-10:
        return 0.0

    measured_centered = measured - measured.mean()

    def residual(delta_phi: float) -> float:
        f = model_fringe(
            model, ps_index, phi_sweep + delta_phi,
            input_port, output_port,
        )
        f_centered = f - f.mean()
        return float(((measured_centered - f_centered) ** 2).sum())

    delta_phi = _wrap_to_pi_scalar(_minimise_on_circle(residual))
    if abs(delta_phi) < dphi_min:
        return 0.0
    return delta_phi


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_phi_ifm_stage(
    model: DigitalTwin,
    phi_ifm_data: dict,
    method: str = "fast",
) -> torch.Tensor:
    """Run one phi-IFM stage. Returns the updated c_0 tensor.

    Reads model.C_2, model.R, model.T_out at call time and never writes
    them (pitfall 16). Does not mutate model.c_0 either -- a *new* tensor
    is returned; the caller is responsible for writing it back to the
    model if desired (``model.c_0.copy_(new)``).

    Args:
        model:         DigitalTwin with C_2, R, T_out fixed by the
                       preceding ML stage. c_0 is read but not mutated.
        phi_ifm_data:  dict produced by generate_synthetic_phi_ifm or
                       loaded from disk. Keys are PS indices, values are
                       dicts with phi_sweep, intensity_sweep, input_port,
                       output_port.
        method:        "fast" -> fit_fringe_fast (3-parameter)
                       "precise" -> fit_fringe_precise (1-parameter)

    Returns:
        Tensor of shape (n_PS,), same dtype/device as model.c_0.
    """
    if method not in ("fast", "precise"):
        raise ValueError(
            f"method must be 'fast' or 'precise', got {method!r}"
        )

    c_0_new = model.c_0.detach().clone()
    for ps_index, fringe in phi_ifm_data.items():
        phi_sweep = np.asarray(fringe["phi_sweep"], dtype=np.float64)
        measured = np.asarray(fringe["intensity_sweep"], dtype=np.float64)
        input_port = int(fringe["input_port"])
        output_port = int(fringe["output_port"])

        if method == "fast":
            model_f = model_fringe(
                model, int(ps_index), phi_sweep,
                input_port, output_port,
            )
            delta_phi = fit_fringe_fast(phi_sweep, measured, model_f)
        else:
            delta_phi = fit_fringe_precise(
                phi_sweep, measured, model,
                int(ps_index), input_port, output_port,
            )
        c_0_new[int(ps_index)] = c_0_new[int(ps_index)] + float(delta_phi)
    return c_0_new
