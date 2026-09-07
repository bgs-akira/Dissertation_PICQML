"""Shared synthetic-experiment scaffolding used by every ``scripts/run_*``
and ``scripts/sweep_*`` entry point.

What lives here:

1. ``n_samples_from_ratio`` -- single source of truth for the
   parameter:data sizing rule used across every sweep and runner.
2. ``ground_truth_model`` -- the canonical ground-truth ``DigitalTwin``
   with paper-magnitude perturbations. Used by ``run_synthetic.py``,
   ``run_iterative.py`` and ``run_phi_ifm.py``; previously copy-pasted
   three times with magic constants that had to be kept in sync by hand.
3. Chip-response cache helpers (``build_chip_response_cache``,
   ``alpha_beta``, ``analytical_fringe``, ``all_port_amplitudes``,
   ``best_port_pair``). For a single-PS phase sweep the chip column
   ``U[:, i](phi)`` is affine in ``exp(i*phi)``; pre-computing
   ``(U_0, U_pi[k])`` once turns every downstream fringe evaluation
   into a cheap tensor expression. Drops the port-pair scan on m=10
   from ~9 min to ~1 s.
4. ``make_phi_ifm_data`` -- experiment-mimicking phi-IFM dataset
   generator. The operator drives ``phi_sweep[k]`` at PS i using their
   own ``c_0_model``; the chip actually applies
   ``phi_sweep[k] + (c_0_gt - c_0_model)[i]``, so each measured fringe
   is shifted by exactly that offset. The fit recovers the shift.

Keeping these in one place means future changes to the synthetic
recipe (different perturbation magnitudes, alternative routing
strategies, ...) propagate without touching every CLI script.
"""

from __future__ import annotations

import numpy as np
import torch

from src.chip_mesh import ChipMesh
from src.model import DigitalTwin


# ---------------------------------------------------------------------------
# Dataset sizing
# ---------------------------------------------------------------------------


def n_samples_from_ratio(m: int, ratio: float, test_frac: float = 0.20) -> int:
    """Return total ``n_samples`` such that the *training* split contains
    ``round(ratio * n_params)`` examples under an 80/20 train/test split.

    ``ratio`` is the parameter:data ratio applied to the **training set**:
    ``n_train = round(ratio * n_params)``. The test set is then sized from
    the same 80/20 rule used by the runner (``test_frac=0.20`` matches
    ``TEST_FRAC`` in ``scripts/run_iterative.py``), giving
    ``n_test = round(n_train * test_frac / (1 - test_frac))``.

    ``n_PS`` and ``n_BS`` are both ``m * (m - 1)`` for a Clements
    rectangular mesh (CLAUDE.md section 3.5).
    """
    n_PS = m * (m - 1)
    n_BS = m * (m - 1)
    n_params = n_PS * n_PS + n_BS + m
    n_train = round(ratio * n_params)
    n_test = round(n_train * test_frac / (1.0 - test_frac))
    return max(2, n_train + n_test)


# ---------------------------------------------------------------------------
# Ground-truth DigitalTwin
# ---------------------------------------------------------------------------

# Magic-number perturbations -- match what the three runner scripts used
# verbatim before this module existed. Edit here, propagate everywhere.
_TRUTH_C0_STD = 0.30          # rad
_TRUTH_C2_DIAG_MEAN = 0.034   # rad / V^2  (paper midpoint)
_TRUTH_C2_DIAG_STD = 2e-3     # rad / V^2  (~6% of mean)
_TRUTH_C2_OFFDIAG_STD = 5e-4  # rad / V^2  (paper-typical crosstalk magnitude)
_TRUTH_R_LOGIT_STD = 0.30     # logit space; sigmoid(0.3) ~= 0.575
_TRUTH_T_LOGIT_MEAN = 6.0     # sigmoid(6.0) ~= 0.998
_TRUTH_T_LOGIT_STD = 0.10


def ground_truth_model(
    m: int, seed: int, *, dtype: torch.dtype = torch.float64,
) -> DigitalTwin:
    """Canonical ground-truth ``DigitalTwin`` for synthetic experiments.

    All three of ``run_synthetic.py``, ``run_iterative.py`` and
    ``run_phi_ifm.py`` use this; runs across scripts are therefore
    comparable as long as ``(m, seed, dtype)`` match.
    """
    mesh = ChipMesh.clements(m)
    g = torch.Generator().manual_seed(seed)
    c_0 = torch.randn(mesh.n_PS, generator=g, dtype=dtype) * _TRUTH_C0_STD
    c2_diag = (
        torch.full((mesh.n_PS,), _TRUTH_C2_DIAG_MEAN, dtype=dtype)
        + torch.randn(mesh.n_PS, generator=g, dtype=dtype) * _TRUTH_C2_DIAG_STD
    )
    model = DigitalTwin(mesh, c_0, c2_diag, dtype=dtype)
    with torch.no_grad():
        model.C_2_raw.add_(
            torch.randn(mesh.n_PS, mesh.n_PS, generator=g, dtype=dtype)
            * _TRUTH_C2_OFFDIAG_STD
        )
        model.R_logit.copy_(
            torch.randn(mesh.n_BS, generator=g, dtype=dtype)
            * _TRUTH_R_LOGIT_STD
        )
        model.T_logit.copy_(
            torch.full((m,), _TRUTH_T_LOGIT_MEAN, dtype=dtype)
            + torch.randn(m, generator=g, dtype=dtype) * _TRUTH_T_LOGIT_STD
        )
    return model


# ---------------------------------------------------------------------------
# Chip-response cache (analytical fringe)
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_chip_response_cache(
    model: DigitalTwin,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute ``(U_0, U_pi)`` for analytical single-PS fringe evaluation.

    Returns:
        ``U_0``    : ``(m, m)`` complex. Chip matrix with all phases at 0.
        ``U_pi``   : ``(n_PS, m, m)`` complex. ``U_pi[k]`` is the chip
                     matrix with PS k at pi and every other PS at 0.

    Cost: ``1 + n_PS`` calls to ``model._build_U``. Every downstream
    fringe is an analytical tensor expression after this.
    """
    device = model.c_0.device
    dtype = model._real_dtype
    n_PS = model.n_PS
    m = model.m

    phi_zero = torch.zeros(n_PS, dtype=dtype, device=device)
    U_0 = model._build_U(phi_zero)

    U_pi = torch.empty(
        (n_PS, m, m), dtype=model._complex_dtype, device=device,
    )
    phi_vec = torch.zeros(n_PS, dtype=dtype, device=device)
    for k in range(n_PS):
        phi_vec[k] = np.pi
        U_pi[k] = model._build_U(phi_vec)
        phi_vec[k] = 0.0
    return U_0, U_pi


@torch.no_grad()
def alpha_beta(
    U_0: torch.Tensor, U_pi_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose the single-PS-swept column as
    ``U(phi) = alpha + beta * exp(i * phi)``. Returns ``(alpha, beta)``."""
    return 0.5 * (U_0 + U_pi_k), 0.5 * (U_0 - U_pi_k)


@torch.no_grad()
def analytical_fringe(
    alpha: torch.Tensor, beta: torch.Tensor,
    T_out: torch.Tensor,
    phi_sweep: np.ndarray,
    input_port: int, output_port: int,
) -> np.ndarray:
    """Normalised intensity at ``output_port`` for a single-PS phase sweep,
    computed without further ``_build_U`` calls."""
    a = alpha[:, input_port]
    b = beta[:, input_port]
    phi_t = torch.as_tensor(
        phi_sweep, dtype=a.real.dtype, device=a.device,
    )
    eiphi = torch.exp(1j * phi_t)
    U_col = a.unsqueeze(-1) + b.unsqueeze(-1) * eiphi.unsqueeze(0)
    intensities = (
        (U_col.real ** 2 + U_col.imag ** 2)
        * T_out.to(U_col.real.dtype).unsqueeze(-1)
    )
    p = intensities / intensities.sum(dim=0, keepdim=True)
    return p[output_port].cpu().numpy().astype(np.float64)


@torch.no_grad()
def all_port_amplitudes(
    alpha: torch.Tensor, beta: torch.Tensor,
    T_out: torch.Tensor, n_probe: int = 16,
) -> torch.Tensor:
    """Per-(input, output) fringe amplitudes for one swept PS.

    Returns a ``(m_out, m_in)`` tensor of ``max_phi(p) - min_phi(p)`` for
    the normalised intensity. The probe grid is dense enough that the
    raised-cosine extrema are bracketed to within ~1 % on every realistic
    chip.
    """
    probe = torch.linspace(
        0.0, 2 * np.pi, n_probe + 1,
        dtype=alpha.real.dtype, device=alpha.device,
    )[:-1]
    eiphi = torch.exp(1j * probe)
    U_col = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * eiphi.view(1, 1, -1)
    intensities = (
        (U_col.real ** 2 + U_col.imag ** 2)
        * T_out.to(U_col.real.dtype).view(-1, 1, 1)
    )
    p = intensities / intensities.sum(dim=0, keepdim=True)
    return p.max(dim=-1).values - p.min(dim=-1).values


def best_port_pair(
    U_0: torch.Tensor, U_pi_k: torch.Tensor, T_out: torch.Tensor,
) -> tuple[int, int, float]:
    """Pick the ``(input, output)`` port pair maximising the analytical
    fringe amplitude for the swept PS. Returns ``(input, output, amp)``."""
    a, b = alpha_beta(U_0, U_pi_k)
    amps = all_port_amplitudes(a, b, T_out)            # (m_out, m_in)
    flat_idx = int(torch.argmax(amps).item())
    m_in = amps.shape[1]
    j, i = divmod(flat_idx, m_in)
    return int(i), int(j), float(amps[j, i].item())


# ---------------------------------------------------------------------------
# phi-IFM data: experiment-mimicking synthesis
# ---------------------------------------------------------------------------


def make_phi_ifm_data(
    truth: DigitalTwin,
    U_0_truth: torch.Tensor,
    U_pi_truth: torch.Tensor,
    c_0_model: torch.Tensor,
    *,
    n_points: int,
    noise_std: float,
    seed: int,
) -> tuple[dict[int, dict], np.ndarray]:
    """Generate phi-IFM data reflecting the current ``c_0_model`` error.

    In the real experiment, the operator targets ``phi_sweep[k]`` at PS i
    by solving V via ``c_0_model``; the chip actually applies
    ``phi_sweep[k] + (truth.c_0 - c_0_model)[i]``. So every fringe is the
    ground truth's response shifted by exactly that offset. The fit
    recovers the shift and pushes ``c_0`` toward truth.

    Returns:
        ``data``    : dict keyed by PS index, each value a dict with
                      ``phi_sweep``, ``intensity_sweep``, ``input_port``,
                      ``output_port``.
        ``offsets`` : ``(n_PS,)`` ndarray of the true per-PS phase offsets
                      ``c_0_gt - c_0_model``. ``run_iterative.py`` ignores
                      this; ``run_phi_ifm.py`` reports recovery error
                      against it.
    """
    rng = np.random.default_rng(seed)
    phi_sweep = np.linspace(0.0, 2 * np.pi, n_points)
    offsets = (truth.c_0 - c_0_model).detach().cpu().numpy()
    T_out_truth = truth.T_out.detach()

    data: dict[int, dict] = {}
    for ps in range(truth.n_PS):
        input_port, output_port, _amp = best_port_pair(
            U_0_truth, U_pi_truth[ps], T_out_truth,
        )
        a, b = alpha_beta(U_0_truth, U_pi_truth[ps])
        intensity = analytical_fringe(
            a, b, T_out_truth,
            phi_sweep + offsets[ps],
            input_port, output_port,
        )
        if noise_std > 0:
            intensity = intensity + rng.normal(
                scale=noise_std, size=intensity.shape
            )
        data[ps] = {
            "phi_sweep": phi_sweep.copy(),
            "intensity_sweep": intensity.astype(np.float64),
            "input_port": int(input_port),
            "output_port": int(output_port),
        }
    return data, offsets
