"""End-to-end (ML + phi-IFM) outer loop until convergence.

Replicates the iterative protocol from Fyrillas et al. on synthetic data:

    V-IFM seeds  ->  ML  ->  phi-IFM  ->  ML  ->  phi-IFM  ->  ...

CLAUDE.md §6.5 specifies the per-cycle schedule:
    - 500 epochs per ML stage (default; CLI-tunable),
    - one phi-IFM stage between successive ML stages,
    - all learning rates multiplied by 0.7 after each cycle,
    - exit when the next ML stage's best test MSE no longer improves.

This script implements that schedule and additionally exposes an explicit
**TVD threshold gate** (default 1e-3 = paper's synthetic target). The
loop stops as soon as either:
    (a) post-phi-IFM test TVD falls below --threshold, or
    (b) the new ML stage's best test MSE >= the previous cycle's best
        (no further improvement, paper's own criterion), or
    (c) --max-cycles iterations are reached.

Per-cycle statistics, the V-IFM-seed initial state, and the final
truth/recovered parameter tensors are saved to ``--out`` (default
``outputs/iterative_m<m>.json``). A sidecar ``.log`` file captures the
console output. Plotting lives in ``notebooks/exploration.ipynb``.

Usage::

    uv run python scripts/run_iterative.py [--m M] [--threshold T]
                                            [--max-cycles N] [--epochs E]
                                            [--n-samples K]
                                            [--out outputs/iterative_m<m>.json]

The defaults are tuned for m=4 to finish in a few minutes on a laptop.
For paper-size (m=12) bump --epochs and --n-samples accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.chip_mesh import MESH_SCHEMES
from src.cli import setup_logging
from src.data import X_MAX, make_synthetic_dataset, train_test_split
from src.evaluate import evaluate
from src.model import DigitalTwin
from src.phi_ifm import run_phi_ifm_stage
from src.power_lookup import K_NOMINAL
from src.synthetic import (
    build_chip_response_cache,
    ground_truth_model,
    make_phi_ifm_data,
    n_samples_from_ratio,
)
from src.training import train


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ML stage defaults (CLAUDE.md §6.2). LRs are decayed by 0.7 per cycle.
#
# lr_C2 is expressed RELATIVE to the scale of C_2 itself, because Adam's
# per-step update is ~lr in parameter units (it normalises away the
# gradient magnitude). What matters is therefore lr / |C_2|, not lr.
#
# The paper's 1e-5 was tuned against a voltage-driven C_2 whose diagonal
# sat at ~0.034 rad/V**2, i.e. a relative step of ~2.9e-4 per iteration.
# Driving in power puts the diagonal at K_NOMINAL ~ 8.98 rad/W instead --
# 264x larger -- so carrying 1e-5 across unchanged would shrink the
# effective step by that same factor and the ML stage would crawl.
# Keeping the RELATIVE rate fixed is what actually transfers.
LR_C2_RELATIVE = 2.9e-4
LR_C2_INIT = LR_C2_RELATIVE * K_NOMINAL   # ~2.6e-3 rad/W per step
# R and T_out are sigmoid logits -- dimensionless, and unaffected by the
# change of drive variable -- so the paper's rates carry over as they are.
LR_R_INIT = 1e-3
LR_TOUT_INIT = 1e-3
LR_DECAY = 0.7

# Synthetic dataset
TEST_FRAC = 0.20
BATCH_SIZE = 64
DATA_NOISE_STD = 0.0   # noise on (x, port, p) tuples; 0 = clean synthetic

# V-IFM seed errors (the noise that phi-IFM is asked to clean up)
C_0_SEED_NOISE = 0.30      # rad
C_2_DIAG_SEED_NOISE = 0.06 * K_NOMINAL  # rad / W (~6% of k)

# phi-IFM measurements
PHI_IFM_N_POINTS = 15
# Synthetic shot-noise on intensity samples. 0 by default: this is a
# protocol-convergence study, not a hardware emulation. Fresh noise on every
# cycle was injecting a random walk into c_0 once the precise fringe-fit
# took over (post-ML TVD drifting back up after ~5 cycles). Bump to 1e-3
# to study the algorithm's sensitivity to measurement noise.
PHI_IFM_NOISE_STD = 0.0

# phi-IFM stagnation detection: if the post-ML TVD improves by less than
# PHI_IFM_STAGNATION_TOL (relative) over PHI_IFM_STAGNATION_WINDOW cycles,
# switch from fast to precise fringe fitting (paper §7.5.7).
PHI_IFM_STAGNATION_WINDOW = 2
PHI_IFM_STAGNATION_TOL = 0.05   # 5 % relative improvement threshold

OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _v_ifm_seeded_model(
    truth: DigitalTwin, seed: int, *, dtype: torch.dtype = torch.float64,
) -> DigitalTwin:
    """Fresh DigitalTwin with V-IFM-style seeds.

    V-IFM provides noisy estimates of the C_2 diagonal and the c_0 vector.
    Both are perturbed off the ground-truth values here (the magnitudes are
    chosen to be paper-plausible). Other parameters start at the framework
    defaults: C_2 off-diagonal = 0, R = 0.5, T_out ~= 1.

    ``dtype`` controls the precision of the returned model (and therefore
    of all gradients, Adam state, and forward intermediates). Must match
    the dtype of ``truth.c_0``, since the seed is built by adding noise to
    the truth tensor.
    """
    g = torch.Generator().manual_seed(seed)
    c_0_seed = truth.c_0.detach().to(dtype) + (
        torch.randn(truth.n_PS, generator=g, dtype=dtype)
        * C_0_SEED_NOISE
    )
    c2_diag_seed = truth.C_2.detach().diag().to(dtype) + (
        torch.randn(truth.n_PS, generator=g, dtype=dtype)
        * C_2_DIAG_SEED_NOISE
    )
    return DigitalTwin(
        truth.mesh, c_0_seed, c2_diag_seed, dtype=dtype,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iterative (ML + phi-IFM) loop until TVD < threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--m", type=int, default=10,
                   help="chip size; even, >= 2")
    p.add_argument("--scheme", choices=list(MESH_SCHEMES), default="bell",
                   help="mesh topology. 'bell' (default) is the compact "
                        "scheme: a PS on both arms of every MZI, no "
                        "external PS, plus independent PSs on the idle "
                        "boundary waveguides, giving n_PS = m**2. "
                        "'clements' is the older scheme (n_PS = m*(m-1)), "
                        "kept for comparison.")
    p.add_argument("--threshold", type=float, default=1e-3,
                   help="test-TVD convergence threshold; loop exits when "
                        "post-phi-IFM TVD falls below this")
    p.add_argument("--max-cycles", type=int, default=8,
                   help="hard cap on outer-loop iterations")
    p.add_argument("--epochs", type=int, default=200,
                   help="epochs per ML stage (paper uses 500)")
    p.add_argument("--n-samples", type=int, default=None,
                   help="size of the (x, port, p) dataset. If omitted, "
                        "auto-derived from --param-data-ratio (see "
                        "src/synthetic.py::n_samples_from_ratio) so the "
                        "dataset scales with chip size.")
    p.add_argument("--param-data-ratio", type=float, default=1.03,
                   help="training points per trainable parameter. The "
                        "total dataset is sized so the 80/20 train/test "
                        "split yields round(ratio * n_params) training "
                        "examples; see src/synthetic.py::n_samples_from_ratio. "
                        "Ignored if --n-samples is given explicitly.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for ground truth, dataset, and V-IFM seeds")
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="none",
                   help="within-stage LR scheduler. 'none' (default) keeps "
                        "the historical constant-LR behaviour; 'cosine' "
                        "anneals each parameter group's LR over the stage. "
                        "Recommended for m>=8 to remove the late-stage Adam "
                        "oscillation that floors the post-ML TVD around 2%%.")
    p.add_argument("--lr-eta-min-frac", type=float, default=0.0,
                   help="cosine schedule's floor as a fraction of each "
                        "group's initial LR. 0.0 = anneal to zero.")
    p.add_argument("--data-noise-std", type=float, default=DATA_NOISE_STD,
                   help="standard deviation of Gaussian noise added to the "
                        "synthetic output distributions p before the "
                        "non-negative clamp + renormalisation. 0.0 (current "
                        "default) = clean synthetic; set to e.g. 1e-3 to "
                        "emulate paper-like measurement noise.")
    p.add_argument("--phi-ifm-noise-std", type=float, default=PHI_IFM_NOISE_STD,
                   help="standard deviation of Gaussian noise added to each "
                        "phi-IFM fringe intensity sample. 0.0 (current "
                        "default) keeps the phi-IFM convergence study clean; "
                        "set to e.g. 1e-3 to emulate measurement shot noise "
                        "and study the algorithm's sensitivity.")
    p.add_argument("--improvement-tolerance", type=float, default=0.0,
                   help="for the no-improvement exit criterion: require the "
                        "next cycle's best test MSE to be < prev_best * "
                        "(1 - tolerance). 0.0 (default) reproduces the "
                        "historical strict 'any improvement' behaviour. "
                        "Set to e.g. 0.05 for noisy data so single-cycle "
                        "stochastic dips at the noise floor don't trigger "
                        "premature exit.")
    p.add_argument("--compile", dest="compile_model", action="store_true",
                   help="wrap the digital twin in torch.compile after "
                        "build. First batch incurs ~5-30 s of compilation "
                        "overhead; subsequent batches run from a fused "
                        "graph. Expected 1.3-1.8x CPU speedup once the "
                        "compile cost is amortised. Off by default.")
    p.add_argument("--num-threads", type=int, default=0,
                   help="if > 0, call torch.set_num_threads(N) early in "
                        "main(). PyTorch's default uses all logical cores; "
                        "passing the physical-core count (e.g. 4 on i5-8365U) "
                        "sometimes wins because hyperthreads compete for "
                        "L1/L2. 0 (default) leaves the PyTorch default "
                        "unchanged.")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64",
                   help="floating-point precision for model parameters, "
                        "gradients, and synthetic data. float64 is the "
                        "default because PyTorch's CPU complex64 (i.e. "
                        "float32-backed complex) dispatch is currently "
                        "slower than complex128 for our small-matrix, "
                        "many-small-ops forward (benchmarked on i5-8365U: "
                        "float32 backward ~40%% slower than float64). On "
                        "GPU or with a more vectorised _build_U, float32 "
                        "would win -- keep this flag for portability.")
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON path. Default: "
                        "outputs/iterative_m<m>.json. A sidecar .log "
                        "file is written alongside.")
    args = p.parse_args()
    if args.m < 2 or args.m % 2 != 0:
        p.error(f"--m must be even and >= 2 (got {args.m})")
    if not 0 < args.threshold < 1:
        p.error(f"--threshold must lie in (0, 1) (got {args.threshold})")
    if args.max_cycles < 1:
        p.error(f"--max-cycles must be >= 1 (got {args.max_cycles})")
    if args.param_data_ratio <= 0:
        p.error(
            f"--param-data-ratio must be > 0 (got {args.param_data_ratio})"
        )
    if args.n_samples is not None and args.n_samples < 1:
        p.error(f"--n-samples must be >= 1 (got {args.n_samples})")
    if args.out is None:
        # Scheme is in the filename so Bell and Clements runs at the same
        # m don't overwrite each other's results.
        args.out = OUTPUT_DIR / f"iterative_{args.scheme}_m{args.m}.json"
    return args


# ---------------------------------------------------------------------------
# Simulation driver (importable -- the notebook calls this directly)
# ---------------------------------------------------------------------------


def run_simulation(
    *,
    m: int = 10,
    scheme: str = "bell",
    threshold: float = 1e-3,
    max_cycles: int = 8,
    epochs: int = 200,
    n_samples: int = 2048,
    seed: int = 0,
    stop_on_no_improvement: bool = True,
    lr_schedule: str | None = None,
    lr_eta_min_frac: float = 0.0,
    dtype: torch.dtype = torch.float64,
    data_noise_std: float = DATA_NOISE_STD,
    phi_ifm_noise_std: float = PHI_IFM_NOISE_STD,
    improvement_tolerance: float = 0.0,
    compile_model: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the full iterative (ML + phi-IFM) loop and return the results.

    No file I/O -- the caller decides whether to plot or persist. ``main()``
    is a thin CLI wrapper; the exploration notebook imports this directly.

    The TVD-threshold convergence check uses the **post-ML** TVD: that is
    the model's true quality at the end of a cycle. The post-phi-IFM TVD
    oscillates upward because phi-IFM rewrites c_0 without re-fitting C_2
    (see CLAUDE.md §6.5 and this session's notes); it is recorded for
    diagnostics but not used as the exit gate.

    Args:
        m:           chip size (even, >= 2).
        scheme:      mesh topology, "bell" (default) or "clements". Changes
                     n_PS (m**2 vs m*(m-1)) and therefore the size of C_2,
                     the ground truth, and the V-IFM seeds.
        threshold:   post-ML TVD convergence target, in (0, 1).
        max_cycles:  hard cap on outer-loop iterations.
        epochs:      epochs per ML stage (held fixed across the sweep).
        n_samples:   size of the synthetic (V, port, p) dataset.
        seed:        RNG seed for ground truth, dataset, and V-IFM seeds.
        stop_on_no_improvement: if True, also exit when the ML stage's best
                     test MSE fails to improve over the previous cycle
                     (the paper's own criterion, CLAUDE.md §6.5). Set False
                     to let the loop run to max_cycles regardless.
        verbose:     print per-cycle progress.

    Returns:
        Dict with keys:
            "history"             : list[dict], per-cycle stats
            "initial_tvd"         : float, V-IFM-seed test TVD
            "converged"           : bool, post-ML TVD reached threshold
            "cycles_to_threshold" : int | None, first cycle with post-ML
                                    TVD < threshold (None if never)
            "exit_reason"         : str
            "elapsed"             : float, wall-clock seconds
            "truth"               : DigitalTwin, ground truth
            "model"               : DigitalTwin, final recovered model
            "m", "threshold", "epochs", "n_samples", "seed": echoed inputs
    """
    if m < 2 or m % 2 != 0:
        raise ValueError(f"m must be even and >= 2 (got {m})")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must lie in (0, 1) (got {threshold})")
    if max_cycles < 1:
        raise ValueError(f"max_cycles must be >= 1 (got {max_cycles})")

    torch.manual_seed(seed)
    truth = ground_truth_model(m, seed=seed, dtype=dtype, scheme=scheme)
    dataset = make_synthetic_dataset(
        truth, n_samples=n_samples,
        x_max=X_MAX, noise_std=data_noise_std,
        seed=seed + 1,
    )
    train_ds, test_ds = train_test_split(
        dataset, test_frac=TEST_FRAC, seed=seed + 2,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = _v_ifm_seeded_model(truth, seed=seed + 100, dtype=dtype)
    if compile_model:
        # torch.compile returns an OptimizedModule that wraps `model` and
        # delegates attribute access (so model.C_2_raw etc. still resolve
        # to the underlying nn.Parameters, and Adam can build its groups
        # from them as usual). First forward incurs the compile cost.
        if verbose:
            print(
                "  torch.compile: wrapping model "
                "(first batch will be slow due to graph compilation).",
                flush=True,
            )
        model = torch.compile(model)
    U_0_truth, U_pi_truth = build_chip_response_cache(truth)

    initial_stats = evaluate(model, test_loader)
    initial_tvd = initial_stats["tvd_mean"]
    if verbose:
        print(
            f"  n_PS = {truth.n_PS}    n_BS = {truth.n_BS}    "
            f"train = {len(train_ds)}    test = {len(test_ds)}",
            flush=True,
        )
        print(
            f"  Initial state (V-IFM seeds only): "
            f"test TVD = {initial_tvd * 100:.4f}%, "
            f"MSE = {initial_stats['mse_mean']:.3e}",
            flush=True,
        )
        print(flush=True)

    lr_C2 = LR_C2_INIT
    lr_R = LR_R_INIT
    lr_Tout = LR_TOUT_INIT
    history: list[dict] = []
    prev_best_mse = float("inf")
    converged = False
    cycles_to_threshold: int | None = None
    exit_reason = "max_cycles reached"
    phi_ifm_method = "fast"   # may switch to "precise" on stagnation

    t0 = time.perf_counter()
    for cycle in range(1, max_cycles + 1):
        if verbose:
            print(
                f"--- Cycle {cycle}  "
                f"(lr_C2={lr_C2:.2e}, lr_R={lr_R:.2e}, "
                f"lr_T={lr_Tout:.2e}) ---",
                flush=True,
            )

        # ML stage. Restore best-MSE state at the end so phi-IFM sees the
        # most accurate C_2, R, T_out (CLAUDE.md §6.4 -- monitor TVD per
        # epoch, keep the best).
        ml_history = train(
            model, train_loader, test_loader,
            epochs=epochs,
            lr_C2=lr_C2, lr_R=lr_R, lr_Tout=lr_Tout,
            lr_schedule=lr_schedule,
            lr_eta_min_frac=lr_eta_min_frac,
            verbose=verbose,
        )
        if ml_history["best_state_dict"] is not None:
            model.load_state_dict(ml_history["best_state_dict"])
        post_ml_stats = evaluate(model, test_loader)
        ml_best_mse = ml_history["best_test_mse"]

        # phi-IFM data with the current c_0_model. Refresh each cycle so
        # the shifts shrink as c_0 converges.
        data, _offsets = make_phi_ifm_data(
            truth, U_0_truth, U_pi_truth,
            c_0_model=model.c_0.detach(),
            n_points=PHI_IFM_N_POINTS,
            noise_std=phi_ifm_noise_std,
            seed=seed + 200 + cycle,
        )

        # phi-IFM stage (writes back into model.c_0).
        c_0_new = run_phi_ifm_stage(model, data, method=phi_ifm_method)
        with torch.no_grad():
            model.c_0.copy_(c_0_new)
        post_phi_ifm_stats = evaluate(model, test_loader)

        history.append({
            "cycle": cycle,
            "lr_C2": lr_C2, "lr_R": lr_R, "lr_Tout": lr_Tout,
            "ml_best_mse": ml_best_mse,
            "ml_best_tvd": float(min(ml_history["test_tvd"])),
            "post_ml_mse": post_ml_stats["mse_mean"],
            "post_ml_tvd": post_ml_stats["tvd_mean"],
            "post_phi_ifm_mse": post_phi_ifm_stats["mse_mean"],
            "post_phi_ifm_tvd": post_phi_ifm_stats["tvd_mean"],
            "phi_ifm_method": phi_ifm_method,
        })

        # Stagnation check: if the fast method's TVD has barely moved over the
        # last WINDOW cycles, switch to precise (paper §7.5.7).
        if phi_ifm_method == "fast" and len(history) > PHI_IFM_STAGNATION_WINDOW:
            tvd_now = history[-1]["post_ml_tvd"]
            tvd_before = history[-1 - PHI_IFM_STAGNATION_WINDOW]["post_ml_tvd"]
            rel_improvement = (tvd_before - tvd_now) / max(tvd_before, 1e-12)
            if rel_improvement < PHI_IFM_STAGNATION_TOL:
                phi_ifm_method = "precise"
                if verbose:
                    print(
                        f"  -> phi-IFM: fast -> precise  "
                        f"(TVD improved only {rel_improvement * 100:.1f}% "
                        f"over {PHI_IFM_STAGNATION_WINDOW} cycles)",
                        flush=True,
                    )

        if verbose:
            print(
                f"  post-ML       : "
                f"TVD = {post_ml_stats['tvd_mean'] * 100:.4f}%   "
                f"MSE = {post_ml_stats['mse_mean']:.3e}",
                flush=True,
            )
            print(
                f"  post-phi-IFM  : "
                f"TVD = {post_phi_ifm_stats['tvd_mean'] * 100:.4f}%   "
                f"MSE = {post_phi_ifm_stats['mse_mean']:.3e}",
                flush=True,
            )

        # Convergence checks. Threshold is on the post-ML TVD.
        if post_ml_stats["tvd_mean"] < threshold:
            converged = True
            cycles_to_threshold = cycle
            exit_reason = (
                f"post-ML TVD < threshold "
                f"({post_ml_stats['tvd_mean'] * 100:.4f}% "
                f"< {threshold * 100:.4f}%)"
            )
            if verbose:
                print(f"  -> {exit_reason}.", flush=True)
            break
        # No-improvement exit. Default tolerance=0 reproduces the historical
        # behaviour (exit on any non-strict regression). With tolerance>0 the
        # next cycle must improve the best test MSE by at least that fraction
        # of the previous best, otherwise we declare convergence. This
        # buffers against single-cycle stochastic dips at a noise floor
        # (see CLAUDE.md §6.4 -- 5% is a sensible default for noisy data).
        no_improvement_threshold = prev_best_mse * (1.0 - improvement_tolerance)
        if (
            stop_on_no_improvement
            and cycle > 1
            and ml_best_mse >= no_improvement_threshold
        ):
            tol_msg = (
                f" (< {improvement_tolerance * 100:.1f}% relative improvement)"
                if improvement_tolerance > 0 else ""
            )
            exit_reason = (
                f"no improvement in best test MSE "
                f"(prev {prev_best_mse:.3e} -> now {ml_best_mse:.3e}"
                f"{tol_msg})"
            )
            if verbose:
                print(f"  -> stopping: {exit_reason}.", flush=True)
            break
        prev_best_mse = ml_best_mse

        # LR decay (CLAUDE.md §6.5).
        lr_C2 *= LR_DECAY
        lr_R *= LR_DECAY
        lr_Tout *= LR_DECAY

    elapsed = time.perf_counter() - t0

    if verbose:
        print(flush=True)
        print("Summary", flush=True)
        print("-" * 70, flush=True)
        print(
            f"  cycles run         : {len(history)} "
            f"(max requested = {max_cycles})",
            flush=True,
        )
        print(f"  converged          : {converged}", flush=True)
        print(f"  cycles to threshold: {cycles_to_threshold}", flush=True)
        print(f"  exit reason        : {exit_reason}", flush=True)
        print(f"  initial TVD        : {initial_tvd * 100:.4f}%", flush=True)
        if history:
            print(
                f"  final post-ML TVD  : "
                f"{history[-1]['post_ml_tvd'] * 100:.4f}%",
                flush=True,
            )
        print(f"  wall-clock         : {elapsed:.1f} s", flush=True)

    return {
        "history": history,
        "initial_tvd": initial_tvd,
        "converged": converged,
        "cycles_to_threshold": cycles_to_threshold,
        "exit_reason": exit_reason,
        "elapsed": elapsed,
        "truth": truth,
        "model": model,
        "m": m,
        "scheme": scheme,
        "threshold": threshold,
        "epochs": epochs,
        "n_samples": n_samples,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Main (CLI wrapper)
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    m = args.m
    log_path = args.out.with_suffix(".log")
    setup_logging(log_path)

    # Auto-derive n_samples from --param-data-ratio unless the caller
    # passed --n-samples explicitly.
    if args.n_samples is None:
        n_samples = n_samples_from_ratio(
            m, args.param_data_ratio, scheme=args.scheme,
        )
        sample_source = f"auto: {args.param_data_ratio} x n_params"
    else:
        n_samples = args.n_samples
        sample_source = "explicit --n-samples"

    print(f"Iterative (ML + phi-IFM) loop  --  scheme={args.scheme}, m={m}, "
          f"threshold={args.threshold * 100:.4f}%, "
          f"max_cycles={args.max_cycles}")
    print(f"  n_samples = {n_samples}  ({sample_source})")
    print(f"  output:    {args.out}")
    print(f"  log:       {log_path}")
    print("=" * 70)

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        print(f"  torch.set_num_threads({args.num_threads}) "
              f"(was using default {torch.get_num_threads()} before this call)")
    lr_schedule = None if args.lr_schedule == "none" else args.lr_schedule
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    result = run_simulation(
        m=m,
        scheme=args.scheme,
        threshold=args.threshold,
        max_cycles=args.max_cycles,
        epochs=args.epochs,
        n_samples=n_samples,
        seed=args.seed,
        lr_schedule=lr_schedule,
        lr_eta_min_frac=args.lr_eta_min_frac,
        dtype=dtype,
        data_noise_std=args.data_noise_std,
        phi_ifm_noise_std=args.phi_ifm_noise_std,
        improvement_tolerance=args.improvement_tolerance,
        compile_model=args.compile_model,
        verbose=True,
    )

    truth = result["truth"]
    model = result["model"]

    bundle = {
        "config": {
            "m": m,
            "scheme": args.scheme,
            "threshold": args.threshold,
            "max_cycles": args.max_cycles,
            "epochs": args.epochs,
            "n_samples": n_samples,
            "n_samples_source": sample_source,
            "param_data_ratio": args.param_data_ratio,
            "seed": args.seed,
            "lr_C2_init": LR_C2_INIT,
            "lr_R_init": LR_R_INIT,
            "lr_Tout_init": LR_TOUT_INIT,
            "lr_decay": LR_DECAY,
            "lr_schedule": args.lr_schedule,
            "lr_eta_min_frac": args.lr_eta_min_frac,
            "dtype": args.dtype,
            "compile_model": args.compile_model,
            "num_threads": args.num_threads,
            "batch_size": BATCH_SIZE,
            "test_frac": TEST_FRAC,
            "x_max": X_MAX,
            "data_noise_std": args.data_noise_std,
            "phi_ifm_n_points": PHI_IFM_N_POINTS,
            "phi_ifm_noise_std": args.phi_ifm_noise_std,
            "improvement_tolerance": args.improvement_tolerance,
            "phi_ifm_stagnation_window": PHI_IFM_STAGNATION_WINDOW,
            "phi_ifm_stagnation_tol": PHI_IFM_STAGNATION_TOL,
        },
        "summary": {
            "initial_tvd": result["initial_tvd"],
            "converged": result["converged"],
            "cycles_to_threshold": result["cycles_to_threshold"],
            "exit_reason": result["exit_reason"],
            "elapsed": result["elapsed"],
        },
        "history": result["history"],
        "parameters": {
            "truth": {
                "C_2": truth.C_2.detach().cpu().tolist(),
                "R": truth.R.detach().cpu().tolist(),
                "T_out": truth.T_out.detach().cpu().tolist(),
                "c_0": truth.c_0.detach().cpu().tolist(),
            },
            "recovered": {
                "C_2": model.C_2.detach().cpu().tolist(),
                "R": model.R.detach().cpu().tolist(),
                "T_out": model.T_out.detach().cpu().tolist(),
                "c_0": model.c_0.detach().cpu().tolist(),
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
