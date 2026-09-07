"""TVD-floor vs measurement-noise level for the iterative protocol.

Maps to paper Fig 3(d) (§3.F.c) but uses Gaussian additive noise as a
stand-in for true Poisson photon-counting. For a faithful reproduction
of Fig 3(d) we'd swap the noise model in src/data.py and src/synthetic.py
to draw from Multinomial(rate * 1s, p_true); that's a follow-up. This
script answers the qualitative shape (TVD floor vs sigma).

Design
------
Orchestrator over ``scripts/run_iterative.run_simulation`` -- no
re-implementation of the protocol. For each noise sigma in
``--noise-values``:

  * Run one iterative loop with data_noise_std == phi_ifm_noise_std == sigma
  * Capture per-cycle post-ML / post-phi-IFM TVDs and the per-ML-stage
    wall-clock time
  * The per-(epochs/5) test_tvd sampling within each ML stage is logged
    *for free* by ``run_simulation(..., verbose=True)`` -- those lines
    land in the same ``.log`` sidecar that ``notebooks/exploration.ipynb``
    Plot 4 already parses for cumulative-epoch trajectories.

The output JSON has the same shape as ``sweep_m_epochs.json`` (config +
total_elapsed + results), so the notebook's analysis cells can be pointed
at it with no code change.

Resume support: re-run with the same ``--out`` and completed noise points
are skipped.

Usage::

    uv run python scripts/sweep_noise.py [--m 8] [--epochs 200]
        [--max-cycles 6] [--lr-schedule cosine] [--threshold 1e-3]
        [--noise-values 0.0 1e-4 1e-3 1e-2] [--seed 0]
        [--out outputs/sweep_noise.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS.parent
for _p in (_PROJECT_ROOT, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch

from run_iterative import run_simulation

from src.cli import setup_logging
from src.synthetic import n_samples_from_ratio


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep TVD floor vs measurement-noise level on the iterative "
            "(ML + phi-IFM) protocol (Gaussian-noise stand-in for paper "
            "Fig 3(d) Poisson photon-counting)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--m", type=int, default=8,
                   help="chip size (even, >= 2)")
    p.add_argument("--epochs", type=int, default=200,
                   help="epochs per ML stage")
    p.add_argument("--max-cycles", type=int, default=10,
                   help="hard cap on outer-loop iterations per noise level. "
                        "Set high; the actual exit is the paper's own "
                        "criterion (best test MSE stops improving), which "
                        "fires naturally at the noise floor.")
    p.add_argument("--no-early-stop", action="store_true",
                   help="disable the paper's 'stop when best test MSE stops "
                        "improving' criterion; run all max_cycles regardless. "
                        "Off by default; only useful for debugging the "
                        "trajectory beyond the natural plateau.")
    p.add_argument("--improvement-tolerance", type=float, default=0.05,
                   help="for the no-improvement exit: require >= tolerance*100%% "
                        "relative improvement in best test MSE cycle-over-cycle. "
                        "Default 0.05 (5%%) is appropriate for noisy data so "
                        "single-cycle stochastic dips at the noise floor don't "
                        "trigger premature exit. Set to 0.0 to reproduce the "
                        "strict 'any improvement' behaviour of run_iterative.py.")
    p.add_argument("--noise-values", type=float, nargs="+",
                   default=[0.0, 1e-4, 1e-3, 1e-2],
                   help="noise sigmas to sweep. Same sigma is applied to "
                        "BOTH the synthetic dataset (data_noise_std) and "
                        "every phi-IFM fringe (phi_ifm_noise_std). 0.0 "
                        "reproduces the clean case (matches the existing "
                        "sweep_m_epochs results).")
    p.add_argument("--threshold", type=float, default=1e-3,
                   help="post-ML TVD convergence target")
    p.add_argument("--lr-schedule", choices=["none", "cosine"],
                   default="cosine",
                   help="within-stage LR scheduler passed to run_simulation")
    p.add_argument("--lr-eta-min-frac", type=float, default=0.0)
    p.add_argument("--param-data-ratio", type=float, default=1.03)
    p.add_argument("--dtype", choices=["float32", "float64"],
                   default="float64")
    p.add_argument("--compile", dest="compile_model", action="store_true",
                   help="wrap the digital twin in torch.compile (opt-in, "
                        "off by default). Compile cost amortised across "
                        "all noise sweep points.")
    p.add_argument("--num-threads", type=int, default=0,
                   help="if > 0, torch.set_num_threads(N) before any run. "
                        "0 = leave PyTorch default.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", type=Path, default=Path("outputs/sweep_noise.json"),
        help="output JSON path; existing results are loaded and resumed",
    )
    args = p.parse_args()

    if args.m < 2 or args.m % 2 != 0:
        p.error(f"--m must be even and >= 2 (got {args.m})")
    if args.epochs < 1:
        p.error(f"--epochs must be >= 1 (got {args.epochs})")
    if args.max_cycles < 1:
        p.error(f"--max-cycles must be >= 1 (got {args.max_cycles})")
    if any(s < 0 for s in args.noise_values):
        p.error(f"--noise-values must all be >= 0 (got {args.noise_values})")
    if not 0 < args.threshold < 1:
        p.error(f"--threshold must lie in (0, 1) (got {args.threshold})")
    if args.param_data_ratio <= 0:
        p.error(
            f"--param-data-ratio must be > 0 (got {args.param_data_ratio})"
        )
    return args


def _save(bundle: dict, path: Path) -> None:
    """Atomically write the bundle JSON (write temp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    tmp.replace(path)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _noise_key(sigma: float) -> str:
    """JSON-key for a noise sigma. Uses scientific notation for tiny values."""
    if sigma == 0:
        return "0"
    return f"{sigma:.3e}"


def main() -> None:
    args = _parse_args()
    log_path = args.out.with_suffix(".log")
    setup_logging(log_path)

    # n_samples derived once from the (m, ratio) pair -- held constant
    # across noise levels so the only varying axis is sigma.
    n_samples = n_samples_from_ratio(args.m, args.param_data_ratio)
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    lr_schedule = None if args.lr_schedule == "none" else args.lr_schedule

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        _log(
            f"  torch.set_num_threads({args.num_threads}) "
            f"(default was {torch.get_num_threads()} threads)"
        )

    _log("=" * 72)
    _log("Noise-sweep  --  TVD floor vs Gaussian noise sigma")
    _log(f"  m              : {args.m}")
    _log(f"  epochs/stage   : {args.epochs}")
    _log(f"  max_cycles     : {args.max_cycles}")
    _log(f"  threshold      : {args.threshold * 100:g}%")
    _log(f"  lr_schedule    : {args.lr_schedule}")
    _log(f"  dtype          : {args.dtype}")
    _log(f"  ratio          : {args.param_data_ratio}")
    _log(f"  n_samples      : {n_samples}")
    _log(f"  noise_values   : {args.noise_values}")
    _log(f"  seed           : {args.seed}")
    _log(f"  output         : {args.out}")
    _log(f"  log            : {log_path}")
    _log("=" * 72)

    config = {
        "m": args.m,
        "epochs": args.epochs,
        "max_cycles": args.max_cycles,
        "threshold": args.threshold,
        "lr_schedule": args.lr_schedule,
        "lr_eta_min_frac": args.lr_eta_min_frac,
        "dtype": args.dtype,
        "compile_model": args.compile_model,
        "num_threads": args.num_threads,
        "improvement_tolerance": args.improvement_tolerance,
        "param_data_ratio": args.param_data_ratio,
        "n_samples": n_samples,
        "noise_values": list(args.noise_values),
        "seed": args.seed,
    }

    # ---- Resume support ---------------------------------------------------
    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        results: dict[str, dict] = existing.get("results", {})
        prior_elapsed: float = existing.get("total_elapsed", 0.0)
        _log(
            f"\nResuming from {args.out}  "
            f"(prior elapsed: {prior_elapsed:.1f} s)"
        )
    else:
        results = {}
        prior_elapsed = 0.0

    # ---- Sweep ------------------------------------------------------------
    wall_t0 = time.perf_counter()
    total_points = len(args.noise_values)
    done = 0

    def _checkpoint(reason: str = "checkpoint") -> None:
        elapsed = prior_elapsed + (time.perf_counter() - wall_t0)
        _save(
            {"config": config, "total_elapsed": elapsed, "results": results},
            args.out,
        )
        _log(f"  -- {reason}: saved -> {args.out}")

    try:
        for sigma in args.noise_values:
            key = _noise_key(sigma)
            existing_entry = results.get(key, {})
            if existing_entry and existing_entry.get("status") != "error":
                done += 1
                _log(
                    f"\n[{done}/{total_points}]  sigma={sigma:.3e}  "
                    f"-- SKIPPED (already done: cycles="
                    f"{existing_entry.get('cycles_to_threshold', 'DNF')}, "
                    f"final TVD="
                    f"{(existing_entry.get('history') or [{}])[-1].get('post_ml_tvd', float('nan'))*100:.4f}%)"
                )
                continue

            done += 1
            _log(
                f"\n{'=' * 72}\n"
                f"[{done}/{total_points}]  sigma={sigma:.3e}  "
                f"(data + phi-IFM both)"
            )

            try:
                r = run_simulation(
                    m=args.m,
                    threshold=args.threshold,
                    max_cycles=args.max_cycles,
                    epochs=args.epochs,
                    n_samples=n_samples,
                    seed=args.seed,
                    # Use the paper's own criterion (§3.D): exit when the
                    # next ML stage's best test MSE no longer improves over
                    # the previous cycle. For noisy data this fires
                    # naturally at the noise floor (~3-5 cycles); for clean
                    # data it fires after ~2 cycles. ``--no-early-stop``
                    # overrides this and runs the full ``max_cycles``.
                    stop_on_no_improvement=not args.no_early_stop,
                    lr_schedule=lr_schedule,
                    lr_eta_min_frac=args.lr_eta_min_frac,
                    dtype=dtype,
                    data_noise_std=sigma,
                    phi_ifm_noise_std=sigma,
                    improvement_tolerance=args.improvement_tolerance,
                    compile_model=args.compile_model,
                    verbose=True,
                )
                results[key] = {
                    "status": "ok",
                    "noise_std": sigma,
                    "converged": r["converged"],
                    "cycles_to_threshold": r["cycles_to_threshold"],
                    "exit_reason": r["exit_reason"],
                    "elapsed": r["elapsed"],
                    "initial_tvd": r["initial_tvd"],
                    "history": [
                        {
                            "cycle": h["cycle"],
                            "post_ml_tvd": h["post_ml_tvd"],
                            "post_phi_ifm_tvd": h["post_phi_ifm_tvd"],
                            "post_ml_mse": h["post_ml_mse"],
                            "post_phi_ifm_mse": h["post_phi_ifm_mse"],
                            "phi_ifm_method": h.get("phi_ifm_method", "fast"),
                            "lr_C2": h.get("lr_C2"),
                            "lr_R": h.get("lr_R"),
                            "lr_Tout": h.get("lr_Tout"),
                        }
                        for h in r["history"]
                    ],
                }
                final_tvd = r["history"][-1]["post_ml_tvd"] * 100 \
                    if r["history"] else float("nan")
                _log(
                    f"  >> runtime: {r['elapsed']:.1f} s  |  "
                    f"cycles: {len(r['history'])}  |  "
                    f"final post-ML TVD: {final_tvd:.4f}%"
                )

            except KeyboardInterrupt:
                raise
            except Exception:
                tb = traceback.format_exc()
                _log(f"  ERROR for sigma={sigma}:\n{tb}")
                results[key] = {
                    "status": "error",
                    "noise_std": sigma,
                    "error": tb,
                }

            _checkpoint("checkpoint")

    except KeyboardInterrupt:
        _log("\n\nCtrl+C received -- saving completed results and exiting.")
        _checkpoint("interrupted")
        _log("Re-run the same command to resume from this point.")
        sys.exit(0)

    _checkpoint("sweep complete")

    total_elapsed = prior_elapsed + (time.perf_counter() - wall_t0)
    _log("\n" + "=" * 72)
    _log(
        f"sweep complete in {total_elapsed:.1f} s  "
        f"({total_elapsed/3600:.2f} h)"
    )

    # Summary table
    _log(
        f"\n{'sigma':>10}  {'cycles':>8}  {'final TVD':>12}  "
        f"{'time (s)':>10}  status"
    )
    _log("-" * 56)
    for sigma in args.noise_values:
        key = _noise_key(sigma)
        entry = results.get(key, {})
        if not entry:
            _log(f"{sigma:>10.3e}  {'MISSING':>8}")
            continue
        if entry.get("status") == "error":
            _log(f"{sigma:>10.3e}  {'ERROR':>8}")
            continue
        cyc = entry["cycles_to_threshold"]
        cyc_str = str(cyc) if cyc is not None else "DNF"
        hist = entry.get("history", [])
        ftv = hist[-1]["post_ml_tvd"] * 100 if hist else float("nan")
        _log(
            f"{sigma:>10.3e}  {cyc_str:>8}  "
            f"{ftv:>11.4f}%  {entry['elapsed']:>10.1f}  ok"
        )


if __name__ == "__main__":
    main()
