"""Fig 3.B sweep: cycles-to-threshold vs epochs, one curve per chip size m.

Runs the iterative (ML + phi-IFM) protocol over a grid of
(m, epochs-per-ML-stage) pairs and writes a JSON bundle for plotting.

Safeguards
----------
* **Incremental saving** — the output JSON is written after every
  (m, epochs) pair. If the run is interrupted, restart with the same
  ``--out`` path and completed pairs are automatically skipped.
* **Per-pair exception handling** — a crash inside one run_simulation
  call records an "error" entry and continues with the remaining grid.
* **Stdout flushing** — progress is printed and flushed immediately so
  you can tail the log: ``uv run python scripts/sweep_m_epochs.py | tee run.log``

Usage::

    uv run python scripts/sweep_m_epochs.py \\
        [--m-values 6 8 10 12] \\
        [--epochs-values 200 300 400 500] \\
        [--threshold 1e-3] \\
        [--max-cycles 20] \\
        [--param-data-ratio 1.03] \\
        [--seed 0] \\
        [--out outputs/sweep_m_epochs.json]

Resume an interrupted run by passing the same --out path; completed
(m, epochs) pairs are detected and skipped automatically.
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

from run_iterative import run_simulation

from src.cli import setup_logging
from src.synthetic import n_samples_from_ratio


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fig 3.B sweep: cycles-to-threshold vs epochs per ML stage, "
            "one curve per chip size m."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--m-values", type=int, nargs="+", default=[6, 8, 10, 12],
        help="chip sizes to sweep (each even, >= 2)",
    )
    p.add_argument(
        "--epochs-values", type=int, nargs="+", default=[200, 300, 400, 500],
        help="epochs-per-ML-stage values to sweep",
    )
    p.add_argument(
        "--threshold", type=float, default=1e-3,
        help="post-ML TVD convergence target (paper Fig 3.B uses 1e-3)",
    )
    p.add_argument(
        "--max-cycles", type=int, default=20,
        help="hard cap on outer-loop iterations per (m, epochs) pair; "
             "pairs that do not converge are recorded as DNF",
    )
    p.add_argument(
        "--param-data-ratio", type=float, default=1.03,
        help="training points per trainable parameter; holds the "
             "parameter:data ratio constant across m. See "
             "src/synthetic.py::n_samples_from_ratio.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed (shared across all pairs for fair comparison)",
    )
    p.add_argument(
        "--lr-schedule", choices=["none", "cosine"], default="none",
        help="within-stage LR scheduler passed to run_simulation. 'none' "
             "keeps the historical constant-LR behaviour; 'cosine' anneals "
             "each parameter group's LR over the stage. Recommended for "
             "m>=8 to escape the 2%% post-ML TVD plateau.",
    )
    p.add_argument(
        "--lr-eta-min-frac", type=float, default=0.0,
        help="cosine schedule's floor as a fraction of each group's "
             "initial LR. 0.0 = anneal to zero.",
    )
    p.add_argument(
        "--out", type=Path, default=Path("outputs/sweep_m_epochs.json"),
        help="output JSON path; existing results are loaded and resumed",
    )
    args = p.parse_args()
    for m in args.m_values:
        if m < 2 or m % 2 != 0:
            p.error(f"--m-values entries must be even and >= 2 (got {m})")
    for e in args.epochs_values:
        if e < 1:
            p.error(f"--epochs-values entries must be >= 1 (got {e})")
    if not 0 < args.threshold < 1:
        p.error(f"--threshold must lie in (0, 1) (got {args.threshold})")
    if args.max_cycles < 1:
        p.error(f"--max-cycles must be >= 1 (got {args.max_cycles})")
    if args.param_data_ratio <= 0:
        p.error(f"--param-data-ratio must be > 0 (got {args.param_data_ratio})")
    return args


def _n_params_samples(m: int, ratio: float) -> tuple[int, int, int, int]:
    """Return (n_PS, n_BS, n_params, n_samples) for a given m and ratio."""
    n_PS = m * (m - 1)
    n_BS = m * (m - 1)
    n_params = n_PS * n_PS + n_BS + m
    n_samples = n_samples_from_ratio(m, ratio)
    return n_PS, n_BS, n_params, n_samples


def _save(bundle: dict, path: Path) -> None:
    """Atomically write the bundle JSON (write temp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    tmp.replace(path)


def _log(msg: str) -> None:
    """Print and flush immediately so tee / tail see it at once."""
    print(msg, flush=True)


def main() -> None:
    args = _parse_args()
    log_path = args.out.with_suffix(".log")
    setup_logging(log_path)

    # ---- Print grid summary ------------------------------------------------
    _log("=" * 72)
    _log("Fig 3.B sweep  —  (m, epochs) grid")
    _log(f"  m_values     : {args.m_values}")
    _log(f"  epochs_values: {args.epochs_values}")
    _log(f"  threshold    : {args.threshold * 100:g}%")
    _log(f"  max_cycles   : {args.max_cycles}")
    _log(f"  ratio        : {args.param_data_ratio}")
    _log(f"  seed         : {args.seed}")
    _log(f"  lr_schedule  : {args.lr_schedule}"
         + (f" (eta_min_frac={args.lr_eta_min_frac})"
            if args.lr_schedule != "none" else ""))
    _log(f"  output       : {args.out}")
    _log(f"  log          : {log_path}")
    _log("")
    for m in args.m_values:
        n_PS, n_BS, n_params, n_samples = _n_params_samples(
            m, args.param_data_ratio
        )
        _log(
            f"  m={m:>2}  n_PS={n_PS:>4}  n_params={n_params:>7}  "
            f"n_samples={n_samples:>7}"
        )
    _log("=" * 72)

    # ---- Load existing results (resume support) ----------------------------
    config = {
        "m_values": list(args.m_values),
        "epochs_values": list(args.epochs_values),
        "threshold": args.threshold,
        "max_cycles": args.max_cycles,
        "param_data_ratio": args.param_data_ratio,
        "seed": args.seed,
        "lr_schedule": args.lr_schedule,
        "lr_eta_min_frac": args.lr_eta_min_frac,
    }

    lr_schedule = None if args.lr_schedule == "none" else args.lr_schedule

    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        results: dict[str, dict] = existing.get("results", {})
        prior_elapsed: float = existing.get("total_elapsed", 0.0)
        _log(f"\nResuming from {args.out}  (prior elapsed: {prior_elapsed:.1f} s)")
    else:
        results = {}
        prior_elapsed = 0.0

    # ---- Main sweep loop ---------------------------------------------------
    wall_t0 = time.perf_counter()
    total_pairs = len(args.m_values) * len(args.epochs_values)
    done_pairs = 0

    def _checkpoint(reason: str = "complete") -> None:
        elapsed = prior_elapsed + (time.perf_counter() - wall_t0)
        _save(
            {"config": config, "total_elapsed": elapsed, "results": results},
            args.out,
        )
        _log(f"  -- {reason}: saved -> {args.out}")

    try:
        for m in args.m_values:
            n_PS, n_BS, n_params, n_samples = _n_params_samples(
                m, args.param_data_ratio
            )
            m_key = str(m)
            if m_key not in results:
                results[m_key] = {}

            for epochs in args.epochs_values:
                e_key = str(epochs)

                # Skip if already completed (not an error record).
                existing_entry = results[m_key].get(e_key, {})
                if existing_entry and existing_entry.get("status") != "error":
                    done_pairs += 1
                    _log(
                        f"\n[{done_pairs}/{total_pairs}]  m={m}, epochs={epochs}  "
                        f"-- SKIPPED (already done: "
                        f"cycles={existing_entry.get('cycles_to_threshold', 'DNF')}, "
                        f"{existing_entry.get('elapsed', 0):.1f} s)"
                    )
                    continue

                done_pairs += 1
                _log(
                    f"\n{'='*72}\n"
                    f"[{done_pairs}/{total_pairs}]  m={m}  epochs={epochs}  "
                    f"(n_PS={n_PS}, n_params={n_params}, n_samples={n_samples})"
                )

                try:
                    r = run_simulation(
                        m=m,
                        threshold=args.threshold,
                        max_cycles=args.max_cycles,
                        epochs=epochs,
                        n_samples=n_samples,
                        seed=args.seed,
                        stop_on_no_improvement=False,
                        lr_schedule=lr_schedule,
                        lr_eta_min_frac=args.lr_eta_min_frac,
                        verbose=True,
                    )
                    results[m_key][e_key] = {
                        "status": "ok",
                        "m": m,
                        "epochs": epochs,
                        "n_PS": n_PS,
                        "n_params": n_params,
                        "n_samples": n_samples,
                        "lr_schedule": args.lr_schedule,
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
                            }
                            for h in r["history"]
                        ],
                    }
                    _log(
                        f"  >> runtime: {r['elapsed']:.1f} s  |  "
                        f"cycles to threshold: "
                        f"{r['cycles_to_threshold'] if r['converged'] else 'DNF'}"
                    )

                except KeyboardInterrupt:
                    raise  # skip the error-record and let the outer handler save
                except Exception:
                    tb = traceback.format_exc()
                    _log(f"  ERROR for m={m}, epochs={epochs}:\n{tb}")
                    results[m_key][e_key] = {
                        "status": "error",
                        "m": m,
                        "epochs": epochs,
                        "error": tb,
                    }

                _checkpoint("checkpoint")

    except KeyboardInterrupt:
        _log("\n\nCtrl+C received — saving completed results and exiting.")
        _checkpoint("interrupted")
        _log("Re-run the same command to resume from this point.")
        sys.exit(0)

    # ---- Final save with total elapsed ------------------------------------
    _checkpoint("sweep complete")

    total_elapsed = prior_elapsed + (time.perf_counter() - wall_t0)
    _log("\n" + "=" * 72)
    _log(f"sweep complete in {total_elapsed:.1f} s  ({total_elapsed/3600:.2f} h)")

    # ---- Summary table -----------------------------------------------------
    _log(
        f"\n{'m':>4}  {'epochs':>8}  {'cycles':>8}  "
        f"{'final TVD':>12}  {'time (s)':>10}"
    )
    _log("-" * 52)
    for m in args.m_values:
        for epochs in args.epochs_values:
            entry = results.get(str(m), {}).get(str(epochs), {})
            if not entry:
                _log(f"{m:>4}  {epochs:>8}  {'MISSING':>8}")
                continue
            if entry.get("status") == "error":
                _log(f"{m:>4}  {epochs:>8}  {'ERROR':>8}")
                continue
            cyc = entry["cycles_to_threshold"]
            cyc_str = str(cyc) if cyc is not None else "DNF"
            hist = entry.get("history", [])
            final_tvd = hist[-1]["post_ml_tvd"] * 100 if hist else float("nan")
            _log(
                f"{m:>4}  {epochs:>8}  {cyc_str:>8}  "
                f"{final_tvd:>11.4f}%  {entry['elapsed']:>10.1f}"
            )


if __name__ == "__main__":
    main()
