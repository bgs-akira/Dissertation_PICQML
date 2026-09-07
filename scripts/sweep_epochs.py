"""Offline epoch-sweep: cycles-to-threshold vs epochs-per-ML-stage (Fig 3.B).

Runs the iterative (ML + phi-IFM) protocol for every ``epochs`` value in a
grid, at a fixed chip size ``m`` and fixed parameter:data ratio, and writes a
JSON bundle. Plotting lives in ``notebooks/exploration.ipynb``, which just
loads that file.

Replicates Figure 3.B of Fyrillas et al. (2024): the number of
(ML + phi-IFM) cycles needed to drive the post-ML test TVD below a threshold
decreases as the number of epochs per ML stage increases (up to a point), at
the cost of longer wall-clock time per cycle.

Fixed parameter:data ratio
--------------------------
Dataset size scales with the trainable parameter count via
``src/synthetic.py::n_samples_from_ratio``: the total dataset is sized so
the 80/20 train/test split yields ``round(param_data_ratio * n_params)``
training examples, where ``n_params = n_PS**2 + n_BS + m``. The default
ratio is 1.03 (same as sweep_m.py).

Usage::

    uv run python scripts/sweep_epochs.py \\
        [--epochs-values 50 100 200 500] \\
        [--m 6] \\
        [--threshold 1e-3] \\
        [--max-cycles 20] \\
        [--param-data-ratio 1.03] \\
        [--seed 0] \\
        [--out outputs/sweep_epochs.json]

See ``scripts/sweep_epochs.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
            "Offline epoch-sweep for Fig 3.B: cycles-to-threshold vs "
            "epochs-per-ML-stage."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--epochs-values", type=int, nargs="+",
        default=[50, 100, 200, 500],
        help="epochs-per-ML-stage values to sweep",
    )
    p.add_argument(
        "--m", type=int, default=6,
        help="chip size; fixed across the sweep (even, >= 2)",
    )
    p.add_argument(
        "--threshold", type=float, default=1e-3,
        help="post-ML TVD convergence target (paper Fig 3.B uses 1e-3)",
    )
    p.add_argument(
        "--max-cycles", type=int, default=20,
        help="hard cap on outer-loop iterations per epochs value; "
             "runs that do not reach threshold within this budget are "
             "recorded as DNF (did not finish)",
    )
    p.add_argument(
        "--param-data-ratio", type=float, default=1.03,
        help="training points per trainable parameter; see "
             "src/synthetic.py::n_samples_from_ratio",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed (shared across all epochs values for fair comparison)",
    )
    p.add_argument(
        "--out", type=Path, default=Path("outputs/sweep_epochs.json"),
        help="output JSON path",
    )
    args = p.parse_args()
    if args.m < 2 or args.m % 2 != 0:
        p.error(f"--m must be even and >= 2 (got {args.m})")
    if not 0 < args.threshold < 1:
        p.error(f"--threshold must lie in (0, 1) (got {args.threshold})")
    if args.max_cycles < 1:
        p.error(f"--max-cycles must be >= 1 (got {args.max_cycles})")
    if args.param_data_ratio <= 0:
        p.error(f"--param-data-ratio must be > 0 (got {args.param_data_ratio})")
    for e in args.epochs_values:
        if e < 1:
            p.error(f"--epochs-values entries must be >= 1 (got {e})")
    return args


def main() -> None:
    args = _parse_args()
    log_path = args.out.with_suffix(".log")
    setup_logging(log_path)
    m = args.m
    n_PS = m * (m - 1)
    n_params = n_PS * n_PS + n_PS + m
    n_samples = n_samples_from_ratio(m, args.param_data_ratio)

    print(
        f"epoch-sweep:  m={m},  n_PS={n_PS},  n_params={n_params},  "
        f"n_samples={n_samples}"
    )
    print(f"  output: {args.out}")
    print(f"  log:    {log_path}")
    print(
        f"              threshold={args.threshold * 100:g}%,  "
        f"max_cycles={args.max_cycles},  "
        f"param_data_ratio={args.param_data_ratio}"
    )
    print(f"              epochs_values={args.epochs_values}")
    print("=" * 70)

    results_json: dict[str, dict] = {}
    t0 = time.perf_counter()

    for epochs in args.epochs_values:
        print(f"\n=========  epochs = {epochs}  =========")
        r = run_simulation(
            m=m,
            threshold=args.threshold,
            max_cycles=args.max_cycles,
            epochs=epochs,
            n_samples=n_samples,
            seed=args.seed,
            stop_on_no_improvement=False,
            verbose=True,
        )
        results_json[str(epochs)] = {
            "epochs": epochs,
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
                }
                for h in r["history"]
            ],
        }

    total_elapsed = time.perf_counter() - t0
    bundle = {
        "config": {
            "epochs_values": list(args.epochs_values),
            "m": m,
            "n_PS": n_PS,
            "n_params": n_params,
            "n_samples": n_samples,
            "threshold": args.threshold,
            "max_cycles": args.max_cycles,
            "param_data_ratio": args.param_data_ratio,
            "seed": args.seed,
        },
        "total_elapsed": total_elapsed,
        "results": results_json,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"sweep done in {total_elapsed:.1f} s")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
