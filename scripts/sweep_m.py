"""Offline m-sweep: run the iterative protocol across chip sizes, save results.

Runs ``scripts/run_iterative.py::run_simulation`` for every ``m`` in a grid and
writes a JSON bundle. Plotting lives in ``notebooks/exploration.ipynb``, which
just loads that file -- so a 20-40 min sweep runs once, headless, and every
later figure tweak is instant.

Constant parameter:data ratio
-----------------------------
Dataset size scales with the trainable parameter count via
``src/synthetic.py::n_samples_from_ratio``: the total dataset is sized so
the 80/20 train/test split yields ``round(param_data_ratio * n_params)``
training examples, where ``n_params = n_PS**2 + n_BS + m`` and
``n_PS = n_BS = m(m-1)``. The default ``--param-data-ratio 1.03`` keeps
the training set slightly larger than the parameter count.

This de-confounds "intrinsically harder" from "data-starved": at *fixed*
n_samples the larger chips are badly under-determined (C_2 alone has
n_PS**2 entries) so they never converge and burn the full cycle budget --
which is what made the earlier fixed-N sweep slow. Holding the ratio
constant puts every m on the same statistical footing.

n_params is dominated by C_2's n_PS**2 entries, so n_samples grows ~m**4.
Across the m <= 12 range this stays in the low tens of thousands (m=12 ~
18k); only well past m=12 does it become the dominant cost.

The 80/20 train/test split is set by ``TEST_FRAC`` in ``run_iterative.py``.

Usage::

    uv run python scripts/sweep_m.py [--m-values 4 6 8 10] [--epochs 80]
        [--param-data-ratio 1.03] [--threshold 0.01] [--max-cycles 10]
        [--seed 0] [--out outputs/sweep_m.json]
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
            "Offline m-sweep for the iterative characterisation protocol."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--m-values", type=int, nargs="+", default=[4, 6, 8, 10],
        help="chip sizes to sweep (each even, >= 2)",
    )
    p.add_argument(
        "--epochs", type=int, default=80,
        help="epochs per ML stage, held fixed across all m",
    )
    p.add_argument(
        "--param-data-ratio", type=float, default=1.03,
        help="training points per trainable parameter; holds the "
             "parameter:data ratio constant across m. See "
             "src/synthetic.py::n_samples_from_ratio for the exact sizing.",
    )
    p.add_argument(
        "--threshold", type=float, default=1e-2,
        help="post-ML TVD convergence target",
    )
    p.add_argument(
        "--max-cycles", type=int, default=10,
        help="hard cap on outer-loop iterations per m",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed (shared across all m)",
    )
    p.add_argument(
        "--out", type=Path, default=Path("outputs/sweep_m.json"),
        help="output JSON path",
    )
    args = p.parse_args()
    for m in args.m_values:
        if m < 2 or m % 2 != 0:
            p.error(f"--m-values entries must be even and >= 2 (got {m})")
    if args.param_data_ratio <= 0:
        p.error(
            f"--param-data-ratio must be > 0 (got {args.param_data_ratio})"
        )
    if not 0 < args.threshold < 1:
        p.error(f"--threshold must lie in (0, 1) (got {args.threshold})")
    if args.max_cycles < 1:
        p.error(f"--max-cycles must be >= 1 (got {args.max_cycles})")
    return args


def main() -> None:
    args = _parse_args()
    log_path = args.out.with_suffix(".log")
    setup_logging(log_path)
    print(
        f"m-sweep: m_values={args.m_values}, epochs={args.epochs}, "
        f"param_data_ratio={args.param_data_ratio}, "
        f"threshold={args.threshold * 100:g}%, max_cycles={args.max_cycles}"
    )
    print(f"  output: {args.out}")
    print(f"  log:    {log_path}")
    print("=" * 70)

    results_json: dict[str, dict] = {}
    t0 = time.perf_counter()
    for m in args.m_values:
        n_PS = m * (m - 1)
        # Full trainable parameter count: C_2 (n_PS**2) + R (n_BS) + T_out (m).
        n_params = n_PS * n_PS + n_PS + m
        n_samples = n_samples_from_ratio(m, args.param_data_ratio)
        print(
            f"\n=========  m = {m}   "
            f"(n_PS = {n_PS}, n_params = {n_params}, "
            f"n_samples = {n_samples})  ========="
        )
        r = run_simulation(
            m=m,
            threshold=args.threshold,
            max_cycles=args.max_cycles,
            epochs=args.epochs,
            n_samples=n_samples,
            seed=args.seed,
            # Full budget every m -> a clean "cycles to threshold" reading
            # rather than an early MSE-stall exit.
            stop_on_no_improvement=False,
            verbose=True,
        )
        # Keep only the JSON-serialisable, plot-relevant subset (the
        # truth/model nn.Modules in the full return dict are dropped).
        results_json[str(m)] = {
            "m": m,
            "n_PS": n_PS,
            "n_params": n_params,
            "n_samples": n_samples,
            "initial_tvd": r["initial_tvd"],
            "converged": r["converged"],
            "cycles_to_threshold": r["cycles_to_threshold"],
            "exit_reason": r["exit_reason"],
            "elapsed": r["elapsed"],
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
            "m_values": list(args.m_values),
            "epochs": args.epochs,
            "param_data_ratio": args.param_data_ratio,
            "threshold": args.threshold,
            "max_cycles": args.max_cycles,
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
