"""End-to-end synthetic ML-stage run -- one chip size, one epoch budget.

1. Build a ground-truth DigitalTwin with random perturbations.
2. Generate a synthetic (x, port, p) dataset from it.
3. Train a fresh DigitalTwin seeded with V-IFM-style values for c_0 and
   diag(C_2) (drawn from the truth, in the absence of real V-IFM output).
4. Save per-epoch train/test MSE and test TVD to JSON; plotting lives in
   ``notebooks/exploration.ipynb``.

Usage::

    uv run python scripts/run_synthetic.py [--m 10] [--scheme bell]
        [--param-data-ratio 1.03] [--epochs 50] [--batch-size 32] [--seed 0]
        [--out outputs/run_synthetic_<scheme>_m<M>.json]

Outputs:
    ``--out``        the JSON bundle (config + per-epoch history + summary)
    ``--out``.log    sidecar console log (same basename, .log suffix)

Dataset size is derived from the parameter:data ratio (training points per
trainable parameter) by ``src/synthetic.py::n_samples_from_ratio``: the
total dataset is sized so the 80/20 train/test split yields
``round(param_data_ratio * n_params)`` training examples, where
``n_params = n_PS**2 + n_BS + m``. On the default Bell mesh
``n_PS = m**2`` and ``n_BS = m(m-1)``, so n_samples scales as ~m**4 with
chip size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.chip_mesh import MESH_SCHEMES, build_mesh
from src.cli import setup_logging
from src.data import X_MAX, make_synthetic_dataset, train_test_split
from src.model import DigitalTwin
from src.synthetic import ground_truth_model, n_samples_from_ratio
from src.training import train


OUTPUT_DIR = Path("outputs")
DTYPE = torch.float64   # matches run_iterative.py and run_phi_ifm.py

LR_C2 = 1e-5    # paper defaults (Section 7.A)
LR_R = 1e-3
LR_TOUT = 1e-3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end synthetic ML-stage run on a Bell or "
                    "Clements mesh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--m", type=int, default=10,
                   help="chip size (even, >= 2)")
    p.add_argument("--scheme", choices=list(MESH_SCHEMES), default="bell",
                   help="mesh topology. 'bell' (default) gives "
                        "n_PS = m**2; 'clements' gives n_PS = m*(m-1).")
    p.add_argument("--param-data-ratio", type=float, default=1.03,
                   help="training points per trainable parameter "
                        "(see src/synthetic.py::n_samples_from_ratio)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON path. Default: "
                        "outputs/run_synthetic_m<M>.json. A sidecar .log "
                        "file is written alongside.")
    args = p.parse_args()
    if args.m < 2 or args.m % 2 != 0:
        p.error(f"--m must be even and >= 2 (got {args.m})")
    if args.param_data_ratio <= 0:
        p.error(
            f"--param-data-ratio must be > 0 (got {args.param_data_ratio})"
        )
    if args.epochs < 1:
        p.error(f"--epochs must be >= 1 (got {args.epochs})")
    if args.batch_size < 1:
        p.error(f"--batch-size must be >= 1 (got {args.batch_size})")
    if args.out is None:
        args.out = OUTPUT_DIR / f"run_synthetic_{args.scheme}_m{args.m}.json"
    return args


def main() -> None:
    args = _parse_args()
    log_path = args.out.with_suffix(".log")
    setup_logging(log_path)

    torch.manual_seed(args.seed)
    n_samples = n_samples_from_ratio(
        args.m, args.param_data_ratio, scheme=args.scheme,
    )
    print(
        f"Synthetic run: scheme={args.scheme}, m={args.m}, "
        f"param_data_ratio={args.param_data_ratio} -> n_samples={n_samples}, "
        f"epochs={args.epochs}"
    )
    print(f"  output: {args.out}")
    print(f"  log:    {log_path}")

    truth = ground_truth_model(
        args.m, seed=args.seed, dtype=DTYPE, scheme=args.scheme,
    )
    dataset = make_synthetic_dataset(
        truth, n_samples=n_samples, x_max=X_MAX, seed=args.seed + 1
    )
    train_ds, test_ds = train_test_split(
        dataset, test_frac=0.25, seed=args.seed + 2
    )
    print(f"  train: {len(train_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    fresh = DigitalTwin(
        build_mesh(args.m, args.scheme),
        truth.c_0.clone(),
        truth.C_2.diag().clone(),
        dtype=DTYPE,
    )

    history = train(
        fresh,
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr_C2=LR_C2,
        lr_R=LR_R,
        lr_Tout=LR_TOUT,
        verbose=True,
    )

    print(
        f"  best test MSE = {history['best_test_mse']:.3e} "
        f"at epoch {history['best_epoch']}"
    )
    print(f"  final test TVD = {history['test_tvd'][-1] * 100:.2f}%")

    bundle = {
        "config": {
            "m": args.m,
            "scheme": args.scheme,
            "n_samples": n_samples,
            "param_data_ratio": args.param_data_ratio,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "lr_C2": LR_C2,
            "lr_R": LR_R,
            "lr_Tout": LR_TOUT,
            "x_max": X_MAX,
            "dtype": str(DTYPE),
            "n_train": len(train_ds),
            "n_test": len(test_ds),
        },
        "history": {
            "train_mse": list(history["train_mse"]),
            "test_mse": list(history["test_mse"]),
            "test_tvd": list(history["test_tvd"]),
        },
        "summary": {
            "best_epoch": history["best_epoch"],
            "best_test_mse": history["best_test_mse"],
            "final_test_mse": history["test_mse"][-1],
            "final_test_tvd": history["test_tvd"][-1],
            "elapsed": history["elapsed"],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
