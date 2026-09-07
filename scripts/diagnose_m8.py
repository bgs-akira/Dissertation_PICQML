"""Diagnostic for the m=8 TVD floor: instrumented single ML stages.

Runs the ML stage on m=8 with the exact ground-truth c_0 seed (so phi-IFM
is irrelevant) and records per-epoch gradient norms and parameter recovery
errors. The motivating observation (from run_synthetic_m8.log) is that the
training MSE oscillates and reaches its best at epoch ~32 of 200, after
which it bounces between values 10x worse than the minimum.

Each experiment block writes its own JSON under outputs/diagnose_m8/.
Plot/analyse downstream in the notebook.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.chip_mesh import ChipMesh
from src.data import make_synthetic_dataset, train_test_split
from src.evaluate import evaluate, mse
from src.model import DigitalTwin
from src.synthetic import ground_truth_model, n_samples_from_ratio


V_MAX = 14.0
DTYPE = torch.float64


def param_errors(model: DigitalTwin, truth: DigitalTwin) -> dict:
    """Per-parameter L2 recovery errors."""
    with torch.no_grad():
        c2_err = (model.C_2 - truth.C_2).abs()
        c2_diag_err = c2_err.diagonal().mean().item()
        c2_offdiag_mask = ~torch.eye(model.n_PS, dtype=torch.bool)
        c2_off_err = c2_err[c2_offdiag_mask].mean().item()
        r_err = (model.R - truth.R).abs().mean().item()
        t_err = (model.T_out - truth.T_out).abs().mean().item()
    return {
        "c2_diag_err": c2_diag_err,
        "c2_off_err": c2_off_err,
        "r_err": r_err,
        "t_err": t_err,
    }


def train_with_diagnostics(
    model, train_loader, test_loader, truth,
    *,
    epochs, lr_C2, lr_R, lr_Tout,
    lr_decay_within_stage=None,  # None or "cosine" or "step"
    restore_best=False,
):
    """Train and record per-epoch diagnostics including grad norms."""
    optim = torch.optim.Adam([
        {"params": [model.C_2_raw], "lr": lr_C2},
        {"params": [model.R_logit], "lr": lr_R},
        {"params": [model.T_logit], "lr": lr_Tout},
    ])

    scheduler = None
    if lr_decay_within_stage == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=epochs, eta_min=0.0,
        )
    elif lr_decay_within_stage == "step":
        # Halve LR every (epochs // 4) epochs.
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=max(1, epochs // 4), gamma=0.5,
        )

    history = {
        "train_mse": [], "test_mse": [], "test_tvd": [],
        "grad_norm_c2": [], "grad_norm_r": [], "grad_norm_t": [],
        "c2_diag_err": [], "c2_off_err": [], "r_err": [], "t_err": [],
    }

    best_mse = float("inf")
    best_state = None
    best_epoch = -1

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        train_mse_total = 0.0
        train_count = 0
        gn_c2 = gn_r = gn_t = 0.0
        n_batches = 0

        for V, port, p in train_loader:
            optim.zero_grad()
            p_hat = model(V, port)
            loss = mse(p, p_hat).mean()
            loss.backward()

            gn_c2 += model.C_2_raw.grad.norm().item()
            gn_r += model.R_logit.grad.norm().item()
            gn_t += model.T_logit.grad.norm().item()
            n_batches += 1

            optim.step()
            train_mse_total += loss.item() * V.shape[0]
            train_count += V.shape[0]

        if scheduler is not None:
            scheduler.step()

        train_mse = train_mse_total / train_count
        test_stats = evaluate(model, test_loader)
        errs = param_errors(model, truth)

        history["train_mse"].append(train_mse)
        history["test_mse"].append(test_stats["mse_mean"])
        history["test_tvd"].append(test_stats["tvd_mean"])
        history["grad_norm_c2"].append(gn_c2 / n_batches)
        history["grad_norm_r"].append(gn_r / n_batches)
        history["grad_norm_t"].append(gn_t / n_batches)
        history["c2_diag_err"].append(errs["c2_diag_err"])
        history["c2_off_err"].append(errs["c2_off_err"])
        history["r_err"].append(errs["r_err"])
        history["t_err"].append(errs["t_err"])

        if test_stats["mse_mean"] < best_mse:
            best_mse = test_stats["mse_mean"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    history["elapsed"] = time.perf_counter() - t0
    history["best_epoch"] = best_epoch
    history["best_test_mse"] = best_mse
    return history


def make_model_and_loaders(m, n_train_target, batch_size, seed=0):
    """Build truth, dataset, and loaders with target n_train; perfect c_0 seed."""
    torch.manual_seed(seed)
    truth = ground_truth_model(m, seed=seed, dtype=DTYPE)
    # Build dataset large enough that 80/20 yields n_train_target training samples.
    n_samples = int(round(n_train_target / 0.8))
    dataset = make_synthetic_dataset(
        truth, n_samples=n_samples, v_max=V_MAX, seed=seed + 1,
    )
    train_ds, test_ds = train_test_split(
        dataset, test_frac=0.20, seed=seed + 2,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Fresh model with EXACT truth c_0 + diag(C_2) (no V-IFM noise here -- we
    # want to isolate the ML-stage problem).
    fresh = DigitalTwin(
        ChipMesh.clements(m),
        truth.c_0.clone(),
        truth.C_2.diag().clone(),
        dtype=DTYPE,
    )
    return truth, fresh, train_loader, test_loader, len(train_ds), len(test_ds)


def run_one(name, m, *, epochs, lr_mult, batch_size,
            lr_decay=None, n_train_target=None, seed=0,
            out_dir=Path("outputs/diagnose_m8")):
    """Run one experiment block, save JSON."""
    lr_C2 = 1e-5 * lr_mult
    lr_R = 1e-3 * lr_mult
    lr_T = 1e-3 * lr_mult

    if n_train_target is None:
        n_train_target = round(1.0 * (m * (m-1)) ** 2 + (m * (m-1)) + m)

    print(f"\n=== {name} ===", flush=True)
    print(
        f"  m={m}  epochs={epochs}  lr_mult={lr_mult}  "
        f"batch={batch_size}  decay={lr_decay}  n_train~{n_train_target}",
        flush=True,
    )

    truth, fresh, train_loader, test_loader, n_tr, n_te = make_model_and_loaders(
        m, n_train_target=n_train_target, batch_size=batch_size, seed=seed,
    )
    print(f"  actual: n_train={n_tr}  n_test={n_te}", flush=True)

    initial_stats = evaluate(fresh, test_loader)
    print(
        f"  initial: test_mse={initial_stats['mse_mean']:.3e} "
        f"test_tvd={initial_stats['tvd_mean']*100:.3f}%",
        flush=True,
    )

    hist = train_with_diagnostics(
        fresh, train_loader, test_loader, truth,
        epochs=epochs,
        lr_C2=lr_C2, lr_R=lr_R, lr_Tout=lr_T,
        lr_decay_within_stage=lr_decay,
    )

    print(
        f"  result: best epoch={hist['best_epoch']} "
        f"best test_mse={hist['best_test_mse']:.3e} "
        f"final test_tvd={hist['test_tvd'][-1]*100:.3f}% "
        f"min test_tvd={min(hist['test_tvd'])*100:.3f}% "
        f"({hist['elapsed']:.1f}s)",
        flush=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "name": name, "m": m, "epochs": epochs,
        "lr_mult": lr_mult, "batch_size": batch_size,
        "lr_decay": lr_decay, "n_train": n_tr, "n_test": n_te,
        "seed": seed,
        "history": hist,
    }
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"  saved -> {out_path}", flush=True)
    return bundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment", choices=[
            "lr_scan", "batch_scan", "decay_scan", "data_scan", "quick", "all",
        ],
        default="lr_scan",
    )
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.experiment in ("lr_scan", "all"):
        # H1: lower LR should remove oscillation
        for lr_mult, name in [
            (1.0, "lr_1x"),
            (0.3, "lr_0p3x"),
            (0.1, "lr_0p1x"),
            (0.03, "lr_0p03x"),
        ]:
            run_one(name, args.m, epochs=args.epochs,
                    lr_mult=lr_mult, batch_size=32, seed=args.seed)

    if args.experiment in ("batch_scan", "all"):
        # H6: larger batch -> less stochastic noise
        for bs, name in [(32, "bs32_lr1x"), (128, "bs128_lr1x"),
                          (256, "bs256_lr1x")]:
            run_one(name, args.m, epochs=args.epochs,
                    lr_mult=1.0, batch_size=bs, seed=args.seed)

    if args.experiment in ("decay_scan", "all"):
        # H3: cosine LR decay within an ML stage
        for decay, name in [(None, "decay_none_lr1x"),
                            ("cosine", "decay_cos_lr1x"),
                            ("step", "decay_step_lr1x")]:
            run_one(name, args.m, epochs=args.epochs,
                    lr_mult=1.0, batch_size=32, lr_decay=decay,
                    seed=args.seed)

    if args.experiment in ("data_scan", "all"):
        # H5: more data (sweep n_train relative to n_params)
        n_params = (args.m * (args.m - 1)) ** 2 + (args.m * (args.m - 1)) + args.m
        for ratio, name in [(0.5, "data_0p5x"), (1.0, "data_1x"),
                             (2.0, "data_2x"), (4.0, "data_4x")]:
            n_train = int(round(ratio * n_params))
            run_one(name, args.m, epochs=args.epochs,
                    lr_mult=1.0, batch_size=32,
                    n_train_target=n_train, seed=args.seed)

    if args.experiment == "quick":
        # quick sanity: just LR=1x and LR=0.1x at low epochs
        run_one("quick_lr1x", args.m, epochs=args.epochs,
                lr_mult=1.0, batch_size=32, seed=args.seed)
        run_one("quick_lr0p1x", args.m, epochs=args.epochs,
                lr_mult=0.1, batch_size=32, seed=args.seed)


if __name__ == "__main__":
    main()
