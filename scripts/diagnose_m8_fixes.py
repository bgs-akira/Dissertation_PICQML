"""Test candidate fixes for the m=8 TVD floor.

Builds on diagnose_m8.py's LR-scan finding (LR oscillation hurts m=8). Tests:

  * cosine LR annealing within an ML stage
  * step LR decay within an ML stage
  * lr scaling vs m^2 (motivated by gradient norm scaling with #params)
  * V-IFM noisy seed variant (matches the iterative loop's actual situation)

Each variant uses the exact c_0 seed unless ``--noisy-seed`` is passed, to
isolate the ML stage from phi-IFM.
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

import torch
from torch.utils.data import DataLoader

from src.chip_mesh import ChipMesh
from src.data import make_synthetic_dataset, train_test_split
from src.evaluate import evaluate, mse
from src.model import DigitalTwin
from src.synthetic import ground_truth_model


from src.data import X_MAX
DTYPE = torch.float64


def train_variant(
    model, train_loader, test_loader, truth,
    *,
    epochs, lr_C2, lr_R, lr_Tout,
    scheduler_kind=None,
    restore_best=True,
):
    optim = torch.optim.Adam([
        {"params": [model.C_2_raw], "lr": lr_C2},
        {"params": [model.R_logit], "lr": lr_R},
        {"params": [model.T_logit], "lr": lr_Tout},
    ])

    sched = None
    if scheduler_kind == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=epochs, eta_min=0.0,
        )
    elif scheduler_kind == "cosine_warm":
        # Cosine annealing with warm restarts every 50 epochs.
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=50, T_mult=1, eta_min=0.0,
        )
    elif scheduler_kind == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            optim, step_size=max(1, epochs // 4), gamma=0.5,
        )
    elif scheduler_kind == "exp":
        # Exponential decay: total decay factor = 0.1 over the stage.
        gamma = 0.1 ** (1.0 / max(1, epochs - 1))
        sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)
    elif scheduler_kind == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.5, patience=10, threshold=1e-3,
        )

    best_mse = float("inf")
    best_state = None
    best_epoch = -1

    train_mse_hist, test_mse_hist, test_tvd_hist = [], [], []

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        tr_total = 0.0
        cnt = 0
        for V, port, p in train_loader:
            optim.zero_grad()
            p_hat = model(V, port)
            loss = mse(p, p_hat).mean()
            loss.backward()
            optim.step()
            tr_total += loss.item() * V.shape[0]
            cnt += V.shape[0]
        tr = tr_total / cnt

        test_stats = evaluate(model, test_loader)
        train_mse_hist.append(tr)
        test_mse_hist.append(test_stats["mse_mean"])
        test_tvd_hist.append(test_stats["tvd_mean"])

        if test_stats["mse_mean"] < best_mse:
            best_mse = test_stats["mse_mean"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if sched is not None:
            if scheduler_kind == "plateau":
                sched.step(test_stats["mse_mean"])
            else:
                sched.step()

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_mse": train_mse_hist,
        "test_mse": test_mse_hist,
        "test_tvd": test_tvd_hist,
        "best_epoch": best_epoch,
        "best_test_mse": best_mse,
        "final_state_test_tvd": test_tvd_hist[-1],
        "best_state_test_tvd": (
            evaluate(model, test_loader)["tvd_mean"] if restore_best else None
        ),
        "elapsed": time.perf_counter() - t0,
    }


def build_setup(m, *, batch_size, noisy_seed, seed=0,
                noise_c0=0.30, noise_c2_diag=2e-3,
                n_train_target=None):
    torch.manual_seed(seed)
    # Pinned to Clements: these scripts document the historic m=8 Adam
    # overshoot on the Clements mesh and must keep reproducing it.
    truth = ground_truth_model(m, seed=seed, dtype=DTYPE, scheme="clements")
    n_PS = m * (m - 1)
    if n_train_target is None:
        n_params = n_PS * n_PS + n_PS + m
        n_train_target = n_params
    n_samples = int(round(n_train_target / 0.8))
    dataset = make_synthetic_dataset(
        truth, n_samples=n_samples, x_max=X_MAX, seed=seed + 1,
    )
    train_ds, test_ds = train_test_split(
        dataset, test_frac=0.20, seed=seed + 2,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if noisy_seed:
        g = torch.Generator().manual_seed(seed + 100)
        c0_seed = truth.c_0.detach() + (
            torch.randn(n_PS, generator=g, dtype=DTYPE) * noise_c0
        )
        c2_diag_seed = truth.C_2.detach().diag() + (
            torch.randn(n_PS, generator=g, dtype=DTYPE) * noise_c2_diag
        )
    else:
        c0_seed = truth.c_0.clone()
        c2_diag_seed = truth.C_2.diag().clone()

    fresh = DigitalTwin(ChipMesh.clements(m), c0_seed, c2_diag_seed,
                        dtype=DTYPE)
    return truth, fresh, train_loader, test_loader, len(train_ds), len(test_ds)


def run_variant(name, m, *, epochs, lr_mult, batch_size,
                scheduler_kind, noisy_seed, seed=0,
                out_dir=Path("outputs/diagnose_m8_fixes")):
    print(f"\n=== {name} ===", flush=True)
    print(
        f"  m={m} epochs={epochs} lr_mult={lr_mult} batch={batch_size} "
        f"sched={scheduler_kind} noisy={noisy_seed}",
        flush=True,
    )
    truth, fresh, tr_loader, te_loader, n_tr, n_te = build_setup(
        m, batch_size=batch_size, noisy_seed=noisy_seed, seed=seed,
    )
    print(f"  n_train={n_tr} n_test={n_te}", flush=True)
    init = evaluate(fresh, te_loader)
    print(
        f"  init: mse={init['mse_mean']:.3e} tvd={init['tvd_mean']*100:.3f}%",
        flush=True,
    )

    lr_C2 = 1e-5 * lr_mult
    lr_R = 1e-3 * lr_mult
    lr_T = 1e-3 * lr_mult
    hist = train_variant(
        fresh, tr_loader, te_loader, truth,
        epochs=epochs, lr_C2=lr_C2, lr_R=lr_R, lr_Tout=lr_T,
        scheduler_kind=scheduler_kind,
    )
    tv = hist["test_tvd"]
    best_state_tvd = hist["best_state_test_tvd"]
    print(
        f"  best epoch={hist['best_epoch']}  "
        f"best test_mse={hist['best_test_mse']:.3e}  "
        f"best_state TVD={best_state_tvd*100:.3f}%  "
        f"final TVD={tv[-1]*100:.3f}%  ({hist['elapsed']:.1f}s)",
        flush=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "name": name, "m": m, "epochs": epochs, "lr_mult": lr_mult,
        "batch_size": batch_size, "scheduler_kind": scheduler_kind,
        "noisy_seed": noisy_seed, "n_train": n_tr, "n_test": n_te,
        "seed": seed, "history": hist,
    }
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment",
        choices=["decay_scan", "noisy_scan", "validate"],
        default="decay_scan",
    )
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.experiment == "decay_scan":
        # Test schedulers at LR=1x to see which one fixes oscillation cleanly.
        for sched, name in [
            (None, "exact_none_lr1x"),
            ("cosine", "exact_cosine_lr1x"),
            ("step", "exact_step_lr1x"),
            ("exp", "exact_exp_lr1x"),
            ("plateau", "exact_plateau_lr1x"),
        ]:
            run_variant(name, args.m, epochs=args.epochs,
                        lr_mult=1.0, batch_size=32,
                        scheduler_kind=sched, noisy_seed=False,
                        seed=args.seed)

    if args.experiment == "noisy_scan":
        # Match the iterative loop's noisy V-IFM seed, see if lower LR / cosine
        # still help.
        for sched, lr_mult, name in [
            (None, 1.0, "noisy_none_lr1x"),
            (None, 0.3, "noisy_none_lr0p3x"),
            ("cosine", 1.0, "noisy_cosine_lr1x"),
            ("cosine", 0.3, "noisy_cosine_lr0p3x"),
        ]:
            run_variant(name, args.m, epochs=args.epochs,
                        lr_mult=lr_mult, batch_size=32,
                        scheduler_kind=sched, noisy_seed=True,
                        seed=args.seed)

    if args.experiment == "validate":
        # The best candidate from earlier scans, at full 200 epoch budget.
        run_variant("validate_cosine_lr1x", args.m, epochs=args.epochs,
                    lr_mult=1.0, batch_size=32,
                    scheduler_kind="cosine", noisy_seed=False,
                    seed=args.seed)


if __name__ == "__main__":
    main()
