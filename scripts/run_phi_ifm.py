"""End-to-end phi-IFM demo.

Simulates what happens *after* an ML stage. The ML stage has learned C_2,
R, T_out close to truth but has left c_0 frozen at its V-IFM seed. The
phi-IFM stage then refines c_0 by fitting per-PS phase-sweep fringes.

What this script does
---------------------
1. Build a ground-truth DigitalTwin with random perturbations off the
   paper defaults.
2. Build a "plausible ML output" model -- same C_2, R, T_out as ground
   truth (simulating perfect ML convergence on those blocks), but with
   c_0 = c_0_gt + Gaussian noise of std PHI_OFFSET_STD (the residual c_0
   error that phi-IFM must fix).
3. Generate a synthetic phi-IFM dataset that *mimics the real experiment*.
   In a real experiment the operator targets phi_sweep[k] at PS i by
   solving V via c_0_model; the chip actually applies
       phi_actual = phi_sweep[k] + (c_0_gt - c_0_model)[i],
   so the measured fringe is shifted by (c_0_gt - c_0_model)[i] relative
   to what the operator intends. The synthetic generator in src/phi_ifm.py
   bypasses this offset (pitfall 15); this script reinstates it because
   without it there is nothing for phi-IFM to recover.
4. Run run_phi_ifm_stage to recover the offsets.
5. Save a JSON bundle (config, per-PS recovery, dense pre-/post-fit
   fringe for one demo PS) and a sidecar console log. Plotting lives in
   ``notebooks/exploration.ipynb``.

Usage:
    uv run python scripts/run_phi_ifm.py [--m M] [--ps PS_INDEX]
        [--out outputs/run_phi_ifm_m<M>.json]

CLI flags:
    --m   chip size (number of modes); even, >= 2. Default 4. Paper uses 12.
    --ps  phase-shifter index to feature in the demo fringe plot in the
          notebook. Must lie in [0, m*(m-1)). Default: auto-pick the
          characterizable PS with the largest model-fringe amplitude.
    --out output JSON path; default outputs/run_phi_ifm_m<M>.json. A
          sidecar .log file is written alongside.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from src.chip_mesh import ChipMesh, PhaseShifterLayer
from src.cli import setup_logging
from src.model import DigitalTwin
from src.phi_ifm import run_phi_ifm_stage
from src.synthetic import (
    alpha_beta,
    analytical_fringe,
    build_chip_response_cache,
    ground_truth_model,
    make_phi_ifm_data,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_POINTS = 15               # samples per fringe (paper uses 15)
NOISE_STD = 1e-3            # shot-noise standard deviation on each sample
PHI_OFFSET_STD = 0.30       # post-ML residual error on c_0 (rad)
METHOD = "fast"             # "fast" or "precise"
SEED = 0
OUTPUT_DIR = Path("outputs")
FLAT_FRINGE_TOL = 1e-6      # below this *noiseless* model amplitude a PS
                            # is physically uncharacterizable (e.g.
                            # external PSs on edge waveguides with no
                            # following BS: their phase only rotates a
                            # final amplitude and cancels in |.|^2).
                            # CLAUDE.md §3.5 notes the paper reports
                            # n_PS=126 for m=12 instead of 132 for
                            # exactly this reason.


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _plausible_ml_output(truth: DigitalTwin, seed: int) -> DigitalTwin:
    """Model that simulates a 'plausible' post-ML state.

    C_2, R, T_out are inherited from the ground truth (the ML stage has
    converged on those). c_0 is shifted by Gaussian noise of std
    PHI_OFFSET_STD -- the residual error from the V-IFM seed that the
    ML stage was frozen against and that phi-IFM must now correct.
    """
    g = torch.Generator().manual_seed(seed)
    offset = torch.randn(
        truth.n_PS, generator=g, dtype=truth.c_0.dtype
    ) * PHI_OFFSET_STD

    model = DigitalTwin(
        truth.mesh,
        truth.c_0.detach() + offset,
        torch.zeros(truth.n_PS, dtype=truth.c_0.dtype),  # placeholder
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.C_2_raw.copy_(truth.C_2_raw.detach())
        model.R_logit.copy_(truth.R_logit.detach())
        model.T_logit.copy_(truth.T_logit.detach())
    return model


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


def _locate_ps(mesh: ChipMesh, ps_index: int) -> str:
    """Human-readable position of PS ps_index in the Clements mesh.

    Layer indices come in groups of four per MZI row: BS_a, PS_int, BS_b,
    PS_ext. So `layer_idx % 4 == 1` is an internal PS and
    `layer_idx % 4 == 3` is an external PS.
    """
    for layer_idx, layer in enumerate(mesh.layers):
        if not isinstance(layer, PhaseShifterLayer):
            continue
        for wg, idx in layer.shifter_indices.items():
            if idx == ps_index:
                role = "internal" if (layer_idx % 4 == 1) else "external"
                mzi_row = layer_idx // 4
                return (
                    f"layer {layer_idx} (MZI row {mzi_row}, {role}), "
                    f"waveguide {wg}"
                )
    raise ValueError(f"PS index {ps_index} not found in mesh")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic phi-IFM demo with visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--m", type=int, default=4,
        help="chip size (number of modes/waveguides); even, >= 2",
    )
    p.add_argument(
        "--ps", type=int, default=None,
        help="phase-shifter index to feature in the fringe plot. "
             "Default: auto-pick the characterizable PS with the largest "
             "model-fringe amplitude.",
    )
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON path. Default: "
                        "outputs/run_phi_ifm_m<M>.json. A sidecar .log "
                        "file is written alongside.")
    args = p.parse_args()
    if args.m < 2 or args.m % 2 != 0:
        p.error(f"--m must be even and >= 2 (got {args.m})")
    n_PS = args.m * (args.m - 1)
    if args.ps is not None and not (0 <= args.ps < n_PS):
        p.error(
            f"--ps must lie in [0, {n_PS}) for m={args.m} (got {args.ps})"
        )
    if args.out is None:
        args.out = OUTPUT_DIR / f"run_phi_ifm_m{args.m}.json"
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    m = args.m
    log_path = args.out.with_suffix(".log")
    setup_logging(log_path)

    print(f"phi-IFM synthetic demo (m={m}, method={METHOD})")
    print(f"  output: {args.out}")
    print(f"  log:    {log_path}")
    print("=" * 60)

    # ---- Build the models ------------------------------------------------
    truth = ground_truth_model(m, seed=SEED)
    model = _plausible_ml_output(truth, seed=SEED + 100)

    print()
    print("Chip mesh (Clements rectangular):")
    print(truth.mesh.to_ascii())
    print()
    print(
        f"  n_modes  = {truth.m}     "
        f"n_PS = {truth.n_PS}     "
        f"n_BS = {truth.n_BS}     "
        f"n_layers = {len(truth.mesh.layers)}"
    )

    # ---- Build chip-response caches -------------------------------------
    # One pair of caches per model (truth, model). For m=10 each is 91
    # _build_U calls; everything that follows is analytical.
    print()
    print("Building chip-response caches ...")
    U_0_truth, U_pi_truth = build_chip_response_cache(truth)
    U_0_model, U_pi_model = build_chip_response_cache(model)

    # ---- Generate "experimental" data ------------------------------------
    data, true_offsets = make_phi_ifm_data(
        truth, U_0_truth, U_pi_truth,
        c_0_model=model.c_0.detach(),
        n_points=N_POINTS,
        noise_std=NOISE_STD,
        seed=SEED + 7,
    )

    # Noisy data amplitude (what the fit actually sees)
    amplitudes = np.array([
        data[i]["intensity_sweep"].max() - data[i]["intensity_sweep"].min()
        for i in range(truth.n_PS)
    ])
    # Noiseless model amplitude (what the chip actually permits). For a
    # truly uncharacterizable PS this is zero up to float precision; the
    # noisy amplitude can still be ~noise_std. Evaluated analytically
    # from the cache: 0 extra _build_U calls.
    T_out_model = model.T_out.detach()
    phi_for_amp = np.linspace(0.0, 2 * np.pi, N_POINTS)
    model_amplitudes = np.empty(truth.n_PS)
    for ps in range(truth.n_PS):
        a, b = alpha_beta(U_0_model, U_pi_model[ps])
        f = analytical_fringe(
            a, b, T_out_model, phi_for_amp,
            data[ps]["input_port"], data[ps]["output_port"],
        )
        model_amplitudes[ps] = float(np.ptp(f))

    # ---- Run phi-IFM -----------------------------------------------------
    c_0_new = run_phi_ifm_stage(model, data, method=METHOD)
    recovered = (c_0_new - model.c_0).detach().cpu().numpy()

    # ---- Classify PSs ----------------------------------------------------
    informative_mask = model_amplitudes > FLAT_FRINGE_TOL
    informative_idx = np.where(informative_mask)[0]
    if informative_idx.size == 0:
        raise RuntimeError(
            "No PS has a non-flat fringe. Check the chip parameters."
        )

    # User-chosen PS via --ps takes precedence over the auto-pick. Even if
    # it falls in the uncharacterizable set we still plot it -- that's a
    # useful diagnostic.
    if args.ps is not None:
        demo_ps = args.ps
        if not informative_mask[demo_ps]:
            print(
                f"  WARNING: --ps {demo_ps} has a flat fringe "
                f"(model amplitude = {model_amplitudes[demo_ps]:.2e}); "
                "the fit cannot recover it."
            )
    else:
        demo_ps = int(
            informative_idx[np.argmax(model_amplitudes[informative_idx])]
        )

    fringe = data[demo_ps]
    print()
    print("Phase shifter under inspection")
    print("-" * 60)
    print(f"  PS index        : {demo_ps}")
    print(f"  location        : {_locate_ps(truth.mesh, demo_ps)}")
    print(f"  input port      : {fringe['input_port']}")
    print(f"  output port     : {fringe['output_port']}")
    print(f"  fringe amplitude: {amplitudes[demo_ps]:.3f}")
    print(
        f"  true offset     : {true_offsets[demo_ps]:+.4f} rad  "
        "(c_0_gt - c_0_model)"
    )
    print(
        f"  recovered offset: {recovered[demo_ps]:+.4f} rad  "
        f"(method={METHOD})"
    )
    print(
        f"  recovery error  : "
        f"{1000 * (recovered[demo_ps] - true_offsets[demo_ps]):+.2f} mrad"
    )

    # ---- Recovery summary across all informative PSs --------------------
    errors_inf = recovered[informative_mask] - true_offsets[informative_mask]
    rms_mrad = float(np.sqrt((errors_inf ** 2).mean()) * 1000)
    print()
    print("Per-PS recovery table")
    print("-" * 76)
    print(
        f"  {'PS':>3}  {'model amp':>9}  {'in':>3}  {'out':>3}  "
        f"{'true (rad)':>11}  {'recov (rad)':>11}  {'err (mrad)':>10}  status"
    )
    for ps in range(truth.n_PS):
        err_mrad = 1000 * (recovered[ps] - true_offsets[ps])
        status = "OK" if informative_mask[ps] else "flat fringe"
        print(
            f"  {ps:>3}  {model_amplitudes[ps]:>9.2e}  "
            f"{data[ps]['input_port']:>3}  {data[ps]['output_port']:>3}  "
            f"{true_offsets[ps]:>+11.4f}  {recovered[ps]:>+11.4f}  "
            f"{err_mrad:>+10.2f}  {status}"
        )

    n_uncharacterizable = int((~informative_mask).sum())
    print()
    print("Recovery summary")
    print("-" * 76)
    print(
        f"  characterizable PSs: {informative_mask.sum()} / {truth.n_PS}  "
        f"(noiseless model fringe amplitude > {FLAT_FRINGE_TOL:.0e})"
    )
    if n_uncharacterizable:
        unc_idx = [int(i) for i in np.where(~informative_mask)[0]]
        print(
            f"  uncharacterizable PSs: {n_uncharacterizable} "
            f"(indices {unc_idx}) -- external PSs sitting after the last "
            "BS that mixes their waveguide, so their phase only rotates "
            "a final amplitude and cancels in |U|^2. c_0_model is left "
            "unchanged for these (which is the right answer: they have "
            "no effect on output statistics)."
        )
    print(
        f"  RMS true offset (residual c_0 error before phi-IFM): "
        f"{1000 * float(np.sqrt((true_offsets[informative_mask]**2).mean())):.1f} mrad"
    )
    print(f"  RMS recovery error on characterizable PSs:  {rms_mrad:.2f} mrad")

    # ---- Dense model fringe for the demo PS (for notebook plotting) -----
    phi_dense = np.linspace(0.0, 2 * np.pi, 200)
    alpha_demo, beta_demo = alpha_beta(
        U_0_model, U_pi_model[demo_ps],
    )
    f_before = analytical_fringe(
        alpha_demo, beta_demo, T_out_model, phi_dense,
        fringe["input_port"], fringe["output_port"],
    )
    f_after = analytical_fringe(
        alpha_demo, beta_demo, T_out_model,
        phi_dense + recovered[demo_ps],
        fringe["input_port"], fringe["output_port"],
    )

    # ---- Save JSON ------------------------------------------------------
    bundle = {
        "config": {
            "m": m,
            "n_PS": truth.n_PS,
            "n_BS": truth.n_BS,
            "n_layers": len(truth.mesh.layers),
            "n_points": N_POINTS,
            "noise_std": NOISE_STD,
            "phi_offset_std": PHI_OFFSET_STD,
            "method": METHOD,
            "seed": SEED,
            "flat_fringe_tol": FLAT_FRINGE_TOL,
        },
        "summary": {
            "demo_ps": int(demo_ps),
            "demo_ps_location": _locate_ps(truth.mesh, demo_ps),
            "n_characterizable": int(informative_mask.sum()),
            "n_uncharacterizable": int((~informative_mask).sum()),
            "uncharacterizable_idx": [
                int(i) for i in np.where(~informative_mask)[0]
            ],
            "rms_true_offset_mrad": 1000.0 * float(
                np.sqrt((true_offsets[informative_mask] ** 2).mean())
            ),
            "rms_recovery_error_mrad": rms_mrad,
        },
        "per_ps": [
            {
                "ps": int(ps),
                "model_amplitude": float(model_amplitudes[ps]),
                "data_amplitude": float(amplitudes[ps]),
                "input_port": int(data[ps]["input_port"]),
                "output_port": int(data[ps]["output_port"]),
                "true_offset": float(true_offsets[ps]),
                "recovered_offset": float(recovered[ps]),
                "status": "OK" if informative_mask[ps] else "flat_fringe",
            }
            for ps in range(truth.n_PS)
        ],
        "demo_fringe": {
            "phi_sweep": fringe["phi_sweep"].tolist(),
            "intensity_sweep": fringe["intensity_sweep"].tolist(),
            "phi_dense": phi_dense.tolist(),
            "f_before": f_before.tolist(),
            "f_after": f_after.tolist(),
            "true_offset": float(true_offsets[demo_ps]),
            "recovered_offset": float(recovered[demo_ps]),
            "input_port": int(fringe["input_port"]),
            "output_port": int(fringe["output_port"]),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
