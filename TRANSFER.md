# TRANSFER.md — project handoff

Snapshot of the Dissertation_PICQML repo (implementation of the ML stage of
the Fyrillas et al. clear-box PIC-characterisation protocol) as of **2026-09-07**.

Read this alongside [CLAUDE.md](CLAUDE.md), which is the canonical spec.
TRANSFER.md is the operational sibling: what is built, what still breaks,
what to touch next.

---

## 1. What has been built

### 1.1 Core ML stage (all in `src/`)

| File | Status | Notes |
|---|---|---|
| [src/chip_mesh.py](src/chip_mesh.py) | ✅ complete | `ChipMesh.clements(m)` builds the Clements rectangular mesh; `to_ascii()` for debugging. Convention: `n_PS = n_BS = m*(m-1)`. |
| [src/model.py](src/model.py) | ✅ complete | `DigitalTwin` — differentiable forward model. Vectorised `_build_U` over both batch and within-layer elements. `float32` / `float64` supported. |
| [src/training.py](src/training.py) | ✅ complete | `train()` with Adam parameter groups + optional cosine LR schedule (`lr_schedule="cosine"`). Returns best_state_dict. |
| [src/data.py](src/data.py) | ✅ complete | `PICDataset`, `make_synthetic_dataset`, `train_test_split`. |
| [src/evaluate.py](src/evaluate.py) | ✅ complete | `mse`, `tvd`, `evaluate`. |

### 1.2 φ-IFM stage (all in `src/`)

| File | Status | Notes |
|---|---|---|
| [src/phase_voltage.py](src/phase_voltage.py) | ✅ complete | Iterative Supplement H solver (NumPy). |
| [src/phi_ifm.py](src/phi_ifm.py) | ✅ complete | `run_phi_ifm_stage`, `fit_fringe_fast` (3-param), `fit_fringe_precise` (1-param). Handles unidentifiable ports. |
| [src/synthetic.py](src/synthetic.py) | ✅ complete | `ground_truth_model`, `n_samples_from_ratio`, chip-response cache (`build_chip_response_cache`, `alpha_beta`, `analytical_fringe`, `best_port_pair`). Cache turned the m=10 port scan from ~9 min to ~1 s. `make_phi_ifm_data` mimics the real experiment (fringes shifted by `c_0_gt − c_0_model`). |

### 1.3 Outer loop + drivers (in `scripts/`)

| File | Status | Notes |
|---|---|---|
| [scripts/run_iterative.py](scripts/run_iterative.py) | ✅ working | Full `V-IFM → (ML + φ-IFM)*` loop with LR decay 0.7 per cycle, TVD-threshold exit, no-improvement exit, fast→precise switchover, cosine option. |
| [scripts/run_synthetic.py](scripts/run_synthetic.py) | ✅ working | One-shot ML stage on synthetic data. |
| [scripts/run_phi_ifm.py](scripts/run_phi_ifm.py) | ✅ working | Standalone φ-IFM demo/plot. |
| [scripts/sweep_m.py](scripts/sweep_m.py), [scripts/sweep_epochs.py](scripts/sweep_epochs.py), [scripts/sweep_m_epochs.py](scripts/sweep_m_epochs.py) | ✅ working | Fig-3-style grid sweeps. |
| [scripts/sweep_noise.py](scripts/sweep_noise.py) | ⚠️ partial | Runs but only completed at σ=0 for m=12 (Ctrl+C on σ=1e-4). |
| [scripts/diagnose_m8.py](scripts/diagnose_m8.py), [scripts/diagnose_m8_fixes.py](scripts/diagnose_m8_fixes.py) | ✅ historic | Used to diagnose the m≥8 Adam overshoot; kept for reproducibility. |
| [scripts/compare_m8_cosine.py](scripts/compare_m8_cosine.py) | ✅ working | Constant-LR vs cosine comparison plot. |

### 1.4 Future-work scaffolding (in `src/`, not on the critical path)

| File | Status | Notes |
|---|---|---|
| [src/pic_graph.py](src/pic_graph.py) | ✅ built, unused by main loop | Graph representation of the chip for real-hardware characterization-order derivation (Supplement C.3). |
| [src/routing.py](src/routing.py), [src/routing_viz.py](src/routing_viz.py) | ✅ built, unused | Direct-path routing for real hardware. Not required for synthetic runs (CLAUDE.md pitfall 15). |
| [src/cli.py](src/cli.py) | ✅ utility | Shared `setup_logging` used by every script. |

### 1.5 Tests (`tests/`) — all green last time they ran

`test_chip_mesh.py`, `test_model.py` (including the unitarity sanity check
from CLAUDE.md §3.5), `test_training.py`, `test_data.py`, `test_evaluate.py`,
`test_phi_ifm.py`, `test_phase_voltage.py`, `test_pic_graph.py`,
`test_routing.py`.

Run: `uv run pytest tests/`.

### 1.6 Result artefacts (`outputs/`)

- Iterative loop: `iterative_m{4,6,8}*.json/.log/.png`, plus `_history.npz` for m=4 and m=8.
- Big grid: [outputs/sweep_m_epochs_cosine.json](outputs/sweep_m_epochs_cosine.json) — 15/16 cells done, m=12 epochs=500 interrupted mid-cycle-1.
- φ-IFM demos: `phi_ifm_demo_m*.png`.
- Noise sweeps: `sweep_noise_m{6,8,10,12}.json` — the m=12 σ=0 run reached **post-ML TVD = 0.0001 %** at cycle 10.

---

## 2. Where the problems still are

### 2.1 Uncommitted work

`git status` shows a large uncommitted diff and 12+ untracked new files
(including all of `scripts/`, half of `src/`, and `pyproject.toml` /
`uv.lock`). **Only two commits exist in main.** First priority for whoever
picks this up: land the current tree in git so future work is diffable.

### 2.2 Float32 diverges on the iterative loop (m=4)

[outputs/iterative_m4_float32.log](outputs/iterative_m4_float32.log): cycle 1
converges to post-ML TVD 9.4 %, cycle 2 goes **backwards** to 12.1 %.
Float64 (default) converges cleanly. Root cause not diagnosed; the model
supports both dtypes but the outer loop's Adam state / cosine schedule /
phi-IFM fit interact poorly at float32 precision. Workaround: leave
`--dtype float64`.

### 2.3 Post-φ-IFM TVD spike on cycle 1 (all m ≥ 6)

Every cosine run shows the same pattern: ML stage brings TVD to ≈3 %, then
φ-IFM makes it **worse** (20–32 %), then cycle 2's ML stage cleans it up.
Example ([outputs/iterative_m8_cosine.log](outputs/iterative_m8_cosine.log)):
`post-ML 2.86 % → post-φ-IFM 21.5 % → post-ML(cycle 2) 0.011 %`.

This is not a hard bug — the loop still converges to <0.02 % in 2 cycles —
but the intermediate spike is not documented in the paper and is worth
explaining before writing up. Likely reason: on cycle 1, `C_2` is still
off by enough that the φ-IFM rewrites `c_0` into a valley that the
current `C_2, R, T_out` cannot reproduce. The stagnation-detection knob
(`PHI_IFM_STAGNATION_WINDOW`, `PHI_IFM_STAGNATION_TOL` in
`scripts/run_iterative.py`) is designed around this and it never trips on
clean data because convergence happens too fast.

### 2.4 CPU wall-clock is prohibitive at paper size

From [outputs/sweep_m_epochs_cosine.log](outputs/sweep_m_epochs_cosine.log):

| m | epochs | wall-clock (2 cycles) |
|---|---|---|
| 6 | 500 | 336 s |
| 8 | 500 | 1617 s |
| 10 | 500 | 7999 s |
| 12 | 400 | 12,397 s (~3.4 h) |
| 12 | 500 | **interrupted before completion** |

`_build_U` is already vectorised over the batch and within layers, but
the outer Python loop over `2m` layers still dominates. Every cell of the
16-cell m∈{6,8,10,12} × epochs∈{200,300,400,500} sweep at m ≥ 10 costs
30+ min per ML stage on CPU.

### 2.5 m = 12 has never completed the full sweep

- [outputs/run_synthetic_m12.log](outputs/run_synthetic_m12.log): crashed with
  `KeyboardInterrupt` inside `loss.backward()` during ML stage.
- [outputs/sweep_m_epochs_cosine.log](outputs/sweep_m_epochs_cosine.log):
  cell [16/16] `m=12 epochs=500` interrupted mid-cycle-1.
- [outputs/sweep_noise_m12.log](outputs/sweep_noise_m12.log): σ=0 completed
  (10 cycles, final TVD 0.0001 %), σ=1e-4 cycle 1 interrupted.

The σ=0 result is a **success** — the iterative loop drives m=12 down to
0.0001 % TVD, well below the paper's 2.9 % — but the noise-sensitivity
study is incomplete.

### 2.6 Minor loose ends

- `T_in` is fixed at identity (correct for the ML stage; see CLAUDE.md §1
  and pitfall 1). The ITM stage that recovers it is out of scope but
  will need its own module.
- `pic_graph.py` / `routing.py` are built and tested but not exercised by
  the main loop; they're there for the eventual real-hardware run.
- The exploration notebook `notebooks/exploration.ipynb` has uncommitted
  edits; treat it as scratch space, not a source of truth.
- `references/` is uncommitted; contains the paper PDFs plus a plaintext
  extract of the supplement.

---

## 3. What to do next

Ordered by priority.

### 3.1 Commit and clean up (30 min)

1. `git add -A && git commit` the current tree, splitting into 2–3 logical
   commits if you want (`src/*` core stage, `scripts/*` drivers, `outputs/*`
   results). The `.gitignore` already exists — check `outputs/` isn't
   accidentally ignored.
2. Fill in `data/README.md` with the phi-IFM / dataset file format that
   the code currently expects.

### 3.2 Finish the m = 12 grid (1 evening + long compute)

The final cell of [outputs/sweep_m_epochs_cosine.json](outputs/sweep_m_epochs_cosine.json)
(m=12, epochs=500) is missing. The sweep script supports resume — re-run:

```powershell
uv run python scripts/sweep_m_epochs.py --lr-schedule cosine
```

Expect ~3–4 h wall-clock on CPU for the remaining cell. Once done, generate
the "recovered vs truth" parameter-histogram plot to compare against
Fig. 3 of the paper.

### 3.3 Complete the noise sweep on m = 12

[scripts/sweep_noise.py](scripts/sweep_noise.py) has the machinery; only the
σ=0 cell has finished for m=12. Re-run to fill in σ ∈ {1e-4, 1e-3, 1e-2}.
This is what will let you claim the algorithm handles paper-realistic
shot noise (paper reports ~2.9 % TVD on measured data).

### 3.4 Diagnose the float32 iterative divergence

Isolate a minimal repro: `run_iterative.run_simulation(m=4, dtype=torch.float32,
seed=0)`. Log the raw Adam state (`m`, `v`) at end of cycle 1 and start of
cycle 2 to see whether the LR-decay + optimizer-rebuild is producing an
unstable first step, or whether the phi-IFM `c_0` update is what breaks
things. If cheap, either fix or add a documented "float64 only" guard on
the outer loop.

### 3.5 Speed up `_build_U` for m ≥ 10 (medium project)

`_build_U` currently:
- constructs a per-layer identity `L` at every step (m×m allocation),
- writes into it with 4 fancy-indexed assignments per BS layer,
- calls `L @ U` (an m×m×m matmul on top of an m-sized diagonal).

Batching the entire layer sequence out of Python — e.g. by pre-materialising
the BS layer matrices once at model build time (they depend only on `R`,
which changes only across optimiser steps, so cache and only recompute
when a step happens) — should cut m=12 ML stages from ~3600 s to well
under 1000 s. GPU support is also within reach if a batched forward is
already in hand; the current CPU-only choice was documented in the
`--dtype` help text.

### 3.6 After all the above: start the ITM stage

Out of scope for the ML implementation per CLAUDE.md §1, but the natural
next milestone. Needs un-normalised intensity sums, which means a second
forward path or a companion `DigitalTwin.forward_unnormalised(...)`.
Nothing in the current codebase blocks this.

---

## 4. How to run the common workflows

```powershell
# Full ML+phi-IFM iterative loop on m=8 with the recommended cosine schedule.
uv run python scripts/run_iterative.py --m 8 --lr-schedule cosine --epochs 200

# Fig 3.B-style grid sweep (resumable — safe to Ctrl+C).
uv run python scripts/sweep_m_epochs.py --lr-schedule cosine

# Standalone phi-IFM demo (generates the phi_ifm_demo_*.png plots).
uv run python scripts/run_phi_ifm.py --m 8

# Tests.
uv run pytest tests/
```

Every long-running script writes both a JSON summary and a `.log` sidecar
to `outputs/`. The JSON is the source of truth for figures; the `.log` is
for debugging.

---

## 5. Contact points inside the code

If you need to change one of these behaviours, edit here first:

- **Per-parameter learning rates**: `LR_C2_INIT`, `LR_R_INIT`, `LR_TOUT_INIT`
  in [scripts/run_iterative.py](scripts/run_iterative.py) (defaults from CLAUDE.md §6.2).
- **LR decay across cycles**: `LR_DECAY = 0.7` same file.
- **Cosine schedule**: [src/training.py](src/training.py) `_build_scheduler`.
- **Ground-truth perturbation magnitudes**: constants at the top of
  [src/synthetic.py](src/synthetic.py).
- **V-IFM seed noise**: `C_0_SEED_NOISE`, `C_2_DIAG_SEED_NOISE` in
  [scripts/run_iterative.py](scripts/run_iterative.py).
- **φ-IFM stagnation detection**: `PHI_IFM_STAGNATION_WINDOW`,
  `PHI_IFM_STAGNATION_TOL` same file.
