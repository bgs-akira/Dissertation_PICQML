# Data directory

## ML-stage dataset

Expected files for one training run:

| File | Shape | dtype | Description |
|---|---|---|---|
| `x.npy` | `(N, n_PS)` | float64 | Applied heater **powers**, watts. |
| `port.npy` | `(N,)` | int64 | Input port index in `[0, m)` for each sample. |
| `p.npy` | `(N, m)` | float64 | Measured output intensity distribution per sample (each row sums to 1). |
| `c_0_seed.npy` | `(n_PS,)` | float64 | Calibration seed for the passive phase vector — the document's `b`, rad. |
| `c2_diag_seed.npy` | `(n_PS,)` | float64 | Calibration seed for the diagonal of `C_2` — the document's `k`, rad/W. |

The drive variable is **electric power**, not voltage or current. The chip's
phase law `theta = k*x + b` is linear in power, so that is the natural unit
for the dataset. The instrument sets a *current*; convert with
[`src/power_lookup.py`](../src/power_lookup.py), which holds the per-channel
`(I, x)` tables. Keeping the current→power nonlinearity outside the dataset
stops it being absorbed into `C_2`, where it would masquerade as crosstalk.

Chip budget (m = 10 Prakash): 100 mA and 15 V per channel over a ~140 Ω
heater. The current limit binds first, giving `x ∈ [0, ~1.4 W]` — roughly
4π of phase, or two full fringes. About 700 mW buys 2π, so
`k ≈ 8.98 rad/W`.

## Indexing convention

The PS/BS indexing in the seeds and dataset **must** match the convention
used by the `ChipMesh` built at training time (see
[`src/chip_mesh.py`](../src/chip_mesh.py) and CLAUDE.md §3.5).

For the Bell mesh the global index relates to the calibration document's
`(i, j)` label — `i` the mode, `j` the layer, both in `[0, m)` — as

```
global_index = j * m + i
```

Use `mesh.ps_label(idx)` and `mesh.ps_index(i, j)` rather than hardcoding
that arithmetic. Roles are available via `mesh.ps_roles[idx]`
(`"internal"` or `"independent"`) and `mesh.ps_indices_with_role(...)`.

At m = 10: 100 phase shifters (90 internal, 2 per MZI, plus 10 independent
on modes 0 and 9 at odd layers) and 90 beamsplitters.

## φ-IFM / calibration fringe data

One fringe per phase shifter, keyed by global PS index. Each entry holds:

| Key | Shape | Description |
|---|---|---|
| `x_sweep` | `(n_points,)` | Swept heater power, watts. |
| `intensity_sweep` | `(n_points,)` | Monitored output intensity, normalised. |
| `input_port` | scalar | Mode light is injected into. |
| `output_port` | scalar | Monitored mode. |

The document uses `n_points = 15` per fringe and fits
`P = a + c·sin(θ(i,j) − θ(i+1,j) + π/2)`.

## Train/test split

Held out at load time via `src.data.train_test_split` (the runners use
80/20).
