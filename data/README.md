# Data directory

Expected files for one training run:

| File | Shape | dtype | Description |
|---|---|---|---|
| `V.npy` | `(N, n_PS)` | float32 | Applied voltage configurations. |
| `port.npy` | `(N,)` | int64 | Input port index in `[0, m)` for each sample. |
| `p.npy` | `(N, m)` | float32 | Measured output intensity distribution per sample (each row sums to 1). |
| `c_0_seed.npy` | `(n_PS,)` | float32 | V-IFM seed for the passive phase vector. |
| `c2_diag_seed.npy` | `(n_PS,)` | float32 | V-IFM seed for the diagonal of `C_2`. |

The PS/BS indexing convention in the seeds and dataset **must** match the
convention used by the `ChipMesh` instance built at training time
(see [`src/chip_mesh.py`](../src/chip_mesh.py) and CLAUDE.md §3.5).

Train/test split: held out at load time (paper uses 16,500 / 4,125 for the
12-mode chip; see CLAUDE.md §12).
