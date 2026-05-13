# CLAUDE.md

This file is the canonical context for an implementation of the **ML (machine-learning) stage** of the photonic-chip characterisation protocol described in:

> A. Fyrillas, O. Faure, N. Maring, J. Senellart, N. Belabas.
> "Scalable machine learning-assisted clear-box characterization for optimally controlled photonic circuits."
> *Optica* **11**, 427 (2024). DOI: 10.1364/OPTICA.512148.

Implementation language: **Python**. Framework: **PyTorch**.

---

## 1. Project goal

Implement the ML stage of the Fyrillas et al. clear-box characterisation protocol. The ML stage takes V-IFM seeds and a labelled dataset of voltage configurations + measured output intensity distributions, then learns the parameters that govern an imperfect linear-optical photonic integrated circuit (PIC) by minimising the MSE between predicted and measured output distributions.

**In scope:**
- A differentiable forward model of the chip (the "digital twin").
- Adam training loop with per-parameter learning rates.
- Train/test split, monitoring of test MSE and TVD.
- Recovery of the crosstalk matrix `C_2`, beamsplitter reflectivity vector `R`, and output transmission vector `T_out`.

**Out of scope:**
- V-IFM data acquisition (the seeds for the diagonal of `C_2` and for `c_0` are inputs to this code).
- φ-IFM phase fitting (a separate stage that refines `c_0` between ML iterations).
- ITM input-transmission measurement (a separate stage that recovers `T_in`).
- Compilation / run-time imperfection mitigation.
- Hardware control or data acquisition.

---

## 2. Paper context

A PIC of `m` modes implements a linear-optical transformation. The transformation is built from a mesh of directional couplers (beamsplitters) and thermo-optic phase shifters (PSs). Each PS is driven by a voltage `V_k` that produces a phase shift `phi_k` through the thermo-optic effect.

Fabrication imperfections mean:
- Beamsplitters do not split at exactly 50:50 → per-coupler reflectivity `R_l`.
- Waveguide-length differences leave per-shifter offsets at zero voltage → passive phase vector `c_0`.
- Heat dissipated by one PS perturbs the phases of its neighbours → **thermal crosstalk**, captured by the off-diagonal of a matrix `C_2`.
- Input/output ports do not couple light with equal efficiency → diagonal vectors `T_in`, `T_out`.

The chip's phase-voltage relation is modelled as

```
phi = C_2 @ (V ** 2) + c_0
```

where `**` is element-wise squaring and `@` is matrix multiplication. The diagonal of `C_2` carries each PS's self-heating coefficient; the off-diagonal carries inter-PS crosstalk.

The chip's full non-unitary transfer matrix `U` is the matrix product of the elementary block matrices along the optical path, with input/output transmissions folded in by row/column scaling. Given an input port `i` and the chip's current voltage configuration `V`, the predicted output intensity distribution is `|U[:, i]| ** 2 / sum(|U[:, i]| ** 2)`.

The ML stage fits a digital twin of this forward model against measured data.

---

## 3. Parameter set

For a chip with `m` modes, `n_PS` phase shifters, `n_BS` beamsplitters:

| Symbol | Shape | dtype | Initialisation | Trainable in ML stage? |
|---|---|---|---|---|
| `C_2` | `(n_PS, n_PS)` | float32 | diag = V-IFM seed; off-diag = 0 | **yes** (all entries) |
| `c_0` | `(n_PS,)` | float32 | V-IFM seed | **no** (frozen here) |
| `R` | `(n_BS,)` | float32 | 0.5 for every BS | **yes** |
| `T_out` | `(m,)` | float32 | 1.0 for every port | **yes** |
| `T_in` | `(m,)` | float32 | 1.0 (not a parameter) | **no** (gradient is zero) |

Reference numbers from the paper's 12-mode Clements chip: `m = 12`, `n_PS = 126`, `n_BS = 132`. Typical values: `C_2` diagonal `≈ 0.034 rad/V²`, off-diagonal `≈ ±5×10⁻⁴ rad/V²`. `R` clusters near `0.56` on the paper's hardware (the operating wavelength differs from the fabrication wavelength).

---

## 3.5 Chip architecture: the Clements rectangular mesh

The 12-mode chip in the paper is a **Clements rectangular mesh**. The mesh is a tiled brick pattern of Mach-Zehnder interferometers (MZIs) drawn between the input ports on the left and the output ports on the right. Light propagates left to right; columns alternate between beamsplitter columns and phase-shifter columns.

### Counting

For `m` modes (`m` even):

| Quantity | Formula | Value at `m = 12` |
|---|---|---|
| MZI unit cells | `m * (m - 1) / 2` | 66 |
| Beamsplitters (2 per MZI) | `m * (m - 1)` | 132 |
| Internal phase shifters (1 per MZI) | `m * (m - 1) / 2` | 66 |
| External phase shifters (1 per MZI) | `m * (m - 1) / 2` | 66 |
| Total phase shifters | `m * (m - 1)` | 132 |

Note: the paper reports `n_PS = 126` for `m = 12`, which corresponds to omitting the six external phase shifters that would otherwise sit on the topmost or bottommost waveguide where they have no effect on the output statistics for an unbiased input. The `ChipMesh` class should match whatever convention the V-IFM seeds were produced with; check the dataset.

### Layer structure

Columns alternate from input to output. There are `2m` columns total. With column indices `c = 0, 1, ..., 2m - 1`:

- **Even-indexed columns** (`c = 0, 2, 4, ...`): a **beamsplitter column**. The beamsplitters act on waveguide pairs `(k, k+1)` for `k` of a fixed parity depending on the column.
- **Odd-indexed columns** (`c = 1, 3, 5, ...`): a **phase-shifter column** with one phase shifter sitting on the upper waveguide of each MZI active in the previous beamsplitter column.

The two parities of the beamsplitter columns:

| Column parity | Beamsplitter waveguide pairs |
|---|---|
| Even within the pattern (`c = 0, 4, 8, ...`) | `(0,1), (2,3), (4,5), ...` |
| Odd within the pattern (`c = 2, 6, 10, ...`) | `(1,2), (3,4), (5,6), ...` |

This brick-pattern alternation is what gives the Clements mesh its universality.

### MZI unit cell

A single MZI unit cell on waveguides `(k, k+1)` consists, in propagation order, of:

1. A beamsplitter (`50:50` in the ideal case).
2. A phase shifter on waveguide `k` (the upper arm). This is the **internal** phase shifter.
3. A second beamsplitter.
4. An **external** phase shifter on waveguide `k` (often drawn outside the unit cell).

In matrix form, a single MZI unit cell on waveguides `(k, k+1)` with internal phase `phi_int`, external phase `phi_ext`, and beamsplitter reflectivities `R_a` and `R_b`:

```
U_MZI = P_ext(phi_ext) @ B(R_b) @ P_int(phi_int) @ B(R_a)
```

The matrix product is taken right-to-left because the rightmost factor acts first on the input mode amplitudes.

### Mapping global indices to topology

The `ChipMesh` class must assign:

- A global index `i_PS ∈ [0, n_PS)` to every phase shifter on the chip.
- A global index `i_BS ∈ [0, n_BS)` to every beamsplitter on the chip.
- For each layer, a dict mapping waveguide-position-in-column → global index.

The order is arbitrary in principle but must be **consistent**: the V-IFM seeds (`c_0`, `C_2` diagonal) and the dataset must use the same ordering convention as the `ChipMesh` instance built at training time. A natural convention is left-to-right column scan, top-to-bottom within each column. Document the choice in `chip_mesh.py` and assert it in the data loader.

### Sanity check at lossless limit

For any Clements mesh constructed by `ChipMesh.clements(m)`, the forward pass with `c_0 = 0`, `C_2 = 0`, `R = 0.5` (every BS), `T_out = 1` (every port) should produce a matrix `U_0` that is unitary up to floating-point precision: `U_0.conj().T @ U_0 ≈ I_m`. Write this as a unit test.

---

## 4. Forward model

The forward pass converts a batch of voltage vectors and input port indices into a batch of predicted output distributions.

### 4.1 Voltages → phases
```
phi = C_2 @ (V ** 2) + c_0     # shape (n_PS,) for a single sample
```
Vectorised over a batch: `phi_batch = V_batch_sq @ C_2.T + c_0`, with `V_batch_sq` of shape `(batch, n_PS)`.

### 4.2 Phases → lossless chip matrix `U_0`
The chip mesh is an ordered sequence of layers (input → output). Each layer is one of:

- **Phase-shifter column**: a diagonal `m × m` matrix. The `(k, k)` entry is `exp(i * phi[shifter_index])` if waveguide `k` hosts an active shifter in this column, otherwise `1`.
- **Beamsplitter column**: identity `m × m` matrix, with the following 2×2 block inserted at rows/columns `(k, k+1)` for each active beamsplitter in this column:
  ```
  [[ sqrt(R)         , i * sqrt(1 - R) ],
   [ i * sqrt(1 - R) , sqrt(R)         ]]
  ```

Within a column, all active blocks act on disjoint waveguides and commute, so the within-column product is a direct sum. Across columns, the matrix product is ordered: `U_0 = L_last @ ... @ L_2 @ L_1`.

### 4.3 Lossless → non-unitary `U` (output transmissions only)
```
U = diag(sqrt(T_out)) @ U_0
```
`T_in` is set to identity in this implementation; it cancels under the normalisation in §4.4 and is recovered in a separate ITM stage.

### 4.4 Output distribution for the given input port
For a sample with input port `i`:
```
unnorm   = abs(U[:, i]) ** 2     # shape (m,)
p_hat    = unnorm / unnorm.sum() # shape (m,), sums to 1
```

For a batch, gather the relevant columns of `U` once per sample. Vectorise carefully so autograd can flow through.

---

## 5. Loss function

Squared L2 distance per sample, averaged over the batch:

```
loss = ((p_hat - p) ** 2).sum(dim=-1).mean()
```

`p_hat` and `p` are both shape `(batch, m)` and both normalised to sum to 1 along the last axis. The paper labels this MSE (one of several conventions); be explicit in the docstrings about which convention is used so the per-parameter learning rates remain meaningful.

---

## 6. Training schedule

### 6.1 What is updated, and when

Within a single ML stage, **all three trainable parameter blocks (`C_2`, `R`, `T_out`) are updated simultaneously** at every Adam step. The differences between them are:

- The **learning rates** are different (per-parameter, see §6.2).
- `c_0` is in the model but is **registered as a buffer**, not a parameter, so the optimiser ignores it. The gradient is not even computed.
- `T_in` is **not in the model** as a learnable quantity; it is set to identity. The gradient with respect to it is mathematically zero (see Pitfall 1 in §10).

There is **no alternating-block freeze schedule** within a single ML stage. This differs from Zheng et al., which alternates between updating `a` and `b` in the phase-current law to escape local minima. Fyrillas avoids local minima differently: by seeding the diagonal of `C_2` and the vector `c_0` from V-IFM first, then freezing `c_0` while learning the rest.

### 6.2 Per-parameter learning rates

| Parameter | Initial LR | Reasoning |
|---|---|---|
| `C_2` (full matrix) | `1e-5` | Each `C_2` entry maps a `V²` (`≤ V_max² ≈ 200 V²`) to a phase. Small steps in `C_2` produce large changes in `phi`, so the LR is two orders of magnitude smaller than for `R` and `T_out`. |
| `R` | `1e-3` | `R ∈ (0, 1)`. Standard "small parameter, modest LR" regime. |
| `T_out` | `1e-3` | Same regime as `R`. |

If you parameterise `R` and `T_out` through a sigmoid (recommended; see Pitfall 3 in §10), the LR applies to the logit, not to the post-sigmoid value. The values above are taken from the paper's Methods and assume the logit parameterisation.

### 6.3 Implementation: Adam parameter groups

```python
optimiser = torch.optim.Adam([
    {"params": [model.C_2_raw],  "lr": lr_C2},
    {"params": [model.R_logit],  "lr": lr_R},
    {"params": [model.T_logit],  "lr": lr_Tout},
])
```

Default `betas`, `eps`, `weight_decay = 0`. The paper does not report deviating from Adam defaults.

### 6.4 Within one ML stage

- **Epochs**: 500.
- **Batch size**: not specified in the paper. Reasonable choices: 32–256. Larger batches are stable; smaller batches add stochastic regularisation. Pick a default of 64 and expose it as an argument.
- **Shuffling**: yes, every epoch.
- **Test set**: monitor MSE and TVD every epoch; do not use the test set for optimiser decisions within a single stage. Keep the model state at the best test-MSE epoch.

### 6.5 Across (ML + φ-IFM) iterations (outer loop, out of scope here)

For completeness (this is not part of the present code, but informs the function signature):

- After each `(ML stage → φ-IFM stage)` round, multiply all learning rates by **`0.7`** before the next ML stage begins.
- The outer loop exits when the next ML stage's best test MSE exceeds the best test MSE of the previous ML stage (no further improvement).
- For the 12-mode chip in the paper, this loop converged after **one** full `(ML + φ-IFM)` pair plus one closing `ML` (Section 7.C: `V-IFM → ML → φ-IFM → ML → ITM`).

### 6.6 Summary diagram

```
                       ┌─────────────────────────────────────┐
                       │  Within one ML stage (this code):   │
                       │                                     │
   c_0  ───frozen──────┤                                     │
   T_in ───identity────┤   Adam, 500 epochs                  │
   C_2  ──learnable────┤   lr_C2  = 1e-5  (whole matrix)     │
   R    ──learnable────┤   lr_R   = 1e-3  (sigmoid logit)    │
   T_out──learnable────┤   lr_T   = 1e-3  (sigmoid logit)    │
                       │   no freeze alternation             │
                       └─────────────────────────────────────┘

   Across iterations (outer loop, NOT in this code):
       LR ← LR × 0.7 between consecutive ML stages.
```

---

## 7. Evaluation metrics

- **Train / Test MSE**: same form as the loss, computed on the respective sets each epoch.
- **TVD (total variation distance)** on the test set:
  ```
  TVD(p, p_hat) = 0.5 * abs(p - p_hat).sum(dim=-1)
  ```
  Report mean and standard deviation. TVD is bounded in `[0, 1]` and is the metric of choice for reporting comparable to the paper (≈ 2.9% on the paper's experimental data).
- Optional sanity: confirm `(p_hat.sum(dim=-1) - 1).abs().max() < 1e-6` to catch normalisation bugs early.

---

## 8. Suggested file structure

```
project/
├── CLAUDE.md                ← this file
├── README.md
├── src/
│   ├── __init__.py
│   ├── chip_mesh.py         ← topology: layers, shifter/BS indexing
│   ├── model.py             ← DigitalTwin nn.Module (parameters + forward)
│   ├── data.py              ← Dataset wrapper for (V, port, p) triples
│   ├── training.py          ← train loop with parameter groups
│   └── evaluate.py          ← TVD, plots, diagnostics
├── data/
│   └── README.md            ← expected data file format
├── tests/
│   ├── test_chip_mesh.py
│   ├── test_model.py
│   └── test_training.py
└── notebooks/
    └── exploration.ipynb    ← scratch space for runs and plots
```

---

## 9. Key API contracts

### `chip_mesh.py`
```python
from dataclasses import dataclass
from typing import Union

@dataclass
class PhaseShifterLayer:
    """One column of the mesh with PSs on disjoint waveguides.

    shifter_indices maps waveguide_index -> global PS index in [0, n_PS).
    """
    shifter_indices: dict[int, int]

@dataclass
class BeamsplitterLayer:
    """One column with BSs on disjoint waveguide pairs.

    bs_pairs maps (waveguide_top, waveguide_bottom) -> global BS index in [0, n_BS).
    """
    bs_pairs: dict[tuple[int, int], int]

Layer = Union[PhaseShifterLayer, BeamsplitterLayer]

class ChipMesh:
    """Ordered list of layers (input to output) plus global counts."""
    m: int                    # number of modes (waveguides)
    n_PS: int                 # number of phase shifters on chip
    n_BS: int                 # number of beamsplitters on chip
    layers: list[Layer]       # ordered, input -> output

    @classmethod
    def clements(cls, m: int) -> "ChipMesh":
        """Construct a Clements rectangular mesh on m modes."""
        ...
```

### `model.py`
```python
import torch
from torch import nn

class DigitalTwin(nn.Module):
    """Differentiable forward model of an imperfect linear-optical PIC.

    Parameters (in the optimiser sense):
        C_2_raw   : (n_PS, n_PS) free, mapped directly to C_2.
        R_logit   : (n_BS,) free, mapped to R via sigmoid (keeps R in (0, 1)).
        T_logit   : (m,)    free, mapped to T_out via sigmoid (keeps T_out in (0, 1]).

    Buffers (not optimised):
        c_0       : (n_PS,) frozen seed from V-IFM.
    """

    def __init__(self,
                 mesh: ChipMesh,
                 c_0_seed: torch.Tensor,
                 c2_diag_seed: torch.Tensor):
        ...

    @property
    def C_2(self) -> torch.Tensor: ...       # (n_PS, n_PS)
    @property
    def R(self) -> torch.Tensor: ...         # (n_BS,) in (0, 1)
    @property
    def T_out(self) -> torch.Tensor: ...     # (m,) in (0, 1]

    def forward(self,
                V: torch.Tensor,             # (batch, n_PS) real
                input_ports: torch.Tensor    # (batch,) long
                ) -> torch.Tensor:           # (batch, m) real, sums to 1 along dim=-1
        ...
```

### `training.py`
```python
def train(model: DigitalTwin,
          train_loader,
          test_loader,
          *,
          epochs: int = 500,
          lr_C2: float = 1e-5,
          lr_R: float = 1e-3,
          lr_Tout: float = 1e-3,
          device: str = "cpu") -> dict:
    """Run one ML stage. Returns a dict with per-epoch train/test MSE and TVD."""
    ...
```

### `data.py`
```python
class PICDataset(torch.utils.data.Dataset):
    """Holds (V, port, p) triples.

    V     : (N, n_PS) real, applied voltages.
    port  : (N,) int, input port index in [0, m).
    p     : (N, m) real, measured normalised output distribution (sums to 1).
    """
    ...
```

---

## 10. Pitfalls and design notes

1. **`T_in` cancels under normalisation.** Multiplying column `i` of `U` by `sqrt(T_in[i])` scales every entry of that column by the same constant, which cancels in the L2-normalisation of §4.4. Do not register `T_in` as a learnable parameter; its gradient is zero. The paper recovers `T_in` in a separate ITM stage that works on un-normalised intensity sums.

2. **`c_0` frozen during ML.** The fringe is 2π-periodic in each `phi_k`, so the gradient with respect to `c_0[k]` carries no information about which period the offset belongs to. The paper alternates ML (c_0 frozen) with φ-IFM (c_0 refined, others frozen). In this implementation, register `c_0` as a `buffer`, not a parameter.

3. **Reflectivity constraint.** `R` must lie in `[0, 1]`. Use the parameterisation `R = torch.sigmoid(R_logit)`. To initialise near 0.5, set `R_logit = 0`.

4. **Output-transmission constraint.** `T_out` should lie in `(0, 1]`. Same sigmoid parameterisation. To initialise at 1, set `T_logit` large positive (e.g. `+6`, giving `sigmoid ≈ 0.998`); or use `torch.clamp` after each step if you prefer a hard constraint.

5. **Complex-valued unitary construction.** Use `torch.complex64`. Phase shifters introduce `exp(1j * phi)`; beamsplitters introduce `1j * sqrt(1 - R)` off-diagonals. Loss values must be real, so take `abs(U[:, port]) ** 2` (modulus-squared) before reduction.

6. **Numerical stability.** When computing `sqrt(R)` and `sqrt(1 - R)`, if a raw clipped `R` ever touches `0` or `1`, the gradient of `sqrt` is infinite. The sigmoid parameterisation in note 3 makes this a non-issue. If you instead use clipping, add `+ 1e-8` inside the sqrt.

7. **Loss reduction convention.** Choose between `mean` over batch + `sum` over modes, or `mean` over both. The paper's reported learning rates were tuned for "mean over batch, sum over modes". Stick with this and document the choice in the docstring.

8. **Per-parameter learning rates.** Implement via Adam parameter groups:
   ```python
   optimiser = torch.optim.Adam([
       {"params": [model.C_2_raw], "lr": lr_C2},
       {"params": [model.R_logit], "lr": lr_R},
       {"params": [model.T_logit], "lr": lr_Tout},
   ])
   ```

9. **Initialisation of `C_2`.** Diagonal: V-IFM seed (provided externally). Off-diagonal: zero. Track the **raw** parameter `C_2_raw` directly (no transformation needed); the off-diagonal entries can take either sign and have no physical positivity constraint.

10. **Batching the matrix product.** Each sample has its own `phi` vector. Building `U` per sample is the simplest implementation: loop over the batch in Python (slow but correct) for an initial version; refactor to batched matrix multiplication once correctness is established. For `m = 12` the per-sample cost is negligible; this is more about code clarity than performance.

11. **Test the forward model first.** With `c_0 = 0`, `C_2 = 0`, `R = 0.5`, `T_out = 1`, the forward pass on a Clements mesh should produce a unitary `U_0` (verify `U_0.conj().T @ U_0 ≈ I` up to float precision). This catches mesh-construction bugs before any training is attempted.

12. **Test the gradient flow.** Run a single training step and check that the gradients of `C_2_raw`, `R_logit`, `T_logit` are non-zero, while `c_0` has no gradient (or is not in the parameter list at all).

---

## 11. Suggested implementation order

1. `chip_mesh.py` — dataclasses, `ChipMesh.clements(m)`. Unit test: shape, layer count, no overlapping waveguide indices within a column.
2. `model.py` — `DigitalTwin` with the forward pass. Unit test: lossless limit produces unitary `U_0`.
3. `data.py` — `PICDataset`. Unit test: shape contracts.
4. `training.py` — one `train()` function. Unit test: loss decreases over 10 epochs on a synthetic dataset built from a known ground-truth model.
5. `evaluate.py` — TVD computation, plotting helpers (histograms of recovered parameters, train/test loss curves).
6. End-to-end check on synthetic data generated from a ground-truth `DigitalTwin` instance with known `C_2`, `R`, `T_out`. Train a fresh `DigitalTwin` against this data; recovered parameters should match the ground truth within a tolerance set by the dataset size and noise level (compare against Fig. 3 of the paper).

---

## 12. Useful references inside the paper

- **Equation 2 (paper main text)**: `phi = C_2 V^⊙2 + c_0`. The core phase-voltage law.
- **Section 3.C "Machine learning"**: the gradient-descent step, parameter list, MSE cost.
- **Section 3.E "ITM"**: how `T_in` is recovered post-hoc (out of scope for this code).
- **Section 7.A "Methods"**: per-parameter learning rates (`C_2`: 1e-5; `R`, `T_out`: 1e-3), schedule factor 0.7, 500 epochs per ML stage, `V_max = 14 V`.
- **Section 7.C "Experimental validation"**: dataset sizes (16,500 train / 4,125 test on a 12-mode chip), train/test TVD of 2.2% / 2.9%.
- **Supplement Section D**: definitions of MSE, TVD, fidelity metrics used in the paper.
- **Supplement Section C.3**: protocol-generation algorithm for V-IFM and φ-IFM (out of scope; affects how the seeds reaching this code are obtained).
