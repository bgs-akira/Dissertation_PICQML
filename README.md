# Dissertation_PICQML

Implementation of the **ML stage** of the photonic-chip clear-box characterisation
protocol from Fyrillas et al., *Optica* **11**, 427 (2024).
DOI: 10.1364/OPTICA.512148.

See [CLAUDE.md](CLAUDE.md) for the full project specification, parameter set,
forward-model derivation, training schedule, and design notes.

## Layout

```
src/         differentiable forward model + training code
data/        input datasets (V, port, p triples) and V-IFM seeds
tests/       unit tests
notebooks/   exploratory notebooks
```

## Quick start

```bash
pip install -r requirements.txt   # (requirements.txt not yet committed)
pytest tests/
```

## Scope

- **In scope**: differentiable digital twin, Adam training loop with
  per-parameter learning rates, recovery of `C_2`, `R`, `T_out`.
- **Out of scope**: V-IFM data acquisition, φ-IFM phase fitting, ITM,
  hardware control. See CLAUDE.md §1 for details.
