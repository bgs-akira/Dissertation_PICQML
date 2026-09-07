# Reference papers

Upstream sources this project depends on. **PDFs are not committed to git**
(copyright + git is bad at binary blobs); see `.gitignore`. Download the
papers below into this directory if you want them locally.

## Primary

- **Fyrillas, A.; Faure, O.; Maring, N.; Senellart, J.; Belabas, N.**
  *Scalable machine learning-assisted clear-box characterization for
  optimally controlled photonic circuits.*
  **Optica 11, 427 (2024).** DOI:
  [10.1364/OPTICA.512148](https://doi.org/10.1364/OPTICA.512148).

  Local filename convention: `fyrillas_2024.pdf`.

  This is the protocol this project implements. `CLAUDE.md` (project root)
  links each algorithmic choice back to a section number in this paper.
  Key sections cross-referenced from `CLAUDE.md` §12:
    - §3.C  Machine learning (gradient-descent step, parameter list, MSE cost)
    - §3.E  ITM (T_in recovery, out of scope here)
    - §3.F  Simulated benchmark (noise model, simulated-vs-experimental TVD)
    - §5    Experimental TVD floor (~2.9 % on the 12-mode chip)
    - §7.A  Methods (LRs, schedule factor, V_max)
    - §7.C  Experimental validation (dataset sizes, train/test TVD)
    - Supplement C.2  Fast vs precise fringe-fit
    - Supplement C.3  V-IFM / phi-IFM protocol generation
    - Supplement D    MSE, TVD, fidelity definitions
    - Supplement H    Iterative phase-voltage solver

## Secondary / cited

Add additional references here as the project grows. Keep the same
formatting: full citation + DOI + local-filename convention.
