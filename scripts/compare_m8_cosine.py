"""Compare m=8 iterative-loop TVD trajectories: constant LR vs cosine annealing.

Reads:
  outputs/sweep_m_epochs.json   (baseline, constant LR)
  outputs/iterative_m8_cosine.json   (cosine annealing fix)
Prints a side-by-side cycle-by-cycle table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

baseline_path = _ROOT / "outputs" / "sweep_m_epochs.json"
cosine_path = _ROOT / "outputs" / "iterative_m8_cosine.json"

if not cosine_path.exists():
    print(f"missing: {cosine_path}", file=sys.stderr)
    sys.exit(1)

baseline = json.loads(baseline_path.read_text())
cosine = json.loads(cosine_path.read_text())

baseline_history = baseline["results"].get("8", {}).get("200", {}).get("history", [])
cosine_history = cosine.get("history", [])

print(
    f"{'cyc':>4}  {'baseline post-ML':>18}  {'cosine post-ML':>18}  "
    f"{'delta':>8}  {'cosine post-phi-IFM':>22}"
)
print("-" * 80)
for i in range(max(len(baseline_history), len(cosine_history))):
    b = baseline_history[i] if i < len(baseline_history) else None
    c = cosine_history[i] if i < len(cosine_history) else None
    b_tvd = b["post_ml_tvd"] * 100 if b else None
    c_tvd = c["post_ml_tvd"] * 100 if c else None
    c_pi_tvd = c["post_phi_ifm_tvd"] * 100 if c else None
    cyc = (b or c)["cycle"]
    delta = (c_tvd - b_tvd) if (b_tvd is not None and c_tvd is not None) else None
    b_str = f"{b_tvd:>16.4f}%" if b_tvd is not None else f"{'—':>17s}"
    c_str = f"{c_tvd:>16.4f}%" if c_tvd is not None else f"{'—':>17s}"
    d_str = f"{delta:>+7.4f}%" if delta is not None else f"{'—':>8s}"
    pi_str = f"{c_pi_tvd:>20.4f}%" if c_pi_tvd is not None else f"{'—':>21s}"
    print(f"{cyc:>4}  {b_str}  {c_str}  {d_str}  {pi_str}")

print()
print("Summary:")
print(f"  Baseline final TVD ({len(baseline_history)} cycles): "
      f"{baseline_history[-1]['post_ml_tvd']*100:.4f}%")
print(f"  Cosine final TVD ({len(cosine_history)} cycles): "
      f"{cosine_history[-1]['post_ml_tvd']*100:.4f}%")
print(f"  Cosine converged: {cosine.get('summary', {}).get('converged')}")
print(f"  Cosine exit reason: {cosine.get('summary', {}).get('exit_reason')}")
