"""Inserter / updater for the 'per-epoch TVD trajectory' cell in
exploration.ipynb.

Idempotent: replaces an existing cell tagged with ANCHOR in-place, or inserts
a fresh markdown + code pair after the wall-clock heatmap cell. Run after
edits to refresh the notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/exploration.ipynb")
ANCHOR = "# >>> per-epoch TVD trajectory cell <<<"

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Plot 4  Per-epoch TVD trajectory across the sweep grid\n",
        "\n",
        "One subplot per `epochs/stage`, all `m` overlaid. Each cycle's ML ",
        "stage is drawn as a connected polyline of the per-epoch samples ",
        "logged every `epochs // 5` steps. Cycle `k` occupies the interval ",
        "`((k-1)*epochs, k*epochs]` on the x axis (the log prints `epoch X/N` ",
        "where X = epochs completed so far in the stage, so cycle 1's last ",
        "sample sits at x = epochs, cycle 2's first sample at x = epochs+40, ",
        "etc.). The φ-IFM jump is drawn explicitly as a thin dashed vertical ",
        "from `post-ML TVD` to `post-φ-IFM TVD` at the boundary (the ",
        "**discontinuity** — the ML curve does not connect to the next ",
        "cycle directly), with an `x` marker at the new post-φ-IFM value. ",
        "From that `x` marker a solid segment reconnects to the first ML ",
        "sample of the next cycle, showing how fast the next ML stage ",
        "descends from the c_0-perturbed starting point.\n",
    ],
}

code_source = [
    f"{ANCHOR}\n",
    "# Per-epoch test-TVD trajectory, sampled from the .log file (every\n",
    "# epochs//5 steps -- that's what the sweep prints; the JSON only stores\n",
    "# per-cycle endpoints, see Plot 2 above).\n",
    "import re\n",
    "from collections import defaultdict\n",
    "\n",
    "ME_LOG_PATH = ME_PATH.with_suffix('.log')\n",
    "if not ME_LOG_PATH.exists():\n",
    "    raise FileNotFoundError(\n",
    "        f'{ME_LOG_PATH} not found.  Set ME_PATH (cell above) to the JSON\\n'\n",
    "        f'whose sidecar log you want to plot.'\n",
    "    )\n",
    "\n",
    "_HEADER_RE  = re.compile(r'\\[\\d+/\\d+\\]\\s+m=(\\d+)\\s+epochs=(\\d+)')\n",
    "_CYCLE_RE   = re.compile(r'---\\s+Cycle\\s+(\\d+)')\n",
    "_EPOCH_RE   = re.compile(r'epoch\\s+(\\d+)/\\d+\\s+.*test_tvd=([\\d.eE+\\-]+)%')\n",
    "_POSTML_RE  = re.compile(r'post-ML\\s*:\\s*TVD\\s*=\\s*([\\d.eE+\\-]+)%')\n",
    "_POSTPHI_RE = re.compile(r'post-phi-IFM\\s*:\\s*TVD\\s*=\\s*([\\d.eE+\\-]+)%')\n",
    "\n",
    "# traj[(m, e)][cycle] -> dict with 'samples', 'post_ml', 'post_phi_ifm'\n",
    "traj: dict = defaultdict(dict)\n",
    "cur_m = cur_e = cur_cyc = None\n",
    "for line in ME_LOG_PATH.read_text(encoding='utf-8').splitlines():\n",
    "    h = _HEADER_RE.search(line)\n",
    "    if h:\n",
    "        cur_m, cur_e = int(h.group(1)), int(h.group(2))\n",
    "        cur_cyc = None\n",
    "        continue\n",
    "    c = _CYCLE_RE.search(line)\n",
    "    if c and cur_m is not None:\n",
    "        cur_cyc = int(c.group(1))\n",
    "        traj[(cur_m, cur_e)].setdefault(cur_cyc, {\n",
    "            'samples': [], 'post_ml': None, 'post_phi_ifm': None,\n",
    "        })\n",
    "        continue\n",
    "    if cur_cyc is None:\n",
    "        continue\n",
    "    em = _EPOCH_RE.search(line)\n",
    "    if em:\n",
    "        traj[(cur_m, cur_e)][cur_cyc]['samples'].append(\n",
    "            (int(em.group(1)), float(em.group(2)))\n",
    "        )\n",
    "        continue\n",
    "    pm = _POSTML_RE.search(line)\n",
    "    if pm:\n",
    "        traj[(cur_m, cur_e)][cur_cyc]['post_ml'] = float(pm.group(1))\n",
    "        continue\n",
    "    pp = _POSTPHI_RE.search(line)\n",
    "    if pp:\n",
    "        traj[(cur_m, cur_e)][cur_cyc]['post_phi_ifm'] = float(pp.group(1))\n",
    "        continue\n",
    "\n",
    "n_e = len(ME_EP_VALUES)\n",
    "fig, axes = plt.subplots(1, n_e, figsize=(4.5 * n_e, 4.5), sharey=True)\n",
    "if n_e == 1:\n",
    "    axes = [axes]\n",
    "\n",
    "for ax, e in zip(axes, ME_EP_VALUES):\n",
    "    max_cycle = 0\n",
    "    for k, m in enumerate(ME_M_VALUES):\n",
    "        cell = traj.get((m, e), {})\n",
    "        if not cell:\n",
    "            continue\n",
    "        color = f'C{k}'\n",
    "        labelled = False\n",
    "        sorted_cycs = sorted(cell)\n",
    "        for cyc in sorted_cycs:\n",
    "            data = cell[cyc]\n",
    "            samples = data['samples']\n",
    "            if not samples:\n",
    "                continue\n",
    "            xs = [(cyc - 1) * e + ep for ep, _ in samples]\n",
    "            ys = [tvd for _, tvd in samples]\n",
    "            ax.semilogy(\n",
    "                xs, ys, '.-', color=color, markersize=4, linewidth=1.2,\n",
    "                label=(None if labelled else f'm = {m}'),\n",
    "            )\n",
    "            labelled = True\n",
    "            # phi-IFM jump at the cycle boundary, only if a next cycle exists\n",
    "            # in this run (the last cycle has no following phi-IFM-fed stage).\n",
    "            post_ml = data['post_ml'] if data['post_ml'] is not None else ys[-1]\n",
    "            post_phi = data['post_phi_ifm']\n",
    "            next_cyc_has_data = (\n",
    "                (cyc + 1) in cell and cell[cyc + 1]['samples']\n",
    "            )\n",
    "            if post_phi is not None and next_cyc_has_data:\n",
    "                x_b = cyc * e\n",
    "                # Vertical dashed 'spike' from end-of-ML up/down to\n",
    "                # post-phi-IFM (the discontinuity).\n",
    "                ax.plot(\n",
    "                    [x_b, x_b], [post_ml, post_phi],\n",
    "                    ':', color=color, linewidth=0.9, alpha=0.55, zorder=2,\n",
    "                )\n",
    "                # 'x' marker at the post-phi-IFM TVD.\n",
    "                ax.plot(\n",
    "                    x_b, post_phi, marker='x', color=color,\n",
    "                    markersize=8, markeredgewidth=1.5, zorder=3,\n",
    "                )\n",
    "                # Connector from the x marker to the first sample of the\n",
    "                # next cycle: the ML stage of cycle k+1 starts from the\n",
    "                # phi-IFM-perturbed state and descends from there.\n",
    "                next_ep, next_tvd = cell[cyc + 1]['samples'][0]\n",
    "                next_x = cyc * e + next_ep\n",
    "                ax.plot(\n",
    "                    [x_b, next_x], [post_phi, next_tvd],\n",
    "                    '-', color=color, linewidth=1.2, alpha=0.9, zorder=2,\n",
    "                )\n",
    "            max_cycle = max(max_cycle, cyc)\n",
    "    # Cycle-boundary guides (one per epoch length per subplot).\n",
    "    for c in range(1, max_cycle):\n",
    "        ax.axvline(\n",
    "            x=c * e, color='lightgray', linestyle='-',\n",
    "            linewidth=0.7, zorder=0,\n",
    "        )\n",
    "    ax.axhline(\n",
    "        ME_THRESHOLD * 100, color='gray', linestyle=':',\n",
    "        linewidth=0.8, label=f'{ME_THRESHOLD*100:g}%',\n",
    "    )\n",
    "    ax.set_title(f'epochs/stage = {e}')\n",
    "    ax.set_xlabel('cumulative epoch')\n",
    "    ax.grid(True, which='both', alpha=0.3)\n",
    "    ax.legend(fontsize=8, loc='upper right')\n",
    "\n",
    "axes[0].set_ylabel('test TVD  [%]')\n",
    "fig.suptitle(\n",
    "    f'Per-epoch test TVD trajectory  '\n",
    "    f'({ME_LOG_PATH.name})  '\n",
    "    f'\\u2014 dashed spike + \\u00d7 = post-\\u03c6-IFM jump',\n",
    "    y=1.02,\n",
    ")\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": code_source,
}


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Idempotent update: if the cell already exists, replace it in place
    # (and the markdown cell immediately above it).
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code" and any(
            ANCHOR in line for line in c.get("source", [])
        ):
            cells[i] = code_cell
            if i > 0 and cells[i - 1].get("cell_type") == "markdown" and any(
                "Per-epoch TVD trajectory" in line
                for line in cells[i - 1].get("source", [])
            ):
                cells[i - 1] = markdown_cell
            NB_PATH.write_text(
                json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"updated trajectory cell at index {i} (notebook unchanged size).")
            return

    # Fresh insert after the wall-clock heatmap cell.
    insert_at = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code" and any(
            "time_matrix" in line for line in c.get("source", [])
        ):
            insert_at = i + 1
            break
    if insert_at is None:
        raise SystemExit("could not find the wall-clock heatmap cell")

    cells.insert(insert_at, markdown_cell)
    cells.insert(insert_at + 1, code_cell)
    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"inserted markdown + code cells at indices "
        f"{insert_at} and {insert_at + 1} (notebook now {len(cells)} cells)."
    )


if __name__ == "__main__":
    main()
