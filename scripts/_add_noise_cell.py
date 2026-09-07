"""Inserter / updater for the noise-sweep analysis cell in
exploration.ipynb. Idempotent: replaces an existing cell tagged with
ANCHOR in-place, or appends after the per-epoch trajectory cell.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/exploration.ipynb")
ANCHOR = "# >>> noise-sweep analysis cell <<<"

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Plot 5  Noise sweep -- TVD floor & per-noise trajectory\n",
        "\n",
        "Two views of the `sweep_noise.json` + `.log` produced by ",
        "`scripts/sweep_noise.py`:\n",
        "\n",
        "* **5a (floor vs noise):** the post-ML TVD reached at the final ",
        "cycle of each sigma run, plotted against sigma on a log-log axis. ",
        "A `y = sigma` reference line lets you see whether the TVD floor ",
        "scales linearly with the injected Gaussian noise (the expectation ",
        "for normalised-intensity output). Maps qualitatively to paper ",
        "Fig 3(d).\n",
        "\n",
        "* **5b (trajectory per sigma):** one subplot per noise level, ",
        "same per-epoch sampling as Plot 4 (every `epochs // 5` epochs). ",
        "Cycle boundaries drawn as thin vertical guides; the phi-IFM jump ",
        "is shown as a dashed spike + 'x' marker. Lets you confirm whether ",
        "each TVD trajectory has actually **settled** at the floor or is ",
        "still descending when the run exits.\n",
    ],
}

code_source = [
    f"{ANCHOR}\n",
    "# Loads outputs/sweep_noise.json (+ sidecar .log for per-epoch\n",
    "# trajectory). Robust to a single-sigma JSON: 5a shows one point\n",
    "# and 5b shows one subplot.\n",
    "import re\n",
    "from collections import defaultdict\n",
    "import numpy as np\n",
    "\n",
    "SWEEP_NOISE_PATH = _ROOT / 'outputs' / 'sweep_noise.json'\n",
    "if not SWEEP_NOISE_PATH.exists():\n",
    "    raise FileNotFoundError(\n",
    "        f'{SWEEP_NOISE_PATH} not found. Run:\\n'\n",
    "        '    uv run python scripts/sweep_noise.py --num-threads 4'\n",
    "    )\n",
    "\n",
    "ns = json.loads(SWEEP_NOISE_PATH.read_text(encoding='utf-8'))\n",
    "ns_cfg = ns['config']\n",
    "# results dict: key = noise sigma as string ('0', '1.000e-04', ...).\n",
    "# Sort numerically by the per-entry 'noise_std' field.\n",
    "ns_results_sorted = sorted(\n",
    "    [\n",
    "        (k, v) for k, v in ns['results'].items()\n",
    "        if v.get('status') == 'ok'\n",
    "    ],\n",
    "    key=lambda kv: kv[1]['noise_std'],\n",
    ")\n",
    "\n",
    "print(f\"loaded {SWEEP_NOISE_PATH.name}  \"\n",
    "      f\"({len(ns_results_sorted)} completed sigma points, \"\n",
    "      f\"total wall-clock {ns['total_elapsed']:.1f} s)\")\n",
    "print(f\"  m = {ns_cfg['m']},  epochs/stage = {ns_cfg['epochs']},  \"\n",
    "      f\"max_cycles = {ns_cfg['max_cycles']},  \"\n",
    "      f\"lr_schedule = {ns_cfg['lr_schedule']}\")\n",
    "\n",
    "# ---- Plot 5a: TVD floor vs sigma ---------------------------------------\n",
    "sigmas = np.array([entry['noise_std'] for _, entry in ns_results_sorted])\n",
    "final_tvds = np.array([\n",
    "    (entry['history'][-1]['post_ml_tvd'] * 100) if entry['history'] else np.nan\n",
    "    for _, entry in ns_results_sorted\n",
    "])\n",
    "cycles_used = np.array([\n",
    "    len(entry['history']) for _, entry in ns_results_sorted\n",
    "])\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(7.0, 5.0))\n",
    "# Replace sigma=0 with a small placeholder for the log axis.\n",
    "sigmas_plot = np.where(sigmas == 0, sigmas[sigmas > 0].min() / 10\n",
    "                       if (sigmas > 0).any() else 1e-6,\n",
    "                       sigmas)\n",
    "ax.loglog(sigmas_plot, final_tvds, 'o-', color='C0', markersize=8,\n",
    "          label='post-ML TVD at final cycle')\n",
    "# y = sigma reference (in percent: y = sigma * 100)\n",
    "if (sigmas > 0).any():\n",
    "    s_ref = np.array([sigmas[sigmas > 0].min(), sigmas.max()])\n",
    "    ax.loglog(s_ref, s_ref * 100, 'k--', linewidth=0.8, alpha=0.6,\n",
    "              label=r'$y = \\sigma$ (reference)')\n",
    "ax.axhline(ns_cfg['threshold'] * 100, color='gray', linestyle=':',\n",
    "           linewidth=0.8, label=f\"threshold = {ns_cfg['threshold']*100:g}%\")\n",
    "for s, t, c in zip(sigmas_plot, final_tvds, cycles_used):\n",
    "    ax.annotate(f'{c} cyc', (s, t), textcoords='offset points',\n",
    "                xytext=(6, -8), fontsize=8, color='C0')\n",
    "ax.set_xlabel(r'Gaussian noise $\\sigma$  (0 plotted at $\\sigma_{\\min}/10$)')\n",
    "ax.set_ylabel('final post-ML test TVD  [%]')\n",
    "ax.set_title(f'TVD floor vs measurement noise  (m={ns_cfg[\"m\"]}, '\n",
    "             f'{ns_cfg[\"lr_schedule\"]} LR)')\n",
    "ax.grid(True, which='both', alpha=0.3)\n",
    "ax.legend(loc='best', fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# ---- Plot 5b: per-sigma per-epoch trajectory ---------------------------\n",
    "SWEEP_NOISE_LOG = SWEEP_NOISE_PATH.with_suffix('.log')\n",
    "if not SWEEP_NOISE_LOG.exists():\n",
    "    print(f'  (skipping 5b: {SWEEP_NOISE_LOG} not found)')\n",
    "else:\n",
    "    _HEADER_NOISE_RE = re.compile(\n",
    "        r'\\[\\d+/\\d+\\]\\s+sigma=([\\d.eE+\\-]+)'\n",
    "    )\n",
    "    _CYCLE_RE = re.compile(r'---\\s+Cycle\\s+(\\d+)')\n",
    "    _EPOCH_RE = re.compile(\n",
    "        r'epoch\\s+(\\d+)/\\d+\\s+.*test_tvd=([\\d.eE+\\-]+)%'\n",
    "    )\n",
    "    _POSTML_RE = re.compile(\n",
    "        r'post-ML\\s*:\\s*TVD\\s*=\\s*([\\d.eE+\\-]+)%'\n",
    "    )\n",
    "    _POSTPHI_RE = re.compile(\n",
    "        r'post-phi-IFM\\s*:\\s*TVD\\s*=\\s*([\\d.eE+\\-]+)%'\n",
    "    )\n",
    "\n",
    "    # traj[sigma_float][cycle] -> {'samples', 'post_ml', 'post_phi_ifm'}\n",
    "    traj: dict = defaultdict(dict)\n",
    "    cur_sigma = cur_cyc = None\n",
    "    for line in SWEEP_NOISE_LOG.read_text(encoding='utf-8').splitlines():\n",
    "        h = _HEADER_NOISE_RE.search(line)\n",
    "        if h:\n",
    "            cur_sigma = float(h.group(1))\n",
    "            cur_cyc = None\n",
    "            continue\n",
    "        c = _CYCLE_RE.search(line)\n",
    "        if c and cur_sigma is not None:\n",
    "            cur_cyc = int(c.group(1))\n",
    "            traj[cur_sigma].setdefault(cur_cyc, {\n",
    "                'samples': [], 'post_ml': None, 'post_phi_ifm': None,\n",
    "            })\n",
    "            continue\n",
    "        if cur_cyc is None:\n",
    "            continue\n",
    "        em = _EPOCH_RE.search(line)\n",
    "        if em:\n",
    "            traj[cur_sigma][cur_cyc]['samples'].append(\n",
    "                (int(em.group(1)), float(em.group(2)))\n",
    "            )\n",
    "            continue\n",
    "        pm = _POSTML_RE.search(line)\n",
    "        if pm:\n",
    "            traj[cur_sigma][cur_cyc]['post_ml'] = float(pm.group(1))\n",
    "            continue\n",
    "        pp = _POSTPHI_RE.search(line)\n",
    "        if pp:\n",
    "            traj[cur_sigma][cur_cyc]['post_phi_ifm'] = float(pp.group(1))\n",
    "            continue\n",
    "\n",
    "    sigma_keys = sorted(traj.keys())\n",
    "    n_s = len(sigma_keys)\n",
    "    if n_s == 0:\n",
    "        print('  (no trajectory data parsed from log)')\n",
    "    else:\n",
    "        fig, axes = plt.subplots(\n",
    "            1, n_s, figsize=(4.5 * n_s, 4.5), sharey=True,\n",
    "        )\n",
    "        if n_s == 1:\n",
    "            axes = [axes]\n",
    "        e = ns_cfg['epochs']\n",
    "        for ax, sigma in zip(axes, sigma_keys):\n",
    "            cell = traj[sigma]\n",
    "            max_cycle = max(cell.keys())\n",
    "            color = 'C0'\n",
    "            for cyc in sorted(cell):\n",
    "                d = cell[cyc]\n",
    "                if not d['samples']:\n",
    "                    continue\n",
    "                xs = [(cyc - 1) * e + ep for ep, _ in d['samples']]\n",
    "                ys = [tvd for _, tvd in d['samples']]\n",
    "                ax.semilogy(xs, ys, '.-', color=color, markersize=4,\n",
    "                            linewidth=1.2)\n",
    "                # phi-IFM spike + connector to next cycle\n",
    "                post_ml = d['post_ml'] if d['post_ml'] is not None else ys[-1]\n",
    "                post_phi = d['post_phi_ifm']\n",
    "                nxt = cell.get(cyc + 1)\n",
    "                if post_phi is not None and nxt and nxt['samples']:\n",
    "                    x_b = cyc * e\n",
    "                    ax.plot([x_b, x_b], [post_ml, post_phi], ':',\n",
    "                            color=color, linewidth=0.9, alpha=0.55, zorder=2)\n",
    "                    ax.plot(x_b, post_phi, marker='x', color=color,\n",
    "                            markersize=8, markeredgewidth=1.5, zorder=3)\n",
    "                    next_ep, next_tvd = nxt['samples'][0]\n",
    "                    next_x = cyc * e + next_ep\n",
    "                    ax.plot([x_b, next_x], [post_phi, next_tvd], '-',\n",
    "                            color=color, linewidth=1.2, alpha=0.9, zorder=2)\n",
    "            for c in range(1, max_cycle):\n",
    "                ax.axvline(x=c * e, color='lightgray', linestyle='-',\n",
    "                           linewidth=0.7, zorder=0)\n",
    "            # noise-floor reference (sigma in percent)\n",
    "            if sigma > 0:\n",
    "                ax.axhline(sigma * 100, color='red', linestyle=':',\n",
    "                           linewidth=0.8, alpha=0.6,\n",
    "                           label=fr'$\\sigma$={sigma:.0e} ({sigma*100:g}%)')\n",
    "            ax.axhline(ns_cfg['threshold'] * 100, color='gray',\n",
    "                       linestyle=':', linewidth=0.8,\n",
    "                       label=f\"thr={ns_cfg['threshold']*100:g}%\")\n",
    "            ax.set_title(fr'$\\sigma$ = {sigma:.0e}')\n",
    "            ax.set_xlabel('cumulative epoch')\n",
    "            ax.grid(True, which='both', alpha=0.3)\n",
    "            ax.legend(fontsize=8, loc='upper right')\n",
    "        axes[0].set_ylabel('test TVD  [%]')\n",
    "        fig.suptitle(\n",
    "            f'Per-epoch TVD trajectory by noise level  '\n",
    "            f'({SWEEP_NOISE_LOG.name})',\n",
    "            y=1.02,\n",
    "        )\n",
    "        plt.tight_layout()\n",
    "        plt.show()\n",
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

    # Idempotent update: replace existing cell if tagged with ANCHOR.
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code" and any(
            ANCHOR in line for line in c.get("source", [])
        ):
            cells[i] = code_cell
            if i > 0 and cells[i - 1].get("cell_type") == "markdown" and any(
                "Noise sweep" in line for line in cells[i - 1].get("source", [])
            ):
                cells[i - 1] = markdown_cell
            NB_PATH.write_text(
                json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"updated noise-sweep cell at index {i}.")
            return

    # Fresh insert after the per-epoch trajectory cell (Plot 4 code cell).
    insert_at = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code" and any(
            "per-epoch TVD trajectory cell" in line
            for line in c.get("source", [])
        ):
            insert_at = i + 1
            break
    if insert_at is None:
        raise SystemExit("could not find the Plot 4 trajectory cell")

    cells.insert(insert_at, markdown_cell)
    cells.insert(insert_at + 1, code_cell)
    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"inserted markdown + code cells at indices "
        f"{insert_at} and {insert_at + 1}."
    )


if __name__ == "__main__":
    main()
