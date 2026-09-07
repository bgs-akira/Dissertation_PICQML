"""Matplotlib visualisation of the IFM characterization protocol.

Mirrors Fig. 4, 6, 7, 8 of the supplement: waveguides as horizontal
lines, MZIs as coloured squares, external PSs as smaller squares on the
upper arm. Each :class:`~src.routing.CharacterizationStep` colours MZIs
by their state and overlays the light path between input and output
ports.

Public entry points:

  - :func:`draw_mesh` -- bare PIC drawing.
  - :func:`draw_step` -- one CharacterizationStep on a fresh axis.
  - :func:`draw_protocol` -- grid of subplots, one per step, tracking the
    running characterized set so each frame correctly colours green MZIs
    that have been done in prior frames.

Colour legend (Fig. 6):

  - light blue   : uncharacterized MZI / external PS.
  - green        : characterized MZI (router pool for routed steps).
  - red outline  : the PS being characterized in this step.
  - orange       : head / tail MZI of a meta-MZI (balanced).
  - red line     : the light path between input and output ports.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

from src.chip_mesh import BeamsplitterLayer, ChipMesh
from src.pic_graph import NodeKind, PICGraph, build_from_mesh
from src.routing import (
    CharacterizationStep,
    StepKind,
    generate_ifm_protocol,
)


# ---------------------------------------------------------------------------
# Colours and sizes
# ---------------------------------------------------------------------------


COLOR_UNCHARAC = "#a8c5e6"
COLOR_CHARAC = "#7fbf7f"
COLOR_TARGET_EDGE = "#d62728"
COLOR_META = "#ffb84d"
COLOR_LIGHT = "#d62728"
COLOR_WG = "#888888"
COLOR_EXT_PS = "#c8a8e6"
COLOR_EXT_PS_TARGET = "#d62728"

# Geometry: every MZI sits in a "column" (one column per MZI row in the
# Clements layout). The MZI square has half-extent MZI_HALF in both x
# and y. External PSs are drawn as smaller squares immediately to the
# right of the MZI, on the upper arm.
MZI_HALF = 0.35
EXT_PS_HALF = 0.12
EXT_PS_DX = 0.6   # horizontal offset of EXT_PS from MZI centre


# ---------------------------------------------------------------------------
# Layout: where to draw each node
# ---------------------------------------------------------------------------


def _mzi_columns(mesh: ChipMesh, graph: PICGraph) -> dict[int, int]:
    """Map MZI node id -> column index (left to right). One column per
    MZI row in the mesh layout (each mesh row contributes 4 atomic layers
    but a single drawing column)."""
    col_of: dict[int, int] = {}
    col = 0
    idx = 0
    layers = mesh.layers
    while idx < len(layers):
        L_bsa = layers[idx]
        assert isinstance(L_bsa, BeamsplitterLayer)
        # Find graph MZIs whose bs_a_index appears in this BS layer.
        bs_a_set = set(L_bsa.bs_pairs.values())
        for n in graph.nodes:
            if n.kind == NodeKind.MZI and n.bs_a_index in bs_a_set:
                col_of[n.id] = col
        col += 1
        idx += 4
    return col_of


def _node_positions(
    mesh: ChipMesh, graph: PICGraph,
) -> tuple[dict[int, tuple[float, float]], int]:
    """Return (positions, n_cols).

    For MZI nodes: centre square at (col, wg_top + 0.5).
    For EXT_PS nodes: centre square at (col + EXT_PS_DX, wg_top).
    For INPUT nodes: (-1, port).
    For OUTPUT nodes: (n_cols, port).
    """
    col_of = _mzi_columns(mesh, graph)
    n_cols = max(col_of.values()) + 1 if col_of else 1

    pos: dict[int, tuple[float, float]] = {}
    for n in graph.nodes:
        if n.kind == NodeKind.MZI:
            assert n.wg_top is not None
            pos[n.id] = (col_of[n.id], n.wg_top + 0.5)
        elif n.kind == NodeKind.EXT_PS:
            host_mzi = graph.nodes[n.parents[0]]
            assert host_mzi.wg_top is not None
            col = col_of[host_mzi.id]
            pos[n.id] = (col + EXT_PS_DX, host_mzi.wg_top)
        elif n.kind == NodeKind.INPUT:
            assert n.port is not None
            pos[n.id] = (-1.0, float(n.port))
        elif n.kind == NodeKind.OUTPUT:
            assert n.port is not None
            pos[n.id] = (float(n_cols), float(n.port))
    return pos, n_cols


# ---------------------------------------------------------------------------
# Base mesh drawing
# ---------------------------------------------------------------------------


def draw_mesh(
    mesh: ChipMesh,
    *,
    ax: Optional[Axes] = None,
    graph: Optional[PICGraph] = None,
) -> Axes:
    """Draw the bare PIC: waveguides and MZI/EXT_PS squares.

    No step-specific colouring -- every MZI and external PS is drawn in
    the uncharacterized colour. Used as the substrate for
    :func:`draw_step`.
    """
    if graph is None:
        graph = build_from_mesh(mesh)
    if ax is None:
        _, ax = plt.subplots(figsize=(2 + 0.6 * len(mesh.layers) / 4, 0.7 * mesh.m + 1))

    positions, n_cols = _node_positions(mesh, graph)

    # Waveguides (horizontal lines).
    for wg in range(mesh.m):
        ax.plot(
            [-1, n_cols], [wg, wg],
            color=COLOR_WG, linewidth=1.5, zorder=1,
        )

    # MZIs.
    for n in graph.nodes:
        if n.kind != NodeKind.MZI:
            continue
        x, y = positions[n.id]
        rect = Rectangle(
            (x - MZI_HALF, y - MZI_HALF),
            2 * MZI_HALF, 2 * MZI_HALF,
            facecolor=COLOR_UNCHARAC, edgecolor="black",
            linewidth=1.2, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            x, y, str(n.internal_ps_index),
            ha="center", va="center", fontsize=7, zorder=4,
        )

    # External PSs.
    for n in graph.nodes:
        if n.kind != NodeKind.EXT_PS:
            continue
        x, y = positions[n.id]
        rect = Rectangle(
            (x - EXT_PS_HALF, y - EXT_PS_HALF),
            2 * EXT_PS_HALF, 2 * EXT_PS_HALF,
            facecolor=COLOR_EXT_PS, edgecolor="black",
            linewidth=0.8, zorder=3,
        )
        ax.add_patch(rect)

    # Port labels.
    for n in graph.nodes:
        if n.kind == NodeKind.INPUT:
            x, y = positions[n.id]
            ax.text(x - 0.15, y, f"in{n.port}", ha="right", va="center", fontsize=8)
        elif n.kind == NodeKind.OUTPUT:
            x, y = positions[n.id]
            ax.text(x + 0.15, y, f"out{n.port}", ha="left", va="center", fontsize=8)

    ax.set_xlim(-2.5, n_cols + 1.5)
    ax.set_ylim(mesh.m - 0.5, -0.7)   # invert y so wg 0 is at the top
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Step rendering
# ---------------------------------------------------------------------------


def _state_colours(
    step: CharacterizationStep,
    graph: PICGraph,
    characterized_before: set[int],
) -> tuple[dict[int, str], dict[int, str], set[int]]:
    """Return (mzi_face, ext_face, mzi_edge_red).

    - ``mzi_face[node_id]`` is the face colour for each MZI.
    - ``ext_face[node_id]`` is the face colour for each external PS.
    - ``mzi_edge_red`` is the set of MZI node ids that should be drawn
      with a red outline (target of the step OR head/tail of a meta-MZI).
    """
    mzi_face: dict[int, str] = {}
    ext_face: dict[int, str] = {}
    mzi_edge_red: set[int] = set()

    routing_mzi_ids = {s.mzi_node_id for s in step.routing}

    for n in graph.nodes:
        if n.kind == NodeKind.MZI:
            if n.id in characterized_before:
                mzi_face[n.id] = COLOR_CHARAC
            else:
                mzi_face[n.id] = COLOR_UNCHARAC
            if n.id in routing_mzi_ids:
                mzi_face[n.id] = COLOR_CHARAC
        elif n.kind == NodeKind.EXT_PS:
            ext_face[n.id] = COLOR_EXT_PS

    if step.kind == StepKind.META_MZI:
        if step.meta_mzi_head is not None:
            mzi_face[step.meta_mzi_head] = COLOR_META
            mzi_edge_red.add(step.meta_mzi_head)
        if step.meta_mzi_tail is not None:
            mzi_face[step.meta_mzi_tail] = COLOR_META
            mzi_edge_red.add(step.meta_mzi_tail)
        ext_face[step.ps_node_id] = COLOR_EXT_PS_TARGET
    else:
        # internal-PS step: highlight the target MZI.
        if step.ps_node_id in mzi_face:
            mzi_edge_red.add(step.ps_node_id)

    return mzi_face, ext_face, mzi_edge_red


def _draw_light_path(
    ax: Axes,
    step: CharacterizationStep,
    graph: PICGraph,
    positions: dict[int, tuple[float, float]],
    n_cols: int,
) -> None:
    """Overlay the light path as a thick red polyline.

    The path goes:
      input port -> routing-MZI centres (in routing order)
                 -> target MZI / meta-MZI head/tail
                 -> output port.

    For DIRECT_PATH steps we have a precomputed ``direct_path_node_ids``
    that traces the actual node sequence -- we use that.

    For other step kinds we connect a sequence of key points: input port,
    then routing MZIs in order, then target (or head, target ext-PS,
    tail for META_MZI), then output port.
    """
    if not step.reachable:
        return

    if step.kind == StepKind.DIRECT_PATH and step.direct_path_node_ids:
        # Trace through the recorded node sequence.
        xs, ys = [], []
        for nid in step.direct_path_node_ids:
            x, y = positions[nid]
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, color=COLOR_LIGHT, linewidth=3.0, alpha=0.8, zorder=2)
        return

    # Build the key-point list.
    key_ids: list[int] = []
    if step.input_port >= 0:
        # Find the INPUT node with this port.
        for n in graph.nodes:
            if n.kind == NodeKind.INPUT and n.port == step.input_port:
                key_ids.append(n.id)
                break

    for s in step.routing:
        key_ids.append(s.mzi_node_id)

    if step.kind == StepKind.META_MZI:
        if step.ps_node_id not in key_ids:
            # Drop ps node (ext PS) in the middle between head and tail.
            key_ids.append(step.ps_node_id)
    else:
        if step.ps_node_id not in key_ids:
            key_ids.append(step.ps_node_id)

    if step.output_port >= 0:
        for n in graph.nodes:
            if n.kind == NodeKind.OUTPUT and n.port == step.output_port:
                key_ids.append(n.id)
                break

    xs = [positions[nid][0] for nid in key_ids]
    ys = [positions[nid][1] for nid in key_ids]
    ax.plot(xs, ys, color=COLOR_LIGHT, linewidth=3.0, alpha=0.8, zorder=2)


def draw_step(
    step: CharacterizationStep,
    graph: PICGraph,
    mesh: ChipMesh,
    *,
    characterized_before: Optional[set[int]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Render one CharacterizationStep on a fresh PIC drawing."""
    if characterized_before is None:
        characterized_before = set()
    if ax is None:
        _, ax = plt.subplots(figsize=(2 + 0.6 * len(mesh.layers) / 4, 0.7 * mesh.m + 1))

    positions, n_cols = _node_positions(mesh, graph)
    mzi_face, ext_face, mzi_edge_red = _state_colours(
        step, graph, characterized_before,
    )

    # Waveguides.
    for wg in range(mesh.m):
        ax.plot(
            [-1, n_cols], [wg, wg],
            color=COLOR_WG, linewidth=1.5, zorder=1,
        )

    # MZIs.
    for n in graph.nodes:
        if n.kind != NodeKind.MZI:
            continue
        x, y = positions[n.id]
        edge_colour = COLOR_TARGET_EDGE if n.id in mzi_edge_red else "black"
        edge_width = 2.2 if n.id in mzi_edge_red else 1.2
        rect = Rectangle(
            (x - MZI_HALF, y - MZI_HALF),
            2 * MZI_HALF, 2 * MZI_HALF,
            facecolor=mzi_face[n.id], edgecolor=edge_colour,
            linewidth=edge_width, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            x, y, str(n.internal_ps_index),
            ha="center", va="center", fontsize=7, zorder=4,
        )

    # External PSs.
    for n in graph.nodes:
        if n.kind != NodeKind.EXT_PS:
            continue
        x, y = positions[n.id]
        edge_colour = (
            COLOR_TARGET_EDGE
            if step.kind == StepKind.META_MZI and step.ps_node_id == n.id
            else "black"
        )
        edge_width = 2.2 if edge_colour == COLOR_TARGET_EDGE else 0.8
        rect = Rectangle(
            (x - EXT_PS_HALF, y - EXT_PS_HALF),
            2 * EXT_PS_HALF, 2 * EXT_PS_HALF,
            facecolor=ext_face[n.id], edgecolor=edge_colour,
            linewidth=edge_width, zorder=3,
        )
        ax.add_patch(rect)

    # Port labels.
    for n in graph.nodes:
        if n.kind == NodeKind.INPUT:
            x, y = positions[n.id]
            highlight = n.port == step.input_port and step.reachable
            ax.text(
                x - 0.15, y, f"in{n.port}",
                ha="right", va="center", fontsize=8,
                color=COLOR_LIGHT if highlight else "black",
                fontweight="bold" if highlight else "normal",
            )
        elif n.kind == NodeKind.OUTPUT:
            x, y = positions[n.id]
            highlight = n.port == step.output_port and step.reachable
            ax.text(
                x + 0.15, y, f"out{n.port}",
                ha="left", va="center", fontsize=8,
                color=COLOR_LIGHT if highlight else "black",
                fontweight="bold" if highlight else "normal",
            )

    # Light path overlay.
    _draw_light_path(ax, step, graph, positions, n_cols)

    # Head / Tail annotations for meta-MZI.
    if step.kind == StepKind.META_MZI:
        if step.meta_mzi_head is not None:
            hx, hy = positions[step.meta_mzi_head]
            ax.text(
                hx, hy - MZI_HALF - 0.15, "H",
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=COLOR_META,
            )
        if step.meta_mzi_tail is not None:
            tx, ty = positions[step.meta_mzi_tail]
            ax.text(
                tx, ty - MZI_HALF - 0.15, "T",
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=COLOR_META,
            )

    title = (
        f"step {step.order}: {step.kind.value}  "
        f"(PS {step.ps_index})"
    )
    if not step.reachable:
        title += "  [unreachable]"
    ax.set_title(title, fontsize=9)

    ax.set_xlim(-2.5, n_cols + 1.5)
    ax.set_ylim(mesh.m - 0.5, -0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def draw_protocol(
    protocol: list[CharacterizationStep],
    graph: PICGraph,
    mesh: ChipMesh,
    *,
    cols: int = 4,
    figsize_per_subplot: tuple[float, float] = (3.5, 2.5),
) -> Figure:
    """Grid of subplots, one per step, in protocol order.

    The running characterized set is propagated so each frame correctly
    colours green MZIs done in prior frames.
    """
    n = len(protocol)
    if n == 0:
        fig, _ = plt.subplots()
        return fig
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * figsize_per_subplot[0], rows * figsize_per_subplot[1]),
        squeeze=False,
    )

    characterized_before: set[int] = set()
    for i, step in enumerate(protocol):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        draw_step(
            step, graph, mesh,
            characterized_before=set(characterized_before),
            ax=ax,
        )
        if step.kind != StepKind.META_MZI:
            characterized_before.add(step.ps_node_id)

    # Blank out unused subplots.
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# One-shot convenience
# ---------------------------------------------------------------------------


def show_protocol(
    mesh: ChipMesh,
    *,
    cols: int = 4,
) -> Figure:
    """Generate the protocol for ``mesh`` and draw it as a grid figure.

    Convenience wrapper around :func:`draw_protocol` for notebook use.
    """
    graph = build_from_mesh(mesh)
    protocol = generate_ifm_protocol(mesh)
    return draw_protocol(protocol, graph, mesh, cols=cols)
