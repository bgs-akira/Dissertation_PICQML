"""IFM-protocol generator (Section C.3 of the supplement).

Given a :class:`~src.chip_mesh.ChipMesh`, produce an ordered list of
:class:`CharacterizationStep` entries: one step per phase shifter to
sweep, each carrying the input port, output port and routing
configuration that puts light through the target PS without uncontrolled
self-interference (Fig. 4).

The order respects the dependency structure of the supplement's
algorithm: every characterized-MZI router referenced by a later step
appears as the target of an earlier step. External-PS steps come after
all MZI internal-PS steps, since the meta-MZI construction (Fig. 8b)
relies on having an already-characterized router pool.

This module also exposes an ASCII renderer (``render_step_ascii``,
``render_protocol_ascii``) that mirrors ``ChipMesh.to_ascii`` and lets
you visually scan a protocol without pulling in matplotlib. For
paper-style figures see :mod:`src.routing_viz`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional

from src.chip_mesh import (
    BeamsplitterLayer,
    ChipMesh,
    PhaseShifterLayer,
)
from src.pic_graph import (
    NodeKind,
    PICGraph,
    build_from_mesh,
    downstream_mzi_neighbours,
    find_direct_paths,
    find_routed_path,
    illumination,
    upstream_mzi_neighbours,
)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


class StepKind(Enum):
    """Where a characterization step sits in the protocol (§C.3.1, §C.3.2)."""

    DIRECT_PATH = "direct_path"   # internal PS on a Fig. 6-step-1 diagonal
    CHILD = "child"               # internal PS, routed through chars. forward
    PARENT = "parent"             # internal PS, routed backwards
    META_MZI = "meta_mzi"         # external PS, enclosed in head+tail balanced


# Roles for MZISetting.role -- documented strings, narrow on purpose.
_ROLE_ROUTER_BAR = "router_bar"
_ROLE_ROUTER_CROSS = "router_cross"
_ROLE_META_HEAD = "meta_head_balanced"
_ROLE_META_TAIL = "meta_tail_balanced"


@dataclass(frozen=True)
class MZISetting:
    """One MZI pinned to a configuration during a characterization sweep."""

    mzi_node_id: int            # PICGraph node id
    internal_ps_index: int      # ChipMesh global PS index
    target_phase: float         # 0 = cross, pi/2 = balanced, pi = bar
    role: str                   # router_bar | router_cross
                                # | meta_head_balanced | meta_tail_balanced


@dataclass(frozen=True)
class CharacterizationStep:
    """One entry of the generated IFM protocol."""

    order: int
    kind: StepKind
    ps_node_id: int
    ps_index: int               # ChipMesh global PS index of the target PS
    input_port: int
    output_port: int
    routing: tuple[MZISetting, ...] = ()
    reachable: bool = True

    # Optional metadata for debugging / phase-offset computation.
    direct_path_node_ids: tuple[int, ...] = ()
    meta_mzi_head: Optional[int] = None
    meta_mzi_tail: Optional[int] = None


# ---------------------------------------------------------------------------
# MZI internal-PS protocol (§C.3.1, Fig. 6)
# ---------------------------------------------------------------------------


def _path_port(graph: PICGraph, path: list[int], *, end: str) -> int:
    """Return the INPUT or OUTPUT port at the requested end of ``path``."""
    nid = path[0] if end == "start" else path[-1]
    node = graph.nodes[nid]
    if node.kind not in (NodeKind.INPUT, NodeKind.OUTPUT) or node.port is None:
        raise ValueError(
            f"path {end} is not a port node: {nid} ({node.kind.value})"
        )
    return node.port


def _mzis_on_path(graph: PICGraph, path: list[int]) -> list[int]:
    return [nid for nid in path if graph.nodes[nid].kind == NodeKind.MZI]


def _settings_from_route(
    route_settings: list[tuple[int, str, int]],
) -> tuple[MZISetting, ...]:
    """Wrap ``find_routed_path`` settings into ``MZISetting`` objects."""
    out: list[MZISetting] = []
    for mzi_id, config, ps_idx in route_settings:
        if config == "bar":
            out.append(
                MZISetting(
                    mzi_node_id=mzi_id,
                    internal_ps_index=ps_idx,
                    target_phase=math.pi,
                    role=_ROLE_ROUTER_BAR,
                )
            )
        elif config == "cross":
            out.append(
                MZISetting(
                    mzi_node_id=mzi_id,
                    internal_ps_index=ps_idx,
                    target_phase=0.0,
                    role=_ROLE_ROUTER_CROSS,
                )
            )
        else:
            raise ValueError(f"unknown route config {config!r}")
    return tuple(out)


def _step_1_direct_paths(
    graph: PICGraph,
    characterized: set[int],
) -> list[CharacterizationStep]:
    """Emit a DIRECT_PATH step for every MZI lying on a direct path from
    an INPUT port. Each input is tried in port order; once an MZI is
    covered it is not characterized again from another port.
    """
    steps: list[CharacterizationStep] = []
    for in_id in graph.inputs:
        for path in find_direct_paths(graph, in_id, direction="downstream"):
            mzis = _mzis_on_path(graph, path)
            in_port = _path_port(graph, path, end="start")
            out_port = _path_port(graph, path, end="end")
            for mzi_nid in mzis:
                if mzi_nid in characterized:
                    continue
                mzi = graph.nodes[mzi_nid]
                assert mzi.internal_ps_index is not None
                steps.append(
                    CharacterizationStep(
                        order=-1,  # filled by generate_ifm_protocol
                        kind=StepKind.DIRECT_PATH,
                        ps_node_id=mzi_nid,
                        ps_index=mzi.internal_ps_index,
                        input_port=in_port,
                        output_port=out_port,
                        routing=(),
                        direct_path_node_ids=tuple(path),
                    )
                )
                characterized.add(mzi_nid)
    return steps


def _direct_detection_path(
    graph: PICGraph,
    source_mzi: int,
    *,
    direction: str,
) -> Optional[list[int]]:
    """Find a direct path from ``source_mzi`` to a port in ``direction``."""
    paths = find_direct_paths(graph, source_mzi, direction=direction)
    if not paths:
        return None
    # Prefer the shortest path -- fewer EXT_PS nodes in the detection arm
    # keeps the phase offset computation tractable (App. C.2 eq. 8).
    return min(paths, key=len)


def _route_from_port(
    graph: PICGraph,
    target_mzi: int,
    characterized: set[int],
    *,
    direction: str,
) -> Optional[tuple[int, list[int], tuple[MZISetting, ...]]]:
    """Try each port (input for downstream, output for upstream) and
    return the first one that successfully routes to ``target_mzi``.

    Returns ``(port_index, path, settings)`` or ``None``.
    """
    sources = graph.inputs if direction == "downstream" else graph.outputs
    for src in sources:
        result = find_routed_path(
            graph, src, target_mzi, characterized, direction=direction
        )
        if result is not None:
            path, raw_settings = result
            port = graph.nodes[src].port
            assert port is not None
            return port, path, _settings_from_route(raw_settings)
    return None


def _step_2_children(
    graph: PICGraph,
    characterized: set[int],
) -> list[CharacterizationStep]:
    """Iterate over uncharacterized child MZIs of characterized MZIs.

    For each, route light from an input port through ``characterized``
    routers to the child, then find a direct path from the child to an
    output port. Repeat until no further progress is possible.
    """
    steps: list[CharacterizationStep] = []
    progressed = True
    while progressed:
        progressed = False
        # Frontier: uncharacterized MZIs adjacent to a characterized one.
        frontier: list[int] = []
        for mzi_id in sorted(characterized):
            for child_mzi in downstream_mzi_neighbours(graph, mzi_id):
                if child_mzi not in characterized and child_mzi not in frontier:
                    frontier.append(child_mzi)

        for target_mzi in frontier:
            # Detection: direct path from target's output to an OUTPUT port.
            det_path = _direct_detection_path(
                graph, target_mzi, direction="downstream"
            )
            if det_path is None:
                continue
            out_port = _path_port(graph, det_path, end="end")

            # Routing: from any input port to the target MZI through
            # characterized routers only.
            routed = _route_from_port(
                graph, target_mzi, characterized, direction="downstream"
            )
            if routed is None:
                continue
            in_port, _, settings = routed

            mzi = graph.nodes[target_mzi]
            assert mzi.internal_ps_index is not None
            steps.append(
                CharacterizationStep(
                    order=-1,
                    kind=StepKind.CHILD,
                    ps_node_id=target_mzi,
                    ps_index=mzi.internal_ps_index,
                    input_port=in_port,
                    output_port=out_port,
                    routing=settings,
                    direct_path_node_ids=tuple(det_path),
                )
            )
            characterized.add(target_mzi)
            progressed = True
    return steps


def _step_3_parents(
    graph: PICGraph,
    characterized: set[int],
) -> list[CharacterizationStep]:
    """Mirror of step 2 walking upstream. Used to characterize MZIs that
    sit "above" the original direct paths.
    """
    steps: list[CharacterizationStep] = []
    progressed = True
    while progressed:
        progressed = False
        frontier: list[int] = []
        for mzi_id in sorted(characterized):
            for parent_mzi in upstream_mzi_neighbours(graph, mzi_id):
                if parent_mzi not in characterized and parent_mzi not in frontier:
                    frontier.append(parent_mzi)

        for target_mzi in frontier:
            det_path = _direct_detection_path(
                graph, target_mzi, direction="upstream"
            )
            if det_path is None:
                continue
            in_port = _path_port(graph, det_path, end="end")  # det_path ends at INPUT

            routed = _route_from_port(
                graph, target_mzi, characterized, direction="upstream"
            )
            if routed is None:
                continue
            out_port, _, settings = routed

            mzi = graph.nodes[target_mzi]
            assert mzi.internal_ps_index is not None
            steps.append(
                CharacterizationStep(
                    order=-1,
                    kind=StepKind.PARENT,
                    ps_node_id=target_mzi,
                    ps_index=mzi.internal_ps_index,
                    input_port=in_port,
                    output_port=out_port,
                    routing=settings,
                    direct_path_node_ids=tuple(det_path),
                )
            )
            characterized.add(target_mzi)
            progressed = True
    return steps


def _mzi_protocol(graph: PICGraph) -> list[CharacterizationStep]:
    """Run §C.3.1 in order: direct paths, children, parents."""
    characterized: set[int] = set()
    steps: list[CharacterizationStep] = []
    steps.extend(_step_1_direct_paths(graph, characterized))
    steps.extend(_step_2_children(graph, characterized))
    steps.extend(_step_3_parents(graph, characterized))
    return steps


# ---------------------------------------------------------------------------
# External-PS protocol (§C.3.2, Fig. 8)
# ---------------------------------------------------------------------------


def _meta_mzi_for(
    graph: PICGraph,
    ext_ps_node_id: int,
) -> Optional[tuple[int, int, list[int]]]:
    """Find ``(H, T, interior_mzis)`` for an external PS, or None.

    H is the head MZI, T the tail MZI of the shortest valid meta-MZI
    enclosing the external PS (Fig. 2c, Fig. 8). ``interior_mzis`` is the
    list of MZIs lying on direct light paths between H and T (excluding H
    and T themselves) -- these must be set to bar to redirect light.
    """
    ext_ps = graph.nodes[ext_ps_node_id]
    assert ext_ps.kind == NodeKind.EXT_PS

    # Walk ancestors (parent MZI first, then further upstream) -- the
    # paper's preferred order, App. C.3.2 last paragraph.
    visited_h: set[int] = set()
    stack: list[int] = list(upstream_mzi_neighbours(graph, ext_ps_node_id))

    while stack:
        h = stack.pop(0)
        if h in visited_h:
            continue
        visited_h.add(h)

        h_node = graph.nodes[h]
        # The external PS must be on a direct path downstream from one of
        # H's children, otherwise enclosing it in a meta-MZI is meaningless.
        # We test this by checking that ext_ps gets illuminated when we
        # propagate from H.
        counts_from_h = illumination(graph, h, direction="downstream")
        if counts_from_h.get(ext_ps_node_id, 0) == 0:
            continue

        # T candidates: illumination exactly 2 from H, less than 2 from
        # either child alone.
        L = {nid for nid, c in counts_from_h.items()
              if c == 2 and graph.nodes[nid].kind == NodeKind.MZI}
        if not L:
            stack.extend(upstream_mzi_neighbours(graph, h))
            continue

        top_child, bot_child = h_node.children[0], h_node.children[1]
        counts_top = illumination(graph, top_child, direction="downstream")
        counts_bot = illumination(graph, bot_child, direction="downstream")
        L_top = {nid for nid, c in counts_top.items()
                 if c == 2 and graph.nodes[nid].kind == NodeKind.MZI}
        L_bot = {nid for nid, c in counts_bot.items()
                 if c == 2 and graph.nodes[nid].kind == NodeKind.MZI}
        candidates = L - (L_top | L_bot)

        # Keep only T's that have the external PS on a direct path from H
        # (illumination 1 at ext_ps from each child alone -- i.e. the PS
        # is on exactly one arm of the meta-MZI).
        candidates = {
            t for t in candidates
            if counts_top.get(t, 0) == 1 and counts_bot.get(t, 0) == 1
        }

        # AND: the external PS must sit *between* H and T (i.e. T is
        # downstream of the external PS). Without this, an arbitrary
        # recombination point downstream of H gets picked even when the
        # PS sits on a wire that doesn't reach it.
        counts_from_ext = illumination(graph, ext_ps_node_id, direction="downstream")
        candidates = {t for t in candidates if counts_from_ext.get(t, 0) >= 1}

        if not candidates:
            stack.extend(upstream_mzi_neighbours(graph, h))
            continue

        # Pick the closest T -- equivalent to the smallest node id since
        # ids are assigned in topological order.
        t = min(candidates)

        # Interior MZIs on the paths H -> T (excluding H, T). These are
        # MZIs with illumination 1 in the H-propagation that lie between
        # H and T topologically. We collect them by finding the direct
        # paths from each H-child that terminate at T, then taking the
        # union of MZI nodes on them.
        interior: list[int] = []
        for child in (top_child, bot_child):
            for path in _paths_to(graph, child, t, counts_from_child={
                top_child: counts_top, bot_child: counts_bot
            }[child]):
                for nid in path[:-1]:
                    if (graph.nodes[nid].kind == NodeKind.MZI
                            and nid not in (h, t)
                            and nid not in interior):
                        interior.append(nid)
        return h, t, interior

    return None


def _paths_to(
    graph: PICGraph,
    source: int,
    target: int,
    counts_from_child: dict[int, int],
) -> list[list[int]]:
    """All direct paths from ``source`` (an MZI or EXT_PS) to ``target``
    that stay within illumination-1 nodes of ``counts_from_child``.

    Used to identify the interior MZIs of a meta-MZI.
    """
    paths: list[list[int]] = []

    def dfs(nid: int, path: list[int]) -> None:
        if nid == target:
            paths.append(path[:])
            return
        if counts_from_child.get(nid, 0) > 1:
            return
        for cid in graph.nodes[nid].children:
            if cid < 0:
                continue
            path.append(cid)
            dfs(cid, path)
            path.pop()

    dfs(source, [source])
    return paths


def _external_ps_protocol(
    graph: PICGraph,
    characterized: set[int],
) -> list[CharacterizationStep]:
    """Emit a META_MZI step for every external PS. Unreachable PSs are
    emitted with ``reachable=False``.

    The order in which external PSs are characterized matches the natural
    topo order of their nodes (left-to-right on the chip). This is the
    order in which previously-characterized external PSs end up at zero
    phase inside subsequent meta-MZIs (App. C.3.2 last paragraph).
    """
    steps: list[CharacterizationStep] = []
    for ps_idx in sorted(graph.ext_ps_by_index.keys()):
        ext_id = graph.ext_ps_by_index[ps_idx]
        meta = _meta_mzi_for(graph, ext_id)
        if meta is None:
            steps.append(
                CharacterizationStep(
                    order=-1,
                    kind=StepKind.META_MZI,
                    ps_node_id=ext_id,
                    ps_index=ps_idx,
                    input_port=-1,
                    output_port=-1,
                    routing=(),
                    reachable=False,
                )
            )
            continue
        h, t, interior = meta
        h_node = graph.nodes[h]
        t_node = graph.nodes[t]
        assert h_node.internal_ps_index is not None
        assert t_node.internal_ps_index is not None

        # Find an input port that reaches H through routers.
        routed_in = _route_from_port(
            graph, h, characterized, direction="downstream"
        )
        if routed_in is None:
            # Fall back: H might be directly addressable from an input via
            # a *direct* path (no routers needed).
            for in_id in graph.inputs:
                paths = find_direct_paths(graph, in_id, direction="downstream")
                if any(h in p for p in paths):
                    routed_in = (graph.nodes[in_id].port, [], ())
                    break
        if routed_in is None:
            steps.append(
                CharacterizationStep(
                    order=-1,
                    kind=StepKind.META_MZI,
                    ps_node_id=ext_id,
                    ps_index=ps_idx,
                    input_port=-1,
                    output_port=-1,
                    routing=(),
                    reachable=False,
                    meta_mzi_head=h,
                    meta_mzi_tail=t,
                )
            )
            continue
        in_port, _, in_settings = routed_in

        # Direct path from T to an output.
        det_path = _direct_detection_path(graph, t, direction="downstream")
        if det_path is None:
            steps.append(
                CharacterizationStep(
                    order=-1,
                    kind=StepKind.META_MZI,
                    ps_node_id=ext_id,
                    ps_index=ps_idx,
                    input_port=-1,
                    output_port=-1,
                    routing=(),
                    reachable=False,
                    meta_mzi_head=h,
                    meta_mzi_tail=t,
                )
            )
            continue
        out_port = _path_port(graph, det_path, end="end")

        head_setting = MZISetting(
            mzi_node_id=h,
            internal_ps_index=h_node.internal_ps_index,
            target_phase=math.pi / 2,
            role=_ROLE_META_HEAD,
        )
        tail_setting = MZISetting(
            mzi_node_id=t,
            internal_ps_index=t_node.internal_ps_index,
            target_phase=math.pi / 2,
            role=_ROLE_META_TAIL,
        )
        interior_settings = tuple(
            MZISetting(
                mzi_node_id=ni,
                internal_ps_index=graph.nodes[ni].internal_ps_index,  # type: ignore[arg-type]
                target_phase=math.pi,
                role=_ROLE_ROUTER_BAR,
            )
            for ni in interior
        )
        routing = in_settings + (head_setting, tail_setting) + interior_settings

        steps.append(
            CharacterizationStep(
                order=-1,
                kind=StepKind.META_MZI,
                ps_node_id=ext_id,
                ps_index=ps_idx,
                input_port=in_port,
                output_port=out_port,
                routing=routing,
                direct_path_node_ids=tuple(det_path),
                meta_mzi_head=h,
                meta_mzi_tail=t,
            )
        )
    return steps


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_ifm_protocol(mesh: ChipMesh) -> list[CharacterizationStep]:
    """Generate the V-IFM / phi-IFM protocol for ``mesh`` (§C.3).

    Returns a list of :class:`CharacterizationStep`, ordered as

      1. DIRECT_PATH internal-PS steps,
      2. CHILD internal-PS steps,
      3. PARENT internal-PS steps,
      4. META_MZI external-PS steps (unreachable ones with
         ``reachable=False`` come last within this group, after the
         reachable ones, by their natural topo order).

    ``step.order`` matches the list index.
    """
    graph = build_from_mesh(mesh)
    characterized: set[int] = set()
    mzi_steps = _mzi_protocol(graph)
    for s in mzi_steps:
        characterized.add(s.ps_node_id)
    ext_steps = _external_ps_protocol(graph, characterized)
    raw = mzi_steps + ext_steps
    return [replace(s, order=i) for i, s in enumerate(raw)]


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------


def render_step_ascii(
    step: CharacterizationStep,
    mesh: ChipMesh,
    graph: PICGraph,
    characterized_before: set[int],
    *,
    cell_width: int | None = None,
) -> str:
    """Render one CharacterizationStep in the ``ChipMesh.to_ascii`` style.

    Decorations relative to ``ChipMesh.to_ascii`` (which uses ``P<idx>``
    and ``B<idx>`` markers):

    - ``G<idx>``: characterized MZI's internal PS (already done before
      this step).
    - ``R<idx>``: the PS being characterized in this step (target).
    - ``M<idx>``: meta-MZI head or tail MZI's internal PS (balanced).
    - ``=``-filled cell instead of ``---`` on every waveguide that
      participates in the light path between ``input_port`` and
      ``output_port`` for this step.

    BS cells and unannotated PS cells follow ``ChipMesh.to_ascii``.
    """
    m = mesh.m
    if cell_width is None:
        max_idx = max(mesh.n_BS, mesh.n_PS, len(mesh.layers)) - 1
        cell_width = max(3, 1 + len(str(max(0, max_idx))))
    if cell_width < 3:
        raise ValueError(f"cell_width must be >= 3, got {cell_width}")

    # Map MZI internal PS -> role for this step.
    router_bar_ps: set[int] = set()
    router_cross_ps: set[int] = set()
    meta_head_ps: set[int] = set()
    meta_tail_ps: set[int] = set()
    for s in step.routing:
        if s.role == _ROLE_ROUTER_BAR:
            router_bar_ps.add(s.internal_ps_index)
        elif s.role == _ROLE_ROUTER_CROSS:
            router_cross_ps.add(s.internal_ps_index)
        elif s.role == _ROLE_META_HEAD:
            meta_head_ps.add(s.internal_ps_index)
        elif s.role == _ROLE_META_TAIL:
            meta_tail_ps.add(s.internal_ps_index)

    characterized_ps = {
        graph.nodes[nid].internal_ps_index
        for nid in characterized_before
        if graph.nodes[nid].kind == NodeKind.MZI
    }

    # Light-path bookkeeping: which waveguides are "lit" on each layer.
    # A simple over-approximation suffices for ASCII: mark every waveguide
    # in [min(input, output), max(input, output)] on every layer that
    # contains a routing MZI or the target itself. The rendering is for
    # visual scanning, not exact path drawing.
    lit_wgs: set[int] = set()
    if step.reachable:
        lit_wgs.update(range(
            min(step.input_port, step.output_port),
            max(step.input_port, step.output_port) + 1,
        ))

    fill_dash = "-" * cell_width
    fill_path = "=" * cell_width
    label_width = len(str(m - 1))
    indent = " " * (label_width + 1)

    rows: list[list[str]] = [[] for _ in range(m)]
    for layer in mesh.layers:
        cells = [
            (fill_path if wg in lit_wgs else fill_dash)
            for wg in range(m)
        ]
        if isinstance(layer, PhaseShifterLayer):
            for wg, idx in layer.shifter_indices.items():
                marker = "P"
                if idx == step.ps_index:
                    marker = "R"
                elif idx in meta_head_ps or idx in meta_tail_ps:
                    marker = "M"
                elif idx in router_bar_ps or idx in router_cross_ps:
                    marker = "G"
                elif idx in characterized_ps:
                    marker = "G"
                cells[wg] = f"{marker}{idx:0{cell_width - 1}d}"
        elif isinstance(layer, BeamsplitterLayer):
            for (k_top, k_bot), idx in layer.bs_pairs.items():
                label = f"B{idx:0{cell_width - 1}d}"
                cells[k_top] = label
                cells[k_bot] = label
        for wg in range(m):
            rows[wg].append(cells[wg])

    header = indent + " ".join(
        f"L{i:0{cell_width - 1}d}" for i in range(len(mesh.layers))
    )
    body = [
        f"{wg:>{label_width}d} " + " ".join(row)
        for wg, row in enumerate(rows)
    ]

    info_lines = [
        f"step {step.order}: {step.kind.value}",
        f"  ps_index = {step.ps_index}   "
        f"in = {step.input_port}  out = {step.output_port}   "
        f"reachable = {step.reachable}",
    ]
    if step.routing:
        roles = ", ".join(
            f"PS{s.internal_ps_index}={s.role}" for s in step.routing
        )
        info_lines.append(f"  routing: {roles}")
    if step.meta_mzi_head is not None and step.meta_mzi_tail is not None:
        info_lines.append(
            f"  meta-MZI: head=node{step.meta_mzi_head}, "
            f"tail=node{step.meta_mzi_tail}"
        )
    return "\n".join(info_lines + ["", header] + body)


def render_protocol_ascii(
    protocol: list[CharacterizationStep],
    mesh: ChipMesh,
    graph: PICGraph,
    *,
    cell_width: int | None = None,
) -> str:
    """Concatenate per-step ASCII renderings, separated by blank lines."""
    out: list[str] = []
    characterized_before: set[int] = set()
    for step in protocol:
        out.append(render_step_ascii(
            step, mesh, graph, characterized_before,
            cell_width=cell_width,
        ))
        out.append("")  # blank separator
        if step.kind != StepKind.META_MZI:
            characterized_before.add(step.ps_node_id)
    return "\n".join(out).rstrip()
