"""PIC graph representation for the Section C characterization algorithm.

Builds an upstream/downstream graph from a ``ChipMesh``. Each MZI becomes
one node (with the two beamsplitters and the internal PS folded in); each
external PS becomes its own node; the m input and m output ports are
explicit nodes. Edges encode the propagation direction.

For an MZI on waveguides ``(wg_top, wg_top + 1)`` the parents and children
lists are *side-indexed*:

    parents[0]  = component on waveguide wg_top      (top input)
    parents[1]  = component on waveguide wg_top + 1  (bottom input)
    children[0] = component on waveguide wg_top      (top output)
    children[1] = component on waveguide wg_top + 1  (bottom output)

For an external PS sitting after an MZI on ``wg_top`` we keep a single
parent and a single child -- the PS doesn't branch -- but they preserve
the wire direction (``parents[0]`` is the host MZI, ``children[0]`` is
whatever is next on the same wire).

The graph is built by walking ``ChipMesh.layers`` left to right. The
Clements convention from ``chip_mesh.py`` always pairs BS_a, PS_int,
BS_b, PS_ext per MZI row, so we consume the layers in 4-tuples.

This module is intentionally self-contained: ``ChipMesh`` is the only
project import, so ``routing.py`` and ``routing_viz.py`` can build on it
without introducing a cycle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from src.chip_mesh import (
    BeamsplitterLayer,
    ChipMesh,
    PhaseShifterLayer,
)


class NodeKind(Enum):
    """Kinds of node in the PIC graph (Fig. 5 of the supplement)."""

    INPUT = "input"
    OUTPUT = "output"
    MZI = "mzi"
    EXT_PS = "external_ps"


@dataclass
class Node:
    """One graph node.

    ``parents`` and ``children`` hold neighbouring node ids. For MZIs the
    lists have length 2 (slot 0 = top waveguide, slot 1 = bottom waveguide).
    For EXT_PS, INPUT, OUTPUT they have length 0 or 1.

    The MZI-specific fields (``wg_top``, ``bs_a_index``, ``bs_b_index``,
    ``internal_ps_index``) are set on MZI nodes only; the EXT_PS-specific
    field (``external_ps_index``) on EXT_PS nodes only; and ``port`` on
    INPUT / OUTPUT nodes only. The unused fields stay ``None``.
    """

    id: int
    kind: NodeKind
    parents: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)

    wg_top: Optional[int] = None
    bs_a_index: Optional[int] = None
    bs_b_index: Optional[int] = None
    internal_ps_index: Optional[int] = None

    external_ps_index: Optional[int] = None

    port: Optional[int] = None


@dataclass
class PICGraph:
    """A built PIC graph.

    Node ids are assigned in topological order: INPUT nodes first, then
    MZIs / EXT_PSes in left-to-right layer order, then OUTPUT nodes last.
    ``illumination`` and the path searches all rely on this property.
    """

    m: int
    nodes: list[Node]
    inputs: list[int]
    outputs: list[int]
    mzi_by_internal_ps: dict[int, int]
    ext_ps_by_index: dict[int, int]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_from_mesh(mesh: ChipMesh) -> PICGraph:
    """Translate a ChipMesh into a PICGraph (Fig. 5).

    Assumes the Clements 4-layer block convention (BS_a, PS_int, BS_b,
    PS_ext) -- the same convention enforced by ``ChipMesh.clements()``.
    Raises ``ValueError`` if a different layer ordering is encountered.

    **Clements only.** A Bell mesh expands each column to three layers
    (BS_a, phase layer carrying both arms, BS_b) with no external PS, so
    the 4-tuple walk below does not apply and the meta-MZI construction
    that ``routing.py`` builds on top of this graph assumes external PSs
    that a Bell mesh does not have. Porting the Supplement C.3 protocol
    generator to Bell is future work; it is not on the ML critical path
    (CLAUDE.md pitfall 15 -- synthetic phi-IFM needs no routing).
    """
    if mesh.scheme != "clements":
        raise ValueError(
            f"build_from_mesh supports the Clements scheme only, got "
            f"scheme={mesh.scheme!r}. The routing/protocol modules "
            f"(pic_graph, routing, routing_viz) have not been ported to "
            f"the Bell mesh; they are unused by the ML and phi-IFM "
            f"stages."
        )
    m = mesh.m
    nodes: list[Node] = []

    def new_node(**kw) -> Node:
        nid = len(nodes)
        n = Node(id=nid, **kw)
        nodes.append(n)
        return n

    # Create INPUT nodes first so they take ids [0, m).
    input_nodes = [new_node(kind=NodeKind.INPUT, port=k) for k in range(m)]
    for n in input_nodes:
        n.children = [-1]

    # trailing[wg] = (upstream_node_id, child_slot_to_fill).
    # Each waveguide carries the most recent node on that wire along with
    # the slot in its ``children`` list that the next downstream component
    # should occupy.
    trailing: list[tuple[int, int]] = [(n.id, 0) for n in input_nodes]

    def attach(downstream_id: int, wg: int, parent_slot: int) -> None:
        """Set children[slot] on the trailing node of ``wg`` and record the
        reciprocal entry on ``downstream_id.parents[parent_slot]``.
        """
        upstream_id, slot = trailing[wg]
        upstream = nodes[upstream_id]
        while len(upstream.children) <= slot:
            upstream.children.append(-1)
        upstream.children[slot] = downstream_id
        downstream = nodes[downstream_id]
        while len(downstream.parents) <= parent_slot:
            downstream.parents.append(-1)
        downstream.parents[parent_slot] = upstream_id

    # Walk the layers in 4-tuples (BS_a, PS_int, BS_b, PS_ext).
    layers = mesh.layers
    idx = 0
    while idx < len(layers):
        L_bsa = layers[idx]
        if not isinstance(L_bsa, BeamsplitterLayer):
            raise ValueError(
                f"layer {idx}: expected BeamsplitterLayer at the start of "
                f"an MZI block, got {type(L_bsa).__name__}"
            )
        if idx + 3 >= len(layers):
            raise ValueError(
                f"layer {idx}: incomplete MZI block (need 4 layers, only "
                f"{len(layers) - idx} remain)"
            )
        L_psint, L_bsb, L_psext = layers[idx + 1], layers[idx + 2], layers[idx + 3]
        if not isinstance(L_psint, PhaseShifterLayer):
            raise ValueError(
                f"layer {idx + 1}: expected PhaseShifterLayer (internal PS)"
            )
        if not isinstance(L_bsb, BeamsplitterLayer):
            raise ValueError(
                f"layer {idx + 2}: expected BeamsplitterLayer (BS_b)"
            )
        if not isinstance(L_psext, PhaseShifterLayer):
            raise ValueError(
                f"layer {idx + 3}: expected PhaseShifterLayer (external PS)"
            )

        # Iterate by ascending top-waveguide so the resulting node order
        # is "top to bottom within a layer" -- matching the ChipMesh
        # indexing convention.
        for (wg_top, wg_bot), bs_a_idx in sorted(L_bsa.bs_pairs.items()):
            if (wg_top, wg_bot) not in L_bsb.bs_pairs:
                raise ValueError(
                    f"layer {idx + 2}: missing BS_b on pair "
                    f"({wg_top}, {wg_bot})"
                )
            bs_b_idx = L_bsb.bs_pairs[(wg_top, wg_bot)]
            if wg_top not in L_psint.shifter_indices:
                raise ValueError(
                    f"layer {idx + 1}: missing PS_int on wg {wg_top}"
                )
            ps_int_idx = L_psint.shifter_indices[wg_top]
            if wg_top not in L_psext.shifter_indices:
                raise ValueError(
                    f"layer {idx + 3}: missing PS_ext on wg {wg_top}"
                )
            ps_ext_idx = L_psext.shifter_indices[wg_top]

            mzi = new_node(
                kind=NodeKind.MZI,
                wg_top=wg_top,
                bs_a_index=bs_a_idx,
                bs_b_index=bs_b_idx,
                internal_ps_index=ps_int_idx,
            )
            mzi.parents = [-1, -1]
            mzi.children = [-1, -1]

            # Wire the parents.
            attach(mzi.id, wg_top, parent_slot=0)
            attach(mzi.id, wg_bot, parent_slot=1)

            # External PS on the upper arm, immediately downstream.
            ext = new_node(
                kind=NodeKind.EXT_PS,
                external_ps_index=ps_ext_idx,
            )
            ext.parents = [mzi.id]
            ext.children = [-1]
            mzi.children[0] = ext.id

            # Update trailing pointers.
            trailing[wg_top] = (ext.id, 0)   # next on wg_top fills EXT_PS.children[0]
            trailing[wg_bot] = (mzi.id, 1)   # next on wg_bot fills MZI.children[1]

        idx += 4

    # OUTPUT nodes last, in port order.
    output_nodes: list[Node] = []
    for wg in range(m):
        out = new_node(kind=NodeKind.OUTPUT, port=wg)
        out.parents = [-1]
        output_nodes.append(out)
        attach(out.id, wg, parent_slot=0)

    mzi_by_int_ps = {
        n.internal_ps_index: n.id for n in nodes if n.kind == NodeKind.MZI
    }
    ext_ps_by_idx = {
        n.external_ps_index: n.id for n in nodes if n.kind == NodeKind.EXT_PS
    }

    graph = PICGraph(
        m=m,
        nodes=nodes,
        inputs=[n.id for n in input_nodes],
        outputs=[n.id for n in output_nodes],
        mzi_by_internal_ps=mzi_by_int_ps,
        ext_ps_by_index=ext_ps_by_idx,
    )
    _validate(graph)
    return graph


def _validate(graph: PICGraph) -> None:
    """Cheap structural checks. Raises ``ValueError`` on violation."""
    for n in graph.nodes:
        for cid in n.children:
            if cid == -1:
                raise ValueError(
                    f"node {n.id} ({n.kind.value}): unfilled child slot"
                )
            if n.id not in graph.nodes[cid].parents:
                raise ValueError(
                    f"node {n.id}: child {cid} does not list it as a parent"
                )
        for pid in n.parents:
            if pid == -1:
                raise ValueError(
                    f"node {n.id} ({n.kind.value}): unfilled parent slot"
                )
            if n.id not in graph.nodes[pid].children:
                raise ValueError(
                    f"node {n.id}: parent {pid} does not list it as a child"
                )


# ---------------------------------------------------------------------------
# Illumination (Fig. 7) and direct paths (Fig. 4, 7)
# ---------------------------------------------------------------------------


def _successors(graph: PICGraph, direction: str) -> Callable[[Node], list[int]]:
    if direction == "downstream":
        return lambda n: n.children
    if direction == "upstream":
        return lambda n: n.parents
    raise ValueError(
        f"direction must be 'downstream' or 'upstream', got {direction!r}"
    )


def illumination(
    graph: PICGraph,
    source_node: int,
    *,
    direction: str = "downstream",
) -> dict[int, int]:
    """Count incident light beams on each node (Fig. 7).

    A unit beam starts at ``source_node``. At each MZI the beam splits in
    two and travels along both outputs (or both inputs, for upstream
    propagation). EXT_PS / INPUT / OUTPUT nodes forward the beam along
    their single neighbour in the chosen direction. The returned dict
    maps every node id to its beam count.
    """
    succ = _successors(graph, direction)
    counts: dict[int, int] = {n.id: 0 for n in graph.nodes}
    counts[source_node] = 1

    # Node ids are assigned in topological order during construction:
    # INPUTs, then layer-major MZI/EXT_PS, then OUTPUTs. Downstream
    # propagation is a forward scan; upstream is the reverse.
    order = range(len(graph.nodes))
    if direction == "upstream":
        order = reversed(order)

    for nid in order:
        c = counts[nid]
        if c == 0:
            continue
        node = graph.nodes[nid]
        for sid in succ(node):
            counts[sid] += c

    return counts


def find_direct_paths(
    graph: PICGraph,
    source_node: int,
    *,
    direction: str = "downstream",
) -> list[list[int]]:
    """All direct light paths from ``source_node`` to a port node.

    A direct path is a sequence ``[source, ..., port]`` such that every
    node on it (intermediates included) has illumination exactly 1 in the
    propagation started at ``source_node``. The target port kind depends
    on direction (OUTPUT for downstream, INPUT for upstream).
    """
    counts = illumination(graph, source_node, direction=direction)
    succ = _successors(graph, direction)
    target_kind = (
        NodeKind.OUTPUT if direction == "downstream" else NodeKind.INPUT
    )

    paths: list[list[int]] = []

    def dfs(nid: int, path: list[int]) -> None:
        if counts[nid] != 1:
            return
        node = graph.nodes[nid]
        if node.kind == target_kind:
            paths.append(path[:])
            return
        for sid in succ(node):
            if counts.get(sid, 0) != 1:
                continue
            path.append(sid)
            dfs(sid, path)
            path.pop()

    dfs(source_node, [source_node])
    return paths


# ---------------------------------------------------------------------------
# Routing through characterized MZIs
# ---------------------------------------------------------------------------


def find_routed_path(
    graph: PICGraph,
    source_node: int,
    target_node: int,
    characterized: set[int],
    *,
    direction: str = "downstream",
) -> Optional[tuple[list[int], list[tuple[int, str, int]]]]:
    """Route light from ``source_node`` to ``target_node``.

    At each *characterized* MZI on the route we may choose either the bar
    or the cross configuration to pass through. Uncharacterized MZIs
    cannot be crossed -- the light would interfere with itself in an
    unknown way (Fig. 4 right panel: the red wavy light).

    Returns ``(path, settings)`` on success, or ``None`` if no route is
    reachable. ``settings`` is a list of ``(mzi_node_id, "bar"|"cross",
    internal_ps_index)`` tuples in the order they appear on the route.
    The caller wraps these into :class:`routing.MZISetting` objects.

    The ``source_node`` and ``target_node`` MZIs (if either is an MZI)
    are *not* added to settings -- the source is where light is injected
    and the target is the one being characterized (its config is set by
    the experiment, not by us).
    """
    succ = _successors(graph, direction)
    pred = _successors(graph, "upstream" if direction == "downstream" else "downstream")

    def entry_slot_for(parent_node: Node, child_id: int) -> Optional[int]:
        """Which slot of ``child`` does ``parent_node`` occupy?

        For a downstream walk we ask "at the next node, which parent slot
        am I?". For an upstream walk we ask "at the next node, which
        child slot am I?".
        """
        child = graph.nodes[child_id]
        if child.kind != NodeKind.MZI:
            return None
        neighbours = child.parents if direction == "downstream" else child.children
        try:
            return neighbours.index(parent_node.id)
        except ValueError:
            return None

    # BFS state: (node_id, entry_slot, path, settings).
    # entry_slot is the slot we *entered* the current node via, or None
    # for non-MZI nodes / for the source.
    State = tuple[int, Optional[int]]
    start: State = (source_node, None)
    queue: deque[tuple[int, Optional[int], list[int], list[tuple[int, str, int]]]] = deque(
        [(source_node, None, [source_node], [])]
    )
    visited: set[State] = {start}

    while queue:
        nid, entry, path, settings = queue.popleft()
        if nid == target_node:
            return path, settings
        node = graph.nodes[nid]

        # Determine which successors and which (optional) settings to add.
        candidates: list[tuple[int, list[tuple[int, str, int]]]] = []
        if node.kind == NodeKind.MZI and nid != source_node and nid not in characterized:
            # Uncharacterized MZI -- can't pass through.
            continue
        if node.kind == NodeKind.MZI and nid != source_node and entry is not None:
            # Routing MZI: pick bar (slot s -> slot s) or cross (slot s -> 1 - s).
            assert node.internal_ps_index is not None  # narrow for type checkers
            for config, exit_slot in (("bar", entry), ("cross", 1 - entry)):
                neighbours = succ(node)
                if exit_slot >= len(neighbours):
                    continue
                cid = neighbours[exit_slot]
                if cid < 0:
                    continue
                candidates.append(
                    (cid, settings + [(nid, config, node.internal_ps_index)])
                )
        else:
            # Pass-through node (INPUT, OUTPUT, EXT_PS) or source MZI:
            # every neighbour is a candidate, no settings added.
            for cid in succ(node):
                if cid < 0:
                    continue
                candidates.append((cid, settings))

        for cid, new_settings in candidates:
            child = graph.nodes[cid]
            new_entry = entry_slot_for(node, cid)
            key: State = (cid, new_entry)
            if key in visited:
                continue
            visited.add(key)
            queue.append((cid, new_entry, path + [cid], new_settings))

    return None


# ---------------------------------------------------------------------------
# Helpers used by routing.py
# ---------------------------------------------------------------------------


def downstream_mzi_neighbours(graph: PICGraph, nid: int) -> list[int]:
    """MZIs immediately downstream of ``nid``, traversing EXT_PSes.

    For step 2 (Children) of the MZI protocol we want the MZIs reachable
    via either child of a characterized MZI; an EXT_PS on the upper arm
    is transparent in this sense.
    """
    result: list[int] = []
    seen: set[int] = set()
    stack = list(graph.nodes[nid].children)
    while stack:
        cid = stack.pop()
        if cid < 0 or cid in seen:
            continue
        seen.add(cid)
        c = graph.nodes[cid]
        if c.kind == NodeKind.MZI:
            result.append(cid)
        elif c.kind == NodeKind.EXT_PS:
            stack.extend(c.children)
        # OUTPUT terminates.
    return result


def upstream_mzi_neighbours(graph: PICGraph, nid: int) -> list[int]:
    """MZIs immediately upstream of ``nid``, traversing EXT_PSes."""
    result: list[int] = []
    seen: set[int] = set()
    stack = list(graph.nodes[nid].parents)
    while stack:
        pid = stack.pop()
        if pid < 0 or pid in seen:
            continue
        seen.add(pid)
        p = graph.nodes[pid]
        if p.kind == NodeKind.MZI:
            result.append(pid)
        elif p.kind == NodeKind.EXT_PS:
            stack.extend(p.parents)
    return result
