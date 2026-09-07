"""Tests for the PIC graph builder, illumination and direct-path search.

Hand-derived expectations for m = 4 Clements; structural invariants for
larger m. Numbering follows the construction order in ``ChipMesh.clements``:

  MZI A on (0,1) row 0   -- internal PS 0, external PS 2
  MZI B on (2,3) row 0   -- internal PS 1, external PS 3
  MZI C on (1,2) row 1   -- internal PS 4, external PS 5
  MZI D on (0,1) row 2   -- internal PS 6, external PS 8
  MZI E on (2,3) row 2   -- internal PS 7, external PS 9
  MZI F on (1,2) row 3   -- internal PS 10, external PS 11
"""

from __future__ import annotations

import pytest

from src.chip_mesh import ChipMesh
from src.pic_graph import (
    NodeKind,
    build_from_mesh,
    downstream_mzi_neighbours,
    find_direct_paths,
    find_routed_path,
    illumination,
    upstream_mzi_neighbours,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [4, 6, 8])
def test_graph_counts_clements(m: int) -> None:
    mesh = ChipMesh.clements(m)
    graph = build_from_mesh(mesh)
    n_mzi = m * (m - 1) // 2
    assert len(graph.inputs) == m
    assert len(graph.outputs) == m
    assert len(graph.mzi_by_internal_ps) == n_mzi
    assert len(graph.ext_ps_by_index) == n_mzi


def test_graph_mzi_children_have_length_two_m4() -> None:
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    for n in graph.nodes:
        if n.kind == NodeKind.MZI:
            assert len(n.parents) == 2
            assert len(n.children) == 2


def test_graph_input_output_reach_m4() -> None:
    """From every input, illumination reaches every output (chip is dense)."""
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    for in_id in graph.inputs:
        counts = illumination(graph, in_id, direction="downstream")
        for out_id in graph.outputs:
            assert counts[out_id] > 0


def test_graph_mzi_ext_ps_indices_unique_m4() -> None:
    """Every internal PS index and every external PS index maps to one node."""
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    int_ids = {n.internal_ps_index for n in graph.nodes if n.kind == NodeKind.MZI}
    ext_ids = {n.external_ps_index for n in graph.nodes if n.kind == NodeKind.EXT_PS}
    assert int_ids.isdisjoint(ext_ids)


# ---------------------------------------------------------------------------
# Illumination (Fig. 7)
# ---------------------------------------------------------------------------


def _mzi_by_int_ps(graph, ps: int) -> int:
    return graph.mzi_by_internal_ps[ps]


def test_illumination_from_in0_m4() -> None:
    """Hand-derived illumination from INPUT 0 on m=4 Clements.

    Beam enters MZI A (1 beam), splits to EXT_PS(2)=1 + MZI_C=1, etc.
    """
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    in0 = graph.inputs[0]
    counts = illumination(graph, in0, direction="downstream")

    A = _mzi_by_int_ps(graph, 0)   # MZI on (0,1), row 0
    B = _mzi_by_int_ps(graph, 1)
    C = _mzi_by_int_ps(graph, 4)
    D = _mzi_by_int_ps(graph, 6)
    E = _mzi_by_int_ps(graph, 7)
    F = _mzi_by_int_ps(graph, 10)

    assert counts[A] == 1
    assert counts[B] == 0   # in0 cannot reach MZI B
    assert counts[C] == 1
    assert counts[D] == 2   # two paths via MZI A -> {top, bottom} -> MZI D
    assert counts[E] == 1
    assert counts[F] == 3   # 2 (via D) + 1 (via E)


def test_illumination_from_output_upstream_m4() -> None:
    """Upstream from OUTPUT 0 should mirror downstream from INPUT 0
    (the m=4 Clements is symmetric under input<->output swap)."""
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    out0 = graph.outputs[0]
    counts = illumination(graph, out0, direction="upstream")
    # All four MZIs reachable from out0: D (direct), A (via D's top input),
    # C (via D's bottom input), B (via C's bottom input).
    D = _mzi_by_int_ps(graph, 6)
    A = _mzi_by_int_ps(graph, 0)
    C = _mzi_by_int_ps(graph, 4)
    B = _mzi_by_int_ps(graph, 1)
    assert counts[D] == 1
    assert counts[A] >= 1
    assert counts[C] >= 1
    assert counts[B] >= 1


# ---------------------------------------------------------------------------
# Direct paths (Fig. 4 / Fig. 7)
# ---------------------------------------------------------------------------


def test_direct_paths_from_in0_m4() -> None:
    """The direct path from INPUT 0 reaches OUTPUT 3 via MZI_A -> MZI_C -> MZI_E."""
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    in0 = graph.inputs[0]
    paths = find_direct_paths(graph, in0, direction="downstream")
    assert len(paths) >= 1
    mzi_nodes_per_path = [
        [n for n in path if graph.nodes[n].kind == NodeKind.MZI]
        for path in paths
    ]
    expected_mzis = {
        _mzi_by_int_ps(graph, 0),    # A
        _mzi_by_int_ps(graph, 4),    # C
        _mzi_by_int_ps(graph, 7),    # E
    }
    assert any(set(mzis) == expected_mzis for mzis in mzi_nodes_per_path)


def test_direct_paths_from_in2_m4() -> None:
    """INPUT 2's direct path goes MZI_B -> MZI_C -> MZI_D -> OUT 0."""
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    in2 = graph.inputs[2]
    paths = find_direct_paths(graph, in2, direction="downstream")
    expected_mzis = {
        _mzi_by_int_ps(graph, 1),    # B
        _mzi_by_int_ps(graph, 4),    # C
        _mzi_by_int_ps(graph, 6),    # D
    }
    mzi_nodes_per_path = [
        [n for n in path if graph.nodes[n].kind == NodeKind.MZI]
        for path in paths
    ]
    assert any(set(mzis) == expected_mzis for mzis in mzi_nodes_per_path)


# ---------------------------------------------------------------------------
# Routed paths
# ---------------------------------------------------------------------------


def test_find_routed_path_m4() -> None:
    """Once MZIs A and D are characterized, light can be routed from
    input 0 through them to MZI F (cross at A and cross at D)."""
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    A = _mzi_by_int_ps(graph, 0)
    D = _mzi_by_int_ps(graph, 6)
    F = _mzi_by_int_ps(graph, 10)
    in0 = graph.inputs[0]

    result = find_routed_path(
        graph, in0, F, characterized={A, D}, direction="downstream",
    )
    assert result is not None
    path, settings = result
    assert path[0] == in0
    assert path[-1] == F
    # The route must only set bar/cross on characterized MZIs.
    mzi_ids_in_settings = {s[0] for s in settings}
    assert mzi_ids_in_settings <= {A, D}


def test_find_routed_path_blocked_by_uncharacterized() -> None:
    """With no MZIs characterized, the only reachable routed path is the
    one-step (in0 -> first MZI) -- no path to a deeper target."""
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    F = _mzi_by_int_ps(graph, 10)
    result = find_routed_path(
        graph, graph.inputs[0], F, characterized=set(),
        direction="downstream",
    )
    assert result is None


# ---------------------------------------------------------------------------
# Neighbour helpers
# ---------------------------------------------------------------------------


def test_downstream_mzi_neighbours_traverses_ext_ps_m4() -> None:
    """MZI A's downstream MZI neighbours are MZI C (direct on wg 1) and
    MZI D (via EXT_PS(2) on wg 0)."""
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    A = _mzi_by_int_ps(graph, 0)
    C = _mzi_by_int_ps(graph, 4)
    D = _mzi_by_int_ps(graph, 6)
    neighbours = set(downstream_mzi_neighbours(graph, A))
    assert neighbours == {C, D}


def test_upstream_mzi_neighbours_m4() -> None:
    """MZI F's upstream MZI neighbours are MZI D and MZI E (via EXT_PS(9))."""
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    D = _mzi_by_int_ps(graph, 6)
    E = _mzi_by_int_ps(graph, 7)
    F = _mzi_by_int_ps(graph, 10)
    neighbours = set(upstream_mzi_neighbours(graph, F))
    assert neighbours == {D, E}
