"""Tests for the IFM-protocol generator.

Structural invariants on m = 4, 6, 8 Clements meshes; hand-derived
expectations for m = 4 where individual MZIs and their meta-MZI head/tail
candidates can be enumerated.
"""

from __future__ import annotations

import math

import pytest

from src.chip_mesh import ChipMesh
from src.pic_graph import NodeKind, build_from_mesh
from src.routing import (
    CharacterizationStep,
    MZISetting,
    StepKind,
    generate_ifm_protocol,
    render_protocol_ascii,
    render_step_ascii,
)


# ---------------------------------------------------------------------------
# Coverage and ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [4, 6, 8])
def test_every_internal_ps_addressed(m: int) -> None:
    mesh = ChipMesh.clements(m)
    graph = build_from_mesh(mesh)
    protocol = generate_ifm_protocol(mesh)
    internal_ps_steps = [
        s for s in protocol
        if s.kind in (StepKind.DIRECT_PATH, StepKind.CHILD, StepKind.PARENT)
    ]
    addressed = {s.ps_index for s in internal_ps_steps}
    expected = set(graph.mzi_by_internal_ps.keys())
    assert addressed == expected


@pytest.mark.parametrize("m", [4, 6, 8])
def test_every_external_ps_in_protocol(m: int) -> None:
    """Every external PS appears once (reachable or not)."""
    mesh = ChipMesh.clements(m)
    graph = build_from_mesh(mesh)
    protocol = generate_ifm_protocol(mesh)
    ext_steps = [s for s in protocol if s.kind == StepKind.META_MZI]
    addressed = {s.ps_index for s in ext_steps}
    expected = set(graph.ext_ps_by_index.keys())
    assert addressed == expected


@pytest.mark.parametrize("m", [4, 6, 8])
def test_no_duplicate_addressing(m: int) -> None:
    mesh = ChipMesh.clements(m)
    protocol = generate_ifm_protocol(mesh)
    ps_indices = [s.ps_index for s in protocol]
    assert len(ps_indices) == len(set(ps_indices))


@pytest.mark.parametrize("m", [4, 6, 8])
def test_routed_steps_only_pin_earlier_mzis(m: int) -> None:
    """Every MZISetting in a step refers to an MZI whose internal PS was
    characterized in an earlier step."""
    mesh = ChipMesh.clements(m)
    protocol = generate_ifm_protocol(mesh)
    seen_ps: set[int] = set()
    for s in protocol:
        for setting in s.routing:
            assert setting.internal_ps_index in seen_ps, (
                f"step {s.order} ({s.kind.value}) pins PS "
                f"{setting.internal_ps_index} but it has not been "
                f"characterized yet"
            )
        if s.kind != StepKind.META_MZI:
            seen_ps.add(s.ps_index)


@pytest.mark.parametrize("m", [4, 6, 8])
def test_step_order_matches_index(m: int) -> None:
    mesh = ChipMesh.clements(m)
    protocol = generate_ifm_protocol(mesh)
    for i, s in enumerate(protocol):
        assert s.order == i


@pytest.mark.parametrize("m", [4, 6, 8])
def test_protocol_order_deterministic(m: int) -> None:
    mesh = ChipMesh.clements(m)
    p1 = generate_ifm_protocol(mesh)
    p2 = generate_ifm_protocol(mesh)
    assert [(s.kind, s.ps_index, s.input_port, s.output_port) for s in p1] == \
           [(s.kind, s.ps_index, s.input_port, s.output_port) for s in p2]


# ---------------------------------------------------------------------------
# Meta-MZI specifics
# ---------------------------------------------------------------------------


def test_meta_mzi_head_and_tail_balanced_m4() -> None:
    mesh = ChipMesh.clements(4)
    protocol = generate_ifm_protocol(mesh)
    for s in protocol:
        if s.kind != StepKind.META_MZI or not s.reachable:
            continue
        roles = {st.role for st in s.routing}
        assert "meta_head_balanced" in roles
        assert "meta_tail_balanced" in roles
        # And the balanced phase is pi/2.
        for st in s.routing:
            if st.role in ("meta_head_balanced", "meta_tail_balanced"):
                assert math.isclose(st.target_phase, math.pi / 2, abs_tol=1e-12)


def test_meta_mzi_unreachable_external_ps_m4() -> None:
    """For m=4 Clements the rim PSs whose top-arm leads directly to an
    output (PS 8 on MZI_D, PS 11 on MZI_F) have no recombination point
    downstream and so no meta-MZI exists."""
    mesh = ChipMesh.clements(4)
    protocol = generate_ifm_protocol(mesh)
    unreachable = {
        s.ps_index for s in protocol
        if s.kind == StepKind.META_MZI and not s.reachable
    }
    assert unreachable == {8, 11}


def test_meta_mzi_reachable_external_ps_m4() -> None:
    """The remaining external PSs (2, 3, 5, 9) all have a valid meta-MZI
    in m=4 Clements. PS 9 is the surprise -- although MZI_E sits on the
    bottom rim, its external PS still has MZI_F downstream so a meta-MZI
    H=MZI_C, T=MZI_F exists."""
    mesh = ChipMesh.clements(4)
    protocol = generate_ifm_protocol(mesh)
    reachable = {
        s.ps_index for s in protocol
        if s.kind == StepKind.META_MZI and s.reachable
    }
    assert reachable == {2, 3, 5, 9}


# ---------------------------------------------------------------------------
# Routing-config phases
# ---------------------------------------------------------------------------


def test_router_phases_are_bar_or_cross() -> None:
    """Every router setting has phase either 0 (cross) or pi (bar)."""
    mesh = ChipMesh.clements(6)
    protocol = generate_ifm_protocol(mesh)
    for s in protocol:
        for setting in s.routing:
            if setting.role == "router_bar":
                assert math.isclose(setting.target_phase, math.pi, abs_tol=1e-12)
            elif setting.role == "router_cross":
                assert math.isclose(setting.target_phase, 0.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# ASCII renderer
# ---------------------------------------------------------------------------


def test_render_step_ascii_smoke_m4() -> None:
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    protocol = generate_ifm_protocol(mesh)
    for step in protocol:
        text = render_step_ascii(step, mesh, graph, set())
        assert text
        # The header line names every layer (L00, L01, ...).
        assert "L00" in text
        # The step header includes the kind name.
        assert step.kind.value in text


def test_render_step_ascii_marks_target_m4() -> None:
    """The target PS should be rendered with the 'R' marker on its layer."""
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    protocol = generate_ifm_protocol(mesh)
    direct_step = next(s for s in protocol if s.kind == StepKind.DIRECT_PATH)
    text = render_step_ascii(direct_step, mesh, graph, set())
    # R<idx> with idx zero-padded to width-1. For m=4, the largest index is
    # ~15 so cell_width=3 -> 2-digit padding.
    target_token = f"R{direct_step.ps_index:02d}"
    assert target_token in text


def test_render_protocol_ascii_smoke_m4() -> None:
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    protocol = generate_ifm_protocol(mesh)
    text = render_protocol_ascii(protocol, mesh, graph)
    # One block per step, separated by blank lines.
    assert text.count("step ") == len(protocol)


# ---------------------------------------------------------------------------
# Matplotlib renderer (smoke only)
# ---------------------------------------------------------------------------


def test_draw_mesh_smoke_m4() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from src.routing_viz import draw_mesh
    mesh = ChipMesh.clements(4)
    ax = draw_mesh(mesh)
    # 6 MZIs on m=4 Clements; each rectangle is one MZI plus 6 EXT_PSes.
    # Patches include MZIs, EXT_PSes, and possibly background.
    n_rects = sum(1 for p in ax.patches if hasattr(p, "get_width"))
    assert n_rects >= 12  # 6 MZIs + 6 EXT_PSes


def test_draw_step_smoke_m4() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from src.routing_viz import draw_step
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    protocol = generate_ifm_protocol(mesh)
    ax = draw_step(protocol[0], graph, mesh)
    # At least one line drawn (the waveguides), plus the light path overlay.
    assert len(ax.lines) >= mesh.m


def test_draw_protocol_smoke_m4() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from src.routing_viz import draw_protocol
    mesh = ChipMesh.clements(4)
    graph = build_from_mesh(mesh)
    protocol = generate_ifm_protocol(mesh)
    fig = draw_protocol(protocol, graph, mesh, cols=4)
    # Figure built without error and has at least len(protocol) axes.
    assert len(fig.axes) >= len(protocol)
