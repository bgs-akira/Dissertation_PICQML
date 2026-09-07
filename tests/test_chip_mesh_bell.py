"""Tests for the Bell compact mesh and the Prakash (i, j) labelling.

The reference is the Prakash calibration document, section 4, which
specifies the m = 10 chip this project targets:

    1. modes i in [0, 10), layers j in [0, 10)
    2. 100 phase shifters, each labelled (i, j)
    3. each MZI is labelled by its upper internal shifter, so i + j is even
    4. 45 MZIs; the lower arm of MZI (i, j) is (i + 1, j)
    5. diagonal index k = (i + j) / 2, nine diagonals
    6. 10 external ("independent") shifters on modes 0 and 9 at odd layers
    7. no eleventh output layer on this chip -- hence 100 and not 110

Every count below is asserted against those rules rather than against
whatever ``ChipMesh.bell`` happens to produce.
"""

import pytest

from src.chip_mesh import (
    BeamsplitterLayer,
    ChipMesh,
    PhaseShifterLayer,
    build_mesh,
)


M = 10


@pytest.fixture(scope="module")
def mesh() -> ChipMesh:
    return ChipMesh.bell(M)


# ---------------------------------------------------------------------------
# Counts (document section 4, labelling rules 1-4 and 7)
# ---------------------------------------------------------------------------


def test_counts_m10(mesh):
    """100 shifters, 90 couplers, 30 atomic layers, 45 MZIs."""
    assert mesh.m == M
    assert mesh.n_PS == M * M == 100
    assert mesh.n_BS == M * (M - 1) == 90
    assert len(mesh.layers) == 3 * M == 30
    assert len(mesh.mzi_labels()) == M * (M - 1) // 2 == 45


def test_ps_role_counts(mesh):
    """90 internal (2 per MZI), 10 independent, no externals."""
    assert len(mesh.ps_indices_with_role("internal")) == M * (M - 1)
    assert len(mesh.ps_indices_with_role("independent")) == M
    assert len(mesh.ps_indices_with_role("external")) == 0


def test_n_PS_is_m_squared_across_sizes():
    """n_PS = m**2 and n_BS = m(m-1) hold generally, not just at m=10."""
    for m in (4, 6, 8, 10, 12):
        b = ChipMesh.bell(m)
        assert b.n_PS == m * m
        assert b.n_BS == m * (m - 1)
        assert len(b.mzi_labels()) == m * (m - 1) // 2


def test_scheme_tag_and_build_mesh_default(mesh):
    """build_mesh defaults to Bell; the tag round-trips."""
    assert mesh.scheme == "bell"
    assert build_mesh(M).scheme == "bell"
    assert build_mesh(M, "clements").scheme == "clements"


# ---------------------------------------------------------------------------
# Layer structure
# ---------------------------------------------------------------------------


def test_layer_pattern_is_bs_ps_bs(mesh):
    """Each of the m columns expands to exactly (BS, PS, BS)."""
    for j in range(M):
        assert isinstance(mesh.layers[3 * j], BeamsplitterLayer)
        assert isinstance(mesh.layers[3 * j + 1], PhaseShifterLayer)
        assert isinstance(mesh.layers[3 * j + 2], BeamsplitterLayer)


def test_column_pairs_alternate_wide_narrow(mesh):
    """Even columns carry m/2 MZIs, odd columns m/2 - 1, brick-fashion."""
    for j in range(M):
        bs_a = mesh.layers[3 * j]
        bs_b = mesh.layers[3 * j + 2]
        expected = (
            [(k, k + 1) for k in range(0, M, 2)] if j % 2 == 0
            else [(k, k + 1) for k in range(1, M - 2, 2)]
        )
        assert sorted(bs_a.bs_pairs) == expected
        assert sorted(bs_b.bs_pairs) == expected


def test_every_phase_layer_covers_every_mode(mesh):
    """The figure's signature: no waveguide is ever without a shifter.

    A wide column has 5 MZIs x 2 arms = 10; a narrow column has 4 x 2 = 8
    internals plus the 2 independents on the idle boundary modes. This is
    what makes n_PS = m * m.
    """
    for j in range(M):
        ps = mesh.layers[3 * j + 1]
        assert sorted(ps.shifter_indices) == list(range(M))


def test_both_arms_of_every_mzi_are_internal(mesh):
    """Bell's defining feature: a shifter on BOTH arms, no external PS."""
    for (i, j) in mesh.mzi_labels():
        upper, lower = mesh.mzi_arm_ps(i, j)
        assert mesh.ps_roles[upper] == "internal"
        assert mesh.ps_roles[lower] == "internal"


def test_independents_on_boundary_modes_at_odd_layers(mesh):
    """Rule 6: externals sit on modes 0 and 9 at layers 1, 3, 5, 7, 9."""
    labels = sorted(
        mesh.ps_label(idx)
        for idx in mesh.ps_indices_with_role("independent")
    )
    assert labels == sorted(
        (i, j) for j in range(1, M, 2) for i in (0, M - 1)
    )


# ---------------------------------------------------------------------------
# Prakash (i, j) labelling (rules 1-5)
# ---------------------------------------------------------------------------


def test_label_round_trip(mesh):
    """ps_label and ps_index invert each other over all 100 shifters."""
    for idx in range(mesh.n_PS):
        i, j = mesh.ps_label(idx)
        assert 0 <= i < M and 0 <= j < M
        assert mesh.ps_index(i, j) == idx


def test_label_grid_is_complete(mesh):
    """Every (mode, layer) pair in the 10x10 grid names a shifter."""
    seen = {mesh.ps_label(idx) for idx in range(mesh.n_PS)}
    assert seen == {(i, j) for i in range(M) for j in range(M)}


def test_global_index_is_layer_major(mesh):
    """The convention this codebase committed to: global = j * m + i."""
    for idx in range(mesh.n_PS):
        i, j = mesh.ps_label(idx)
        assert idx == j * M + i


def test_mzi_labels_have_even_sum(mesh):
    """Rule 4: every MZI label satisfies i + j even."""
    for (i, j) in mesh.mzi_labels():
        assert (i + j) % 2 == 0


def test_mzi_lower_arm_is_i_plus_one(mesh):
    """Rule 5: MZI (i, j) has arms (i, j) and (i + 1, j)."""
    for (i, j) in mesh.mzi_labels():
        upper, lower = mesh.mzi_arm_ps(i, j)
        assert mesh.ps_label(upper) == (i, j)
        assert mesh.ps_label(lower) == (i + 1, j)


def test_nine_diagonals(mesh):
    """Rule 6 of section 4: k = (i + j) / 2 takes nine values, 0..8."""
    ks = {mesh.mzi_diagonal(i, j) for (i, j) in mesh.mzi_labels()}
    assert ks == set(range(9))


def test_main_diagonal_has_nine_mzis(mesh):
    """The k = 4 main diagonal, calibrated first, holds nine MZIs.

    The document names them (0,8), (1,7), (2,6), (3,5), (4,4), (5,3),
    (6,2), (7,1), (8,0) in sections 4.2.1 and 4.3.1.
    """
    main = sorted(
        (i, j) for (i, j) in mesh.mzi_labels()
        if mesh.mzi_diagonal(i, j) == 4
    )
    assert main == [(0, 8), (1, 7), (2, 6), (3, 5), (4, 4),
                    (5, 3), (6, 2), (7, 1), (8, 0)]


def test_mzi_bs_pairs_are_distinct_and_ordered(mesh):
    """Each MZI owns two distinct couplers: alpha (first) then beta."""
    seen = set()
    for (i, j) in mesh.mzi_labels():
        a, b = mesh.mzi_bs(i, j)
        assert a != b
        assert a not in seen and b not in seen
        seen.update((a, b))
    assert seen == set(range(mesh.n_BS))


def test_labelling_rejected_on_clements():
    """The (i, j) grid is Bell-only; Clements has two PS layers per column."""
    c = build_mesh(M, "clements")
    for call in (
        lambda: c.ps_label(0),
        lambda: c.ps_index(0, 0),
        lambda: c.mzi_labels(),
        lambda: c.mzi_arm_ps(0, 0),
        lambda: c.mzi_bs(0, 0),
    ):
        with pytest.raises(ValueError, match="Bell scheme only"):
            call()


def test_mzi_diagonal_rejects_odd_sum():
    with pytest.raises(ValueError, match="i \\+ j even"):
        ChipMesh.mzi_diagonal(1, 2)


# ---------------------------------------------------------------------------
# Validation / guards
# ---------------------------------------------------------------------------


def test_odd_m_rejected():
    with pytest.raises(ValueError):
        ChipMesh.bell(5)


def test_too_small_m_rejected():
    with pytest.raises(ValueError):
        ChipMesh.bell(1)


def test_build_mesh_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="scheme must be one of"):
        build_mesh(M, "clemens")


def test_indices_are_contiguous(mesh):
    """No gaps or repeats in either global index space."""
    ps_seen, bs_seen = [], []
    for layer in mesh.layers:
        if isinstance(layer, PhaseShifterLayer):
            ps_seen.extend(layer.shifter_indices.values())
        else:
            bs_seen.extend(layer.bs_pairs.values())
    assert sorted(ps_seen) == list(range(mesh.n_PS))
    assert sorted(bs_seen) == list(range(mesh.n_BS))


def test_to_ascii_marks_independents(mesh):
    """Independents render as I<idx>, internals as P<idx>."""
    lines = mesh.to_ascii().split("\n")
    assert len(lines) == M + 1
    top = lines[1]
    bottom = lines[M]
    # Modes 0 and 9 host every independent shifter.
    assert "I" in top and "I" in bottom
    # Interior modes host none.
    assert all("I" not in lines[1 + i] for i in range(1, M - 1))
