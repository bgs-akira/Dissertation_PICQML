"""Tests for ChipMesh topology construction."""

import pytest

from src.chip_mesh import (
    BeamsplitterLayer,
    ChipMesh,
    PhaseShifterLayer,
)


def test_clements_counts_m12():
    """m = 12 produces n_BS = 132 and n_PS = 132 (1 internal + 1 external per MZI)."""
    mesh = ChipMesh.clements(12)
    assert mesh.m == 12
    assert mesh.n_BS == 132
    assert mesh.n_PS == 132


@pytest.mark.parametrize("m", [4, 6, 8, 10, 12])
def test_clements_counts_general(m):
    """n_BS = m(m-1) and n_PS = m(m-1) across the supported range."""
    mesh = ChipMesh.clements(m)
    assert mesh.n_BS == m * (m - 1)
    assert mesh.n_PS == m * (m - 1)


@pytest.mark.parametrize("m", [4, 6, 8, 10, 12])
def test_layer_count(m):
    """Each MZI row expands to four atomic layers (BS_a, PS_int, BS_b, PS_ext)."""
    mesh = ChipMesh.clements(m)
    assert len(mesh.layers) == 4 * m


@pytest.mark.parametrize("m", [4, 6, 8, 10, 12])
def test_no_duplicate_indices(m):
    """Every global PS index appears exactly once, same for BS."""
    mesh = ChipMesh.clements(m)
    ps_seen: list[int] = []
    bs_seen: list[int] = []
    for layer in mesh.layers:
        if isinstance(layer, PhaseShifterLayer):
            ps_seen.extend(layer.shifter_indices.values())
        else:
            bs_seen.extend(layer.bs_pairs.values())
    assert sorted(ps_seen) == list(range(mesh.n_PS))
    assert sorted(bs_seen) == list(range(mesh.n_BS))


@pytest.mark.parametrize("m", [4, 6, 8, 10, 12])
def test_no_overlapping_waveguides_in_column(m):
    """Within any single layer, every waveguide appears in at most one block."""
    mesh = ChipMesh.clements(m)
    for layer in mesh.layers:
        if isinstance(layer, PhaseShifterLayer):
            wgs = list(layer.shifter_indices.keys())
            assert len(set(wgs)) == len(wgs)
        else:
            wgs: list[int] = []
            for k_top, k_bot in layer.bs_pairs.keys():
                wgs.extend([k_top, k_bot])
            assert len(set(wgs)) == len(wgs)


@pytest.mark.parametrize("m", [4, 6, 8, 10, 12])
def test_bs_pairs_adjacent(m):
    """Every BS pair satisfies pair[1] == pair[0] + 1."""
    mesh = ChipMesh.clements(m)
    for layer in mesh.layers:
        if isinstance(layer, BeamsplitterLayer):
            for k_top, k_bot in layer.bs_pairs.keys():
                assert k_bot == k_top + 1


def test_no_external_ps_in_first_ps_column():
    """First PS column (PS_int of row 0, wide) has m/2 internals only."""
    m = 12
    mesh = ChipMesh.clements(m)
    first_ps = mesh.layers[1]
    assert isinstance(first_ps, PhaseShifterLayer)
    assert len(first_ps.shifter_indices) == m // 2
    assert set(first_ps.shifter_indices.keys()) == set(range(0, m, 2))


@pytest.mark.parametrize("m", [4, 6, 8, 10, 12])
def test_no_waveguide_collisions_in_ps_column(m):
    """Within any PhaseShifterLayer, every waveguide index is unique."""
    mesh = ChipMesh.clements(m)
    for layer in mesh.layers:
        if isinstance(layer, PhaseShifterLayer):
            keys = list(layer.shifter_indices.keys())
            assert len(set(keys)) == len(keys)


def test_externals_on_mzi_upper_waveguides():
    """Every PS_ext layer's waveguides match the upper arms of the MZIs in the
    preceding BS_b layer (one external per MZI on the upper waveguide)."""
    m = 12
    mesh = ChipMesh.clements(m)
    for row_start in range(0, len(mesh.layers), 4):
        bs_b = mesh.layers[row_start + 2]
        ps_ext = mesh.layers[row_start + 3]
        assert isinstance(bs_b, BeamsplitterLayer)
        assert isinstance(ps_ext, PhaseShifterLayer)
        upper_waveguides = {k_top for (k_top, _) in bs_b.bs_pairs.keys()}
        assert set(ps_ext.shifter_indices.keys()) == upper_waveguides


def test_internals_on_mzi_upper_waveguides():
    """Every PS_int layer's waveguides match the upper arms of the MZIs in the
    preceding BS_a layer (one internal per MZI on the upper waveguide)."""
    m = 12
    mesh = ChipMesh.clements(m)
    for row_start in range(0, len(mesh.layers), 4):
        bs_a = mesh.layers[row_start]
        ps_int = mesh.layers[row_start + 1]
        assert isinstance(bs_a, BeamsplitterLayer)
        assert isinstance(ps_int, PhaseShifterLayer)
        upper_waveguides = {k_top for (k_top, _) in bs_a.bs_pairs.keys()}
        assert set(ps_int.shifter_indices.keys()) == upper_waveguides


def test_mode_0_ps_count_m12():
    """Mode 0 is the upper waveguide of the (0, 1) MZI in every wide row.
    Each of the m/2 wide rows places one internal and one external on mode 0,
    so mode 0 hosts exactly m PSs at m = 12."""
    m = 12
    mesh = ChipMesh.clements(m)
    count = sum(
        1
        for layer in mesh.layers
        if isinstance(layer, PhaseShifterLayer) and 0 in layer.shifter_indices
    )
    assert count == m


def test_mode_m_minus_1_has_no_ps_m12():
    """Mode m-1 is only ever the LOWER waveguide of the (m-2, m-1) MZI. The
    standard-Clements convention places internals and externals on upper arms
    only, so mode m-1 hosts zero PSs. The paper's ITM stage handles edge
    transmissions separately, so this asymmetry is acceptable here."""
    m = 12
    mesh = ChipMesh.clements(m)
    count = sum(
        1
        for layer in mesh.layers
        if isinstance(layer, PhaseShifterLayer) and (m - 1) in layer.shifter_indices
    )
    assert count == 0


@pytest.mark.parametrize("m", [4, 6, 8, 10, 12])
def test_to_ascii_line_count(m):
    """ASCII output has 1 header line plus m waveguide rows."""
    mesh = ChipMesh.clements(m)
    lines = mesh.to_ascii().split("\n")
    assert len(lines) == m + 1


def test_to_ascii_contains_first_and_last_indices_m4():
    """ASCII output for m=4 contains B00, P00 and the maximum BS/PS labels."""
    mesh = ChipMesh.clements(4)
    out = mesh.to_ascii()
    assert "B00" in out
    assert "P00" in out
    assert f"B{mesh.n_BS - 1:02d}" in out
    assert f"P{mesh.n_PS - 1:02d}" in out


def test_to_ascii_first_cell_for_wave_0_is_B00():
    """In m=4, the first cell of waveguide 0's row is B00 (BS_a of (0,1))."""
    mesh = ChipMesh.clements(4)
    lines = mesh.to_ascii().split("\n")
    wave_0_cells = lines[1].split()
    assert wave_0_cells[0] == "0"
    assert wave_0_cells[1] == "B00"


def test_to_ascii_rejects_too_small_cell_width():
    """cell_width < 3 raises ValueError."""
    mesh = ChipMesh.clements(4)
    with pytest.raises(ValueError):
        mesh.to_ascii(cell_width=2)


def test_repr_contains_counts():
    """__repr__ surfaces m, n_BS, n_PS and layer count."""
    mesh = ChipMesh.clements(4)
    text = repr(mesh)
    assert "m=4" in text
    assert "n_BS=12" in text
    assert "n_PS=12" in text


def test_odd_m_rejected():
    """ChipMesh.clements(5) raises ValueError because m must be even."""
    with pytest.raises(ValueError):
        ChipMesh.clements(5)


def test_too_small_m_rejected():
    """ChipMesh.clements(1) raises ValueError because m must be >= 2."""
    with pytest.raises(ValueError):
        ChipMesh.clements(1)
