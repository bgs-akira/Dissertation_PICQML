"""Tests for the diagonal-ordered calibration protocol.

The m = 10 assertions come from section 4.2 of the Prakash calibration
document; the general-m ones check the closed forms hold beyond the
worked example.
"""

import numpy as np
import pytest
import torch

from src.calibration_order import (
    ROUTER_DELTA,
    calibration_order,
    delta_steps,
    diagonal_mzis,
    diagonal_ports,
    main_diagonal,
    n_diagonals,
    router_settings,
)
from src.chip_mesh import ChipMesh
from src.model import DigitalTwin


M = 10


@pytest.fixture(scope="module")
def mesh() -> ChipMesh:
    return ChipMesh.bell(M)


# ---------------------------------------------------------------------------
# Diagonal geometry
# ---------------------------------------------------------------------------


def test_main_diagonal_and_count():
    """m = 10 has 9 diagonals with the main one at k = 4."""
    assert main_diagonal(M) == 4
    assert n_diagonals(M) == 9


def test_ports_match_the_document(mesh):
    """Sections 4.2.1-4.2.4 name these input/output pairs explicitly."""
    expected = {
        4: (9, 0, False),   # 4.2.1 main diagonal
        5: (9, 1, False),   # 4.2.2
        6: (9, 3, False),   # 4.2.3
        7: (9, 5, False),
        8: (9, 7, False),
        3: (0, 7, True),    # 4.2.4, light sent in reverse
        2: (0, 5, True),
        1: (0, 3, True),
        0: (0, 1, True),
    }
    for k, want in expected.items():
        assert diagonal_ports(M, k) == want


def test_every_mzi_belongs_to_exactly_one_diagonal(mesh):
    seen = []
    for k in range(n_diagonals(M)):
        seen.extend(diagonal_mzis(mesh, k))
    assert sorted(seen) == sorted(mesh.mzi_labels())
    assert len(seen) == 45


def test_main_diagonal_membership(mesh):
    """Section 4.3.1 names all nine main-diagonal MZIs."""
    assert sorted(diagonal_mzis(mesh, 4)) == [
        (0, 8), (1, 7), (2, 6), (3, 5), (4, 4),
        (5, 3), (6, 2), (7, 1), (8, 0),
    ]


def test_diagonal_walk_order_follows_travel(mesh):
    """Forward diagonals walk from the injection edge (largest i) inward."""
    assert diagonal_mzis(mesh, 4)[0] == (8, 0)
    assert diagonal_mzis(mesh, 5)[0] == (8, 2)
    # Reversed diagonals inject at mode 0, so they start at the small-i end.
    assert diagonal_mzis(mesh, 3)[0][0] < diagonal_mzis(mesh, 3)[-1][0]


def test_calibration_order_starts_at_the_main_diagonal(mesh):
    """The main diagonal is router-free, so nothing can precede it."""
    order = calibration_order(mesh)
    assert order[0] == main_diagonal(M)
    assert sorted(order) == list(range(n_diagonals(M)))
    assert order == [4, 5, 6, 7, 8, 3, 2, 1, 0]


def test_ports_generalise_beyond_m10():
    """The closed forms hold for other even m."""
    for m in (4, 6, 8, 12):
        k_main = main_diagonal(m)
        assert diagonal_ports(m, k_main) == (m - 1, 0, False)
        for k in range(k_main + 1, n_diagonals(m)):
            in_p, out_p, rev = diagonal_ports(m, k)
            assert (in_p, rev) == (m - 1, False)
            assert 0 <= out_p < m
        for k in range(0, k_main):
            in_p, out_p, rev = diagonal_ports(m, k)
            assert (in_p, rev) == (0, True)
            assert 0 <= out_p < m


def test_diagonal_index_out_of_range_rejected():
    with pytest.raises(ValueError, match="diagonal k must be in"):
        diagonal_ports(M, 9)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------


def test_main_diagonal_needs_no_routers(mesh):
    assert router_settings(mesh, 4) == {}


def test_k5_routers_match_the_document(mesh):
    """Section 4.2.2: MZI (8,0) to bar, (7,1)...(0,8) to cross."""
    routers = router_settings(mesh, 5)
    assert routers[(8, 0)] == "bar"
    for mzi in diagonal_mzis(mesh, 4):
        if mzi != (8, 0):
            assert routers[mzi] == "cross"
    assert set(routers) == set(diagonal_mzis(mesh, 4))


def test_reverse_routers_bar_the_other_end(mesh):
    """Reversed steps inject at mode 0, so the small-i end is barred."""
    routers = router_settings(mesh, 3)
    barred = [mzi for mzi, s in routers.items() if s == "bar"]
    assert barred == [(0, 8)]


def test_router_states_are_known_names(mesh):
    for k in range(n_diagonals(M)):
        for state in router_settings(mesh, k).values():
            assert state in ROUTER_DELTA


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def test_delta_steps_cover_every_mzi_once(mesh):
    steps = delta_steps(mesh)
    assert len(steps) == 45
    assert sorted(s.mzi for s in steps) == sorted(mesh.mzi_labels())


def test_delta_step_sweeps_both_arms(mesh):
    for s in delta_steps(mesh):
        i, j = s.mzi
        assert s.swept_ps == mesh.mzi_arm_ps(i, j)
        assert mesh.ps_label(s.swept_ps[0]) == (i, j)
        assert mesh.ps_label(s.swept_ps[1]) == (i + 1, j)


def test_target_is_never_also_a_router(mesh):
    """A step cannot pin the very MZI it is trying to sweep."""
    for s in delta_steps(mesh):
        assert s.mzi not in s.routers


def test_steps_only_route_through_already_calibrated_diagonals(mesh):
    """Causality: a router must belong to a diagonal done earlier."""
    order = calibration_order(mesh)
    done: set[int] = set()
    current = None
    for s in delta_steps(mesh):
        if s.diagonal != current:
            if current is not None:
                done.add(current)
            current = s.diagonal
        for mzi in s.routers:
            assert ChipMesh.mzi_diagonal(*mzi) in done, (
                f"step on diagonal {s.diagonal} routes through "
                f"{mzi} on diagonal {ChipMesh.mzi_diagonal(*mzi)}, "
                f"which has not been calibrated yet"
            )
    assert order[0] == 4


# ---------------------------------------------------------------------------
# The routing actually works -- simulated, not just bookkept
# ---------------------------------------------------------------------------


def _ideal_model(mesh: ChipMesh) -> DigitalTwin:
    model = DigitalTwin(
        mesh,
        torch.zeros(mesh.n_PS, dtype=torch.float64),
        torch.zeros(mesh.n_PS, dtype=torch.float64),
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.T_logit.fill_(30.0)     # T_out -> 1
    return model


def _set_mzi(mesh, phi, mzi, delta, sigma=0.0):
    up, lo = mesh.mzi_arm_ps(*mzi)
    phi[up] = sigma + delta
    phi[lo] = sigma - delta


def _monitored(model, phi, in_port, out_port, reverse):
    """Reverse injection reads a ROW of U (reciprocity), forward a column."""
    with torch.no_grad():
        U = model._build_U(phi)
        vec = U[in_port, :] if reverse else U[:, in_port]
        inten = vec.real ** 2 + vec.imag ** 2
        return float((inten[out_port] / inten.sum()).item())


@pytest.mark.parametrize("m", [4, 6, 8, 10, 12])
def test_every_step_yields_a_live_fringe(m):
    """The point of the routing: sweeping the target must move the monitor.

    A flat fringe would mean light never reaches the target through the
    chosen routers, and that MZI could not be calibrated at all. This is
    the test that pinned down the bar-the-injection-edge rule -- swapping
    it makes half the diagonals go dark.

    Parametrised over m because the protocol is closed-form in m and the
    port/router formulas were derived from the document's m = 10 example;
    this is what shows they generalise rather than merely coinciding
    there. On an ideal chip the routing is lossless, so contrast is total
    at every m.
    """
    mesh = ChipMesh.bell(m)
    model = _ideal_model(mesh)
    probe = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    worst = 1.0
    for s in delta_steps(mesh):
        vals = []
        for d in probe:
            phi = torch.zeros(mesh.n_PS, dtype=torch.float64)
            for mzi, state in s.routers.items():
                _set_mzi(mesh, phi, mzi, ROUTER_DELTA[state])
            _set_mzi(mesh, phi, s.mzi, float(d))
            vals.append(
                _monitored(model, phi, s.input_port, s.output_port, s.reverse)
            )
        amp = max(vals) - min(vals)
        worst = min(worst, amp)
        assert amp > 0.5, (
            f"m={m}: step on MZI {s.mzi} (k={s.diagonal}) is flat"
        )
    assert worst > 0.99


@pytest.mark.parametrize("m", [4, 6, 8, 10, 12, 14])
def test_protocol_covers_every_mzi_at_any_m(m):
    """Coverage and causality hold for arbitrary even m, not just 10."""
    mesh = ChipMesh.bell(m)
    steps = delta_steps(mesh)
    assert len(steps) == m * (m - 1) // 2
    assert sorted(s.mzi for s in steps) == sorted(mesh.mzi_labels())

    done: set[int] = set()
    current = None
    for s in steps:
        if s.diagonal != current:
            if current is not None:
                done.add(current)
            current = s.diagonal
        for mzi in s.routers:
            assert ChipMesh.mzi_diagonal(*mzi) in done
