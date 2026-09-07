"""Diagonal-ordered calibration protocol for the Bell mesh.

Implements the ordering of section 4.2 of the Prakash calibration
document, generalised to arbitrary even ``m``.

The idea
--------
An MZI labelled ``(i, j)`` (upper mode i, layer j, i + j even) sits on
diagonal ``k = (i + j) / 2``. There are ``m - 1`` diagonals. Light
injected at one edge mode walks along a diagonal and leaves at a
predictable output mode, so a whole diagonal can be calibrated from a
single input/output pair, one MZI at a time.

Calibration proceeds outward from the main diagonal ``k_main = m/2 - 1``,
which is the direct path from mode ``m-1`` to mode ``0`` and needs no
routers at all. Diagonals above it are reached by injecting at ``m-1``
and stepping past the already-calibrated diagonals; diagonals below it
are reached the same way with the light sent in REVERSE from mode 0.

    diagonal            input mode      monitored output
    ------------------------------------------------------
    k == k_main         m - 1           0
    k >  k_main         m - 1           2k - m + 1
    k <  k_main         0               2k + 1

At m = 10 (k_main = 4) this reproduces the document exactly:
k=4 -> 0, k=5 -> 1, k=6 -> 3, k=7 -> 5, k=8 -> 7 forward; and
k=3 -> 7, k=2 -> 5, k=1 -> 3, k=0 -> 1 reversed.

What a step is
--------------
Two kinds, mirroring the document's two sub-procedures:

``DeltaStep``  sweeps the two arms of one MZI and fits
               ``P = a + c sin(theta(i,j) - theta(i+1,j) + pi/2)``,
               yielding ``k`` for both arms and the offset DIFFERENCE
               ``b(i,j) - b(i+1,j)``. That is enough to set ``delta``.

``SigmaStep``  nests the target MZI inside a larger interferometer to
               expose the offset SUM. It yields ``Sigma(i,j)`` relative
               to ``Sigma`` of the reference MZI two modes up, so the
               protocol produces a CHAIN of relative phases which
               ``chain_to_ground`` then anchors at mode 0.

Together they determine every ``b`` up to one global phase, which is
unobservable.

This module is pure topology and bookkeeping: it says what to measure and
how to set the chip, but performs no measurement and holds no model. The
simulation and the fits live in ``src.calibration``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.chip_mesh import ChipMesh


#: Router MZI states. ``delta`` values from the document's equation 6:
#: bar is delta = pi/2, cross is delta = 0, and a 50:50 split is pi/4.
#: ``PHASE`` (delta = pi) turns an MZI into a pure phase element, which is
#: how the Sigma construction exposes the offset sum.
ROUTER_DELTA: dict[str, float] = {
    "cross": 0.0,
    "split": 0.7853981633974483,   # pi / 4
    "bar": 1.5707963267948966,     # pi / 2
    "phase": 3.141592653589793,    # pi
}


@dataclass(frozen=True)
class DeltaStep:
    """Calibrate ``delta`` of one MZI by sweeping each of its arms.

    Attributes:
        mzi:          label ``(i, j)`` of the target.
        diagonal:     ``k = (i + j) / 2``.
        input_port:   mode light is injected into.
        output_port:  monitored mode.
        reverse:      True when light travels right-to-left (the
                      ``k < k_main`` diagonals).
        routers:      MZI label -> router state name, applied before the
                      sweep to steer light onto this diagonal.
        swept_ps:     the two global PS indices swept, upper arm first.
    """

    mzi: tuple[int, int]
    diagonal: int
    input_port: int
    output_port: int
    reverse: bool
    routers: dict[tuple[int, int], str] = field(default_factory=dict)
    swept_ps: tuple[int, int] = (-1, -1)


@dataclass(frozen=True)
class SigmaStep:
    """Calibrate ``Sigma`` of one MZI relative to a reference MZI.

    The document nests the target inside a larger MZI: two neighbours act
    as approximate 50:50 beamsplitters (``delta = pi/4``) while the
    target and the reference act as pure phase elements (``delta = pi``).
    Sweeping then exposes ``Sigma(reference) - Sigma(target)``.

    Attributes:
        mzi:         label ``(i, j)`` of the target.
        reference:   label of the MZI whose Sigma this is measured
                     against, or None for a chain root.
        splitters:   the two MZIs held at delta = pi/4.
        diagonal, input_port, output_port, reverse: as for DeltaStep.
        routers:     router states for everything else on the path.
        needs_fibre_bs: True for the one case the document cannot do
                     on-chip and closes with an off-chip fibre 50:50.
    """

    mzi: tuple[int, int]
    reference: tuple[int, int] | None
    splitters: tuple[tuple[int, int], ...]
    diagonal: int
    input_port: int
    output_port: int
    reverse: bool
    routers: dict[tuple[int, int], str] = field(default_factory=dict)
    needs_fibre_bs: bool = False


def main_diagonal(m: int) -> int:
    """Index of the direct-path diagonal, ``m/2 - 1``."""
    if m < 2 or m % 2:
        raise ValueError(f"m must be even and >= 2, got {m}")
    return m // 2 - 1


def n_diagonals(m: int) -> int:
    """Number of diagonals, ``m - 1``."""
    if m < 2 or m % 2:
        raise ValueError(f"m must be even and >= 2, got {m}")
    return m - 1


def diagonal_ports(m: int, k: int) -> tuple[int, int, bool]:
    """``(input_port, output_port, reverse)`` for diagonal ``k``.

    See the module docstring for the three cases. ``reverse`` marks the
    diagonals the document reaches by injecting at mode 0 instead.
    """
    if not 0 <= k < n_diagonals(m):
        raise ValueError(f"diagonal k must be in [0, {n_diagonals(m)}), got {k}")
    k_main = main_diagonal(m)
    if k == k_main:
        return m - 1, 0, False
    if k > k_main:
        return m - 1, 2 * k - m + 1, False
    return 0, 2 * k + 1, True


def diagonal_mzis(mesh: ChipMesh, k: int) -> list[tuple[int, int]]:
    """MZI labels on diagonal ``k``, ordered along the direction of travel.

    Forward diagonals are walked from the injection edge (largest ``i``)
    towards mode 0; reversed diagonals the other way. The order matters:
    the Sigma chain references the MZI two modes upstream, so a step's
    reference must already have been calibrated.
    """
    labels = [
        (i, j) for (i, j) in mesh.mzi_labels()
        if ChipMesh.mzi_diagonal(i, j) == k
    ]
    _, _, reverse = diagonal_ports(mesh.m, k)
    return sorted(labels, key=lambda ij: ij[0], reverse=not reverse)


def router_settings(mesh: ChipMesh, k: int) -> dict[tuple[int, int], str]:
    """Router states that steer light onto diagonal ``k``.

    Every diagonal lying between the main one and ``k`` has already been
    calibrated by the time ``k`` is reached, so it can be used as a
    router. Within each such diagonal, the MZI at the INJECTION EDGE is
    set to bar, letting light slip past that diagonal, and the rest to
    cross.

    Which end counts as the injection edge follows the direction of
    travel: forward steps inject at mode ``m-1`` so the largest ``i``
    end is barred; reversed steps inject at mode 0 so the smallest ``i``
    end is. Verified by simulation -- with these settings every one of
    the 45 MZIs at m = 10 shows a full-contrast fringe at its monitored
    port, and swapping the two rules makes the opposite half go flat.

    The main diagonal needs no routers: it is the direct path.
    """
    k_main = main_diagonal(mesh.m)
    if k == k_main:
        return {}
    _, _, reverse = diagonal_ports(mesh.m, k)
    between = range(k_main, k) if k > k_main else range(k + 1, k_main + 1)

    routers: dict[tuple[int, int], str] = {}
    for kk in between:
        labels = [
            (i, j) for (i, j) in mesh.mzi_labels()
            if ChipMesh.mzi_diagonal(i, j) == kk
        ]
        entry = (
            min(labels, key=lambda ij: ij[0]) if reverse
            else max(labels, key=lambda ij: ij[0])
        )
        for mzi in labels:
            routers[mzi] = "bar" if mzi == entry else "cross"
    return routers


def delta_steps(mesh: ChipMesh) -> list[DeltaStep]:
    """Every delta-calibration step, in protocol order.

    One step per MZI: 45 at m = 10. Each sweeps the MZI's two arms and
    fits the raised-cosine fringe to recover both arms' ``k`` and their
    offset difference.
    """
    steps: list[DeltaStep] = []
    for k in calibration_order(mesh):
        in_port, out_port, reverse = diagonal_ports(mesh.m, k)
        routers = router_settings(mesh, k)
        for (i, j) in diagonal_mzis(mesh, k):
            steps.append(DeltaStep(
                mzi=(i, j),
                diagonal=k,
                input_port=in_port,
                output_port=out_port,
                reverse=reverse,
                routers=dict(routers),
                swept_ps=mesh.mzi_arm_ps(i, j),
            ))
    return steps


def calibration_order(mesh: ChipMesh) -> list[int]:
    """Diagonals in the order the document calibrates them.

    Main diagonal first (it is the only router-free one and so the only
    one that can be measured before anything else is known), then
    outward through the forward diagonals, then the reversed ones.
    """
    m = mesh.m
    k_main = main_diagonal(m)
    forward = [k for k in range(k_main + 1, n_diagonals(m))]
    backward = [k for k in range(k_main - 1, -1, -1)]
    return [k_main] + forward + backward
