"""Chip topology: ordered layers, global PS/BS indexing.

Indexing convention (must match V-IFM seeds and dataset):
    Left-to-right column scan, top-to-bottom within each column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class PhaseShifterLayer:
    """One column of the mesh with PSs on disjoint waveguides.

    shifter_indices maps waveguide_index -> global PS index in [0, n_PS).
    """

    shifter_indices: dict[int, int]


@dataclass
class BeamsplitterLayer:
    """One column with BSs on disjoint waveguide pairs.

    bs_pairs maps (waveguide_top, waveguide_bottom) -> global BS index in [0, n_BS).
    """

    bs_pairs: dict[tuple[int, int], int]


Layer = Union[PhaseShifterLayer, BeamsplitterLayer]


class ChipMesh:
    """Ordered list of layers (input to output) plus global counts."""

    m: int
    n_PS: int
    n_BS: int
    layers: list[Layer]

    def __init__(self, m: int, n_PS: int, n_BS: int, layers: list[Layer]) -> None:
        raise NotImplementedError

    @classmethod
    def clements(cls, m: int) -> "ChipMesh":
        """Construct a Clements rectangular mesh on m modes.

        See CLAUDE.md §3.5 for the brick-pattern column structure and
        the MZI unit-cell decomposition.
        """
        raise NotImplementedError
