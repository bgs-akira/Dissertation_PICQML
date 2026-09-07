"""Chip topology: ordered atomic layers, global PS/BS indexing.

Indexing convention (must match V-IFM seeds and dataset)
---------------------------------------------------------
Atomic layers are ordered left-to-right (input to output). Within an atomic
layer, PSs are indexed by ascending waveguide number; BSs are indexed by
ascending top-waveguide number of the pair. Global indices are then assigned
by layer-major, top-to-bottom-within-a-layer scan, with no gaps:

    for layer in layers:                    # left to right
        for waveguide (or pair) in layer:   # top to bottom
            assign next global index

Clements rectangular mesh, m modes (m even, m >= 2)
---------------------------------------------------
- m "MZI rows", alternating "wide" (parity 0) and "narrow" (parity 1):
    wide   rows host MZIs on pairs (0, 1), (2, 3), ..., (m-2, m-1)
    narrow rows host MZIs on pairs (1, 2), (3, 4), ..., (m-3, m-2)
- Each MZI unit cell on waveguides (k, k+1) expands to four atomic layers,
  in propagation order:
    1. BeamsplitterLayer  -- BS_a (first beamsplitter)
    2. PhaseShifterLayer  -- internal PS, on waveguide k (upper arm)
    3. BeamsplitterLayer  -- BS_b (second beamsplitter)
    4. PhaseShifterLayer  -- external PS, on waveguide k (upper arm)
- Phase shifters come in two physical roles, both modelled identically as
  single-waveguide phase shifts:
    - Internal PS: sits between BS_a and BS_b of an MZI. One per MZI, on
      the upper arm.
    - External PS: sits after BS_b of an MZI, before the next MZI on that
      waveguide (or before chip output). One per MZI, on the upper arm.
      Contributes a passive phase to the MZI's output with no following
      beamsplitter.
- An empty MZI row contributes no atomic layers. This only happens for
  m == 2 (the narrow row has m/2 - 1 = 0 MZIs and is skipped); for m >= 4
  every MZI row is non-empty, and the layer count is exactly 4 * m.
- Counts:
    n_BS = m * (m - 1)             (2 BSs per MZI, m(m-1)/2 MZIs)
    n_PS = m * (m - 1)             (1 internal + 1 external per MZI)
    total atomic layers = 4 * m    (for m >= 4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class PhaseShifterLayer:
    """One column of the mesh with PSs on disjoint waveguides.

    shifter_indices maps waveguide_index -> global PS index in [0, n_PS).
    Within a single PhaseShifterLayer no waveguide appears twice (an internal
    PS and an external PS on the same waveguide always live in different
    layers).
    """

    shifter_indices: dict[int, int]


@dataclass(frozen=True)
class BeamsplitterLayer:
    """One column with BSs on disjoint, adjacent waveguide pairs.

    bs_pairs maps (waveguide_top, waveguide_bottom) -> global BS index in
    [0, n_BS). Every pair satisfies waveguide_bottom == waveguide_top + 1.
    """

    bs_pairs: dict[tuple[int, int], int]


Layer = Union[PhaseShifterLayer, BeamsplitterLayer]


class ChipMesh:
    """Ordered list of atomic layers (input -> output) plus global counts."""

    m: int
    n_PS: int
    n_BS: int
    layers: list[Layer]

    def __init__(
        self,
        m: int,
        n_PS: int,
        n_BS: int,
        layers: list[Layer],
    ) -> None:
        self.m = m
        self.n_PS = n_PS
        self.n_BS = n_BS
        self.layers = layers
        self.validate()

    def validate(self) -> None:
        """Check structural invariants. Raises ValueError on violation."""
        if self.n_PS != self.m * (self.m - 1):
            raise ValueError(
                f"n_PS = {self.n_PS} does not match m * (m - 1) "
                f"= {self.m * (self.m - 1)}"
            )

        ps_indices: list[int] = []
        bs_indices: list[int] = []

        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, PhaseShifterLayer):
                waveguides = list(layer.shifter_indices.keys())
                if len(waveguides) == 0:
                    raise ValueError(f"layer {layer_idx}: empty phase-shifter layer")
                if any(not (0 <= w < self.m) for w in waveguides):
                    raise ValueError(
                        f"layer {layer_idx}: PS waveguide out of range [0, {self.m})"
                    )
                if len(set(waveguides)) != len(waveguides):
                    raise ValueError(
                        f"layer {layer_idx}: waveguide repeated within a "
                        "phase-shifter layer"
                    )
                ps_indices.extend(layer.shifter_indices.values())

            elif isinstance(layer, BeamsplitterLayer):
                if len(layer.bs_pairs) == 0:
                    raise ValueError(f"layer {layer_idx}: empty beamsplitter layer")
                used: set[int] = set()
                for k_top, k_bot in layer.bs_pairs.keys():
                    if k_bot != k_top + 1:
                        raise ValueError(
                            f"layer {layer_idx}: BS pair ({k_top}, {k_bot}) is not "
                            "adjacent (must be (k, k+1))"
                        )
                    if not (0 <= k_top and k_bot < self.m):
                        raise ValueError(
                            f"layer {layer_idx}: BS pair ({k_top}, {k_bot}) out of "
                            f"waveguide range [0, {self.m})"
                        )
                    if k_top in used or k_bot in used:
                        raise ValueError(
                            f"layer {layer_idx}: waveguide repeated within a "
                            "beamsplitter layer"
                        )
                    used.add(k_top)
                    used.add(k_bot)
                bs_indices.extend(layer.bs_pairs.values())

            else:
                raise TypeError(f"layer {layer_idx}: unknown layer type {type(layer)!r}")

        if sorted(ps_indices) != list(range(self.n_PS)):
            raise ValueError(
                f"PS global indices are not the contiguous range [0, {self.n_PS})"
            )
        if sorted(bs_indices) != list(range(self.n_BS)):
            raise ValueError(
                f"BS global indices are not the contiguous range [0, {self.n_BS})"
            )

    def __repr__(self) -> str:
        return (
            f"ChipMesh(m={self.m}, n_BS={self.n_BS}, n_PS={self.n_PS}, "
            f"n_layers={len(self.layers)})"
        )

    def to_ascii(self, *, cell_width: int | None = None) -> str:
        """ASCII visualization of the mesh.

        One row per waveguide, one column per atomic layer. The first line is
        a header (`L<idx>`) naming each layer; subsequent lines are waveguides
        labelled by their mode index. Cell contents:
            `B<idx>` -- beamsplitter on this pair. The same label appears on
                        both the top and bottom waveguides of the pair, so a
                        column with two identical `B<idx>` cells one above
                        the other is one beamsplitter (not two).
            `P<idx>` -- phase shifter on this waveguide.
            dashes   -- passthrough.
        `<idx>` is the global BS or PS index used elsewhere in the codebase.

        Args:
            cell_width: width of each column in characters (>= 3). When
                None, picks the smallest width that fits the largest index.
        """
        if cell_width is None:
            max_idx = max(self.n_BS, self.n_PS, len(self.layers)) - 1
            cell_width = max(3, 1 + len(str(max(0, max_idx))))
        if cell_width < 3:
            raise ValueError(f"cell_width must be >= 3, got {cell_width}")

        fill = "-" * cell_width
        label_width = len(str(self.m - 1))
        indent = " " * (label_width + 1)

        rows: list[list[str]] = [[] for _ in range(self.m)]
        for layer in self.layers:
            cells = [fill] * self.m
            if isinstance(layer, PhaseShifterLayer):
                for wg, idx in layer.shifter_indices.items():
                    cells[wg] = f"P{idx:0{cell_width - 1}d}"
            elif isinstance(layer, BeamsplitterLayer):
                for (k_top, k_bot), idx in layer.bs_pairs.items():
                    label = f"B{idx:0{cell_width - 1}d}"
                    cells[k_top] = label
                    cells[k_bot] = label
            for wg in range(self.m):
                rows[wg].append(cells[wg])

        header = indent + " ".join(
            f"L{i:0{cell_width - 1}d}" for i in range(len(self.layers))
        )
        body = [
            f"{wg:>{label_width}d} " + " ".join(row)
            for wg, row in enumerate(rows)
        ]
        return "\n".join([header] + body)

    @classmethod
    def clements(cls, m: int) -> "ChipMesh":
        """Construct a Clements rectangular mesh on m modes.

        See the module docstring for the layer structure: each MZI row
        contributes four atomic layers (BS_a, PS_int, BS_b, PS_ext).
        """
        if m < 2:
            raise ValueError(f"m must be >= 2, got {m}")
        if m % 2 != 0:
            raise ValueError(f"m must be even, got {m}")

        layers: list[Layer] = []
        ps_counter = 0
        bs_counter = 0

        for mzi_row in range(m):
            is_wide = mzi_row % 2 == 0
            start = 0 if is_wide else 1
            n_pairs = m // 2 if is_wide else m // 2 - 1
            pair_tops = [start + 2 * i for i in range(n_pairs)]

            if not pair_tops:
                continue

            bs_a: dict[tuple[int, int], int] = {}
            for k in pair_tops:
                bs_a[(k, k + 1)] = bs_counter
                bs_counter += 1
            layers.append(BeamsplitterLayer(bs_pairs=bs_a))

            ps_int: dict[int, int] = {}
            for k in pair_tops:
                ps_int[k] = ps_counter
                ps_counter += 1
            layers.append(PhaseShifterLayer(shifter_indices=ps_int))

            bs_b: dict[tuple[int, int], int] = {}
            for k in pair_tops:
                bs_b[(k, k + 1)] = bs_counter
                bs_counter += 1
            layers.append(BeamsplitterLayer(bs_pairs=bs_b))

            ps_ext: dict[int, int] = {}
            for k in pair_tops:
                ps_ext[k] = ps_counter
                ps_counter += 1
            layers.append(PhaseShifterLayer(shifter_indices=ps_ext))

        return cls(m=m, n_PS=ps_counter, n_BS=bs_counter, layers=layers)
