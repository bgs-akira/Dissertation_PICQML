"""Chip topology: ordered atomic layers, global PS/BS indexing.

Two mesh schemes are built here. ``ChipMesh.bell(m)`` is the project
default (see CLAUDE.md §3.5); ``ChipMesh.clements(m)`` is retained as a
regression reference for the scheme-agnostic model/training code.

Indexing convention (must match V-IFM seeds and dataset)
---------------------------------------------------------
Atomic layers are ordered left-to-right (input to output). Within an atomic
layer, PSs are indexed by ascending waveguide number; BSs are indexed by
ascending top-waveguide number of the pair. Global indices are then assigned
by layer-major, top-to-bottom-within-a-layer scan, with no gaps:

    for layer in layers:                    # left to right
        for waveguide (or pair) in layer:   # top to bottom
            assign next global index

Phase-shifter roles
-------------------
Every PS is modelled identically (a single-waveguide phase), but they play
three distinct physical roles, recorded in ``ChipMesh.ps_roles``:

    "internal"     -- sits between the two beamsplitters of an MZI.
    "external"     -- sits after an MZI's second beamsplitter, on the same
                      waveguide. Clements only.
    "independent"  -- sits on a waveguide that hosts no MZI in that column.
                      Bell only: the top and bottom waveguides are idle in
                      every narrow column, and those slots carry the phases
                      that a Clements mesh would put in its external layer.

Bell compact mesh, m modes (m even) -- THE PROJECT DEFAULT
-----------------------------------------------------------
The compactified rectangular mesh of Bell & Walmsley, *Further
compactifying linear optical unitaries*, APL Photonics 6, 070804 (2021).

Each MZI carries a phase shifter on **both** internal arms instead of one
internal plus one external. That absorbs the external phase layer, so the
circuit is m columns of MZIs deep ("optical depth m") rather than needing
a separate phase column after every beamsplitter column.

- m columns, alternating "wide" (parity 0) and "narrow" (parity 1),
  starting wide and ending narrow:
    wide   columns host MZIs on pairs (0, 1), (2, 3), ..., (m-2, m-1)
    narrow columns host MZIs on pairs (1, 2), (3, 4), ..., (m-3, m-2)
- Each column expands to three atomic layers, in propagation order:
    1. BeamsplitterLayer  -- BS_a (first beamsplitter of every MZI)
    2. PhaseShifterLayer  -- both internal PSs of every MZI, PLUS, in a
                             narrow column, the two independent PSs on the
                             idle top (0) and bottom (m-1) waveguides
    3. BeamsplitterLayer  -- BS_b (second beamsplitter of every MZI)
  The independent PSs share layer 2 with the internals rather than needing
  a layer of their own: a narrow column's internal PSs occupy waveguides
  1 .. m-2, so waveguides 0 and m-1 are free and the layer stays
  collision-free.
- Counts:
    MZIs            = m * (m - 1) / 2       (m/2 wide cols * m/2 MZIs
                                             + m/2 narrow cols * (m/2 - 1))
    n_BS            = m * (m - 1)           (2 BSs per MZI)
    internal PSs    = m * (m - 1)           (2 per MZI)
    independent PSs = m                     (2 per narrow column)
    n_PS            = m ** 2                (internal + independent)
    atomic layers   = 3 * m                 (for m >= 4)

  ``n_PS = m**2`` is exactly the real-parameter count of U(m): the Bell
  mesh is universal with no phases left over. (The Clements builder below
  stops at ``m * (m - 1)``, m short of universal -- see CLAUDE.md §3.5.)

  At the project's target size m = 10: 45 MZIs, n_BS = 90, n_PS = 100
  (90 internal + 10 independent), 30 atomic layers.

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


#: Valid values of ``ChipMesh.ps_roles``. See the module docstring.
PS_ROLES = ("internal", "external", "independent")


@dataclass(frozen=True)
class PhaseShifterLayer:
    """One column of the mesh with PSs on disjoint waveguides.

    shifter_indices maps waveguide_index -> global PS index in [0, n_PS).
    Within a single PhaseShifterLayer no waveguide appears twice. In the
    Bell scheme a single layer mixes the internal PSs of that column's MZIs
    with the independent PSs on the idle boundary waveguides; they are
    still on disjoint waveguides, so the invariant holds.
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
    scheme: str
    ps_roles: dict[int, str]

    def __init__(
        self,
        m: int,
        n_PS: int,
        n_BS: int,
        layers: list[Layer],
        *,
        scheme: str = "custom",
        ps_roles: dict[int, str] | None = None,
    ) -> None:
        """Build a mesh from pre-assembled layers.

        Args:
            m:        number of modes (waveguides).
            n_PS:     total phase shifters; must equal the number of PS
                      entries across ``layers``.
            n_BS:     total beamsplitters; likewise.
            layers:   ordered input -> output.
            scheme:   provenance tag, e.g. ``"bell"`` or ``"clements"``.
                      Downstream code branches on this (``pic_graph`` only
                      supports ``"clements"``) and runners record it in
                      their JSON config.
            ps_roles: global PS index -> one of ``PS_ROLES``. When None, an
                      all-``"internal"`` map is assumed; the classmethod
                      builders always supply a real one.
        """
        self.m = m
        self.n_PS = n_PS
        self.n_BS = n_BS
        self.layers = layers
        self.scheme = scheme
        self.ps_roles = (
            dict(ps_roles)
            if ps_roles is not None
            else {i: "internal" for i in range(n_PS)}
        )

        # (mode, phase-layer-ordinal) labels -- the Prakash document's
        # (i, j). The ordinal counts PhaseShifterLayers only, so for a Bell
        # mesh (exactly one phase layer per column) it equals the column
        # index j. A Clements mesh has two phase layers per column, so the
        # ordinal is NOT the column and the public accessors below refuse
        # to run on it.
        self._label_of_ps: dict[int, tuple[int, int]] = {}
        self._ps_of_label: dict[tuple[int, int], int] = {}
        # Column ordinal -> (BS_a layer index or None, PS layer index,
        # BS_b layer index or None). Read off the actual layer sequence
        # rather than assuming a fixed stride: a column with no MZIs
        # contributes only a phase layer (this happens at m = 2, where the
        # narrow column is just the two independent shifters), so a
        # "3 * j" stride would walk off the pattern.
        self._column_layers: dict[int, tuple[int | None, int, int | None]] = {}
        ordinal = 0
        for layer_idx, layer in enumerate(layers):
            if isinstance(layer, PhaseShifterLayer):
                for wg, idx in layer.shifter_indices.items():
                    self._label_of_ps[idx] = (wg, ordinal)
                    self._ps_of_label[(wg, ordinal)] = idx
                before = (
                    layer_idx - 1
                    if layer_idx > 0
                    and isinstance(layers[layer_idx - 1], BeamsplitterLayer)
                    else None
                )
                after = (
                    layer_idx + 1
                    if layer_idx + 1 < len(layers)
                    and isinstance(layers[layer_idx + 1], BeamsplitterLayer)
                    else None
                )
                # A column's two couplers act on IDENTICAL waveguide pairs.
                # Demanding that disambiguates the neighbours: without it,
                # a column that hosts no MZIs would adopt the previous
                # column's BS_b as its own BS_a and invent MZIs that do not
                # exist. Adjacent columns have opposite brick parity, so
                # their pair sets never coincide.
                if before is None or after is None or (
                    layers[before].bs_pairs.keys()
                    != layers[after].bs_pairs.keys()
                ):
                    before = after = None
                self._column_layers[ordinal] = (before, layer_idx, after)
                ordinal += 1

        self.validate()

    def validate(self) -> None:
        """Check structural invariants. Raises ValueError on violation.

        Scheme-specific count formulas are NOT checked here -- ``n_PS`` and
        ``n_BS`` differ between Bell and Clements. Each classmethod builder
        asserts its own counts; this method only enforces what must hold
        for any mesh: in-range waveguides, no within-layer collisions,
        adjacent BS pairs, and contiguous gap-free global indexing.
        """
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

        if sorted(self.ps_roles.keys()) != list(range(self.n_PS)):
            raise ValueError(
                f"ps_roles keys are not the contiguous range [0, {self.n_PS})"
            )
        bad = {r for r in self.ps_roles.values() if r not in PS_ROLES}
        if bad:
            raise ValueError(
                f"ps_roles contains unknown role(s) {sorted(bad)}; "
                f"valid roles are {PS_ROLES}"
            )

    def ps_indices_with_role(self, role: str) -> list[int]:
        """Sorted global PS indices whose role is ``role``.

        Lets the evaluation and plotting code report recovered phases
        grouped the way the mesh figure draws them (internal vs
        independent), and lets phi-IFM diagnostics separate the two.
        """
        if role not in PS_ROLES:
            raise ValueError(f"role must be one of {PS_ROLES}, got {role!r}")
        return sorted(i for i, r in self.ps_roles.items() if r == role)

    # ------------------------------------------------------------------
    # Prakash (i, j) labelling -- Bell scheme only
    #
    # The calibration document labels every phase shifter by a two-tuple
    # (i, j): i the mode index, j the layer index, both in [0, m). An MZI
    # is named by its UPPER internal shifter, so an MZI label always has
    # i + j even, and its lower arm is (i + 1, j). Its diagonal index is
    # k = (i + j) / 2. These accessors are the bridge between that
    # notation and this codebase's flat global indices.
    # ------------------------------------------------------------------

    def _require_bell(self, what: str) -> None:
        if self.scheme != "bell":
            raise ValueError(
                f"{what} uses the Prakash (i, j) labelling, which is "
                f"defined for the Bell scheme only (one phase layer per "
                f"column); this mesh has scheme={self.scheme!r}."
            )

    def ps_label(self, ps_index: int) -> tuple[int, int]:
        """Prakash label ``(i, j)`` = (mode, layer) for a global PS index."""
        self._require_bell("ps_label")
        if ps_index not in self._label_of_ps:
            raise ValueError(f"PS index {ps_index} not in [0, {self.n_PS})")
        return self._label_of_ps[ps_index]

    def ps_index(self, i: int, j: int) -> int:
        """Global PS index for Prakash label ``(i, j)`` = (mode, layer)."""
        self._require_bell("ps_index")
        if (i, j) not in self._ps_of_label:
            raise ValueError(
                f"no phase shifter at label (i={i}, j={j}) on this mesh"
            )
        return self._ps_of_label[(i, j)]

    @staticmethod
    def mzi_diagonal(i: int, j: int) -> int:
        """Diagonal index ``k = (i + j) / 2`` of the MZI labelled (i, j)."""
        if (i + j) % 2 != 0:
            raise ValueError(
                f"MZI label (i={i}, j={j}) must have i + j even"
            )
        return (i + j) // 2

    def mzi_labels(self) -> list[tuple[int, int]]:
        """Every MZI label ``(i, j)``, sorted. i is the upper mode."""
        self._require_bell("mzi_labels")
        out: list[tuple[int, int]] = []
        for j, (bs_a_idx, _, _) in self._column_layers.items():
            if bs_a_idx is None:
                continue          # column with no MZIs (only at m = 2)
            bs_a = self.layers[bs_a_idx]
            assert isinstance(bs_a, BeamsplitterLayer)
            out.extend((top, j) for (top, _bot) in bs_a.bs_pairs)
        return sorted(out)

    def mzi_arm_ps(self, i: int, j: int) -> tuple[int, int]:
        """Global PS indices of the MZI's ``(upper arm, lower arm)``.

        The document's theta(i,j) and theta(i+1,j); the pair whose
        half-difference is delta and half-sum is Sigma.
        """
        self._require_bell("mzi_arm_ps")
        return self.ps_index(i, j), self.ps_index(i + 1, j)

    def mzi_bs(self, i: int, j: int) -> tuple[int, int]:
        """Global BS indices of the MZI's ``(first, second)`` couplers.

        In the document's notation these carry the deviations alpha (first,
        acting on the input) and beta (second). Here they are two entries
        of the ``R`` vector; ``R = cos**2(pi/4 + alpha)``.
        """
        self._require_bell("mzi_bs")
        if j not in self._column_layers:
            raise ValueError(f"no column at layer index j={j}")
        bs_a_idx, _, bs_b_idx = self._column_layers[j]
        if bs_a_idx is None or bs_b_idx is None:
            raise ValueError(f"column j={j} hosts no MZIs")
        bs_a = self.layers[bs_a_idx]
        bs_b = self.layers[bs_b_idx]
        assert isinstance(bs_a, BeamsplitterLayer)
        assert isinstance(bs_b, BeamsplitterLayer)
        pair = (i, i + 1)
        if pair not in bs_a.bs_pairs:
            raise ValueError(f"no MZI at label (i={i}, j={j})")
        return bs_a.bs_pairs[pair], bs_b.bs_pairs[pair]

    def __repr__(self) -> str:
        return (
            f"ChipMesh(scheme={self.scheme!r}, m={self.m}, n_BS={self.n_BS}, "
            f"n_PS={self.n_PS}, n_layers={len(self.layers)})"
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
            `P<idx>` -- internal or external phase shifter on this waveguide.
            `I<idx>` -- independent phase shifter (Bell scheme) on this
                        waveguide.
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
                    prefix = (
                        "I" if self.ps_roles.get(idx) == "independent" else "P"
                    )
                    cells[wg] = f"{prefix}{idx:0{cell_width - 1}d}"
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

    @staticmethod
    def _column_pair_tops(m: int, column: int) -> list[int]:
        """Top waveguides of the MZIs in ``column`` of a rectangular mesh.

        Even columns are "wide" -- pairs (0,1), (2,3), ..., (m-2, m-1).
        Odd columns are "narrow" -- pairs (1,2), (3,4), ..., (m-3, m-2).
        Shared by both scheme builders so the brick pattern is defined once.
        """
        is_wide = column % 2 == 0
        start = 0 if is_wide else 1
        n_pairs = m // 2 if is_wide else m // 2 - 1
        return [start + 2 * i for i in range(n_pairs)]

    @classmethod
    def bell(cls, m: int) -> "ChipMesh":
        """Construct a Bell compact rectangular mesh on m modes.

        The project default. See the module docstring for the layer
        structure: m columns, each expanding to (BS_a, phase layer, BS_b),
        with two independent PSs riding in the phase layer of every narrow
        column. Yields ``n_PS = m ** 2`` and ``n_BS = m * (m - 1)``.

        Written for general even m, but only exercised at the project's
        target size m = 10 (CLAUDE.md §3.5).
        """
        if m < 2:
            raise ValueError(f"m must be >= 2, got {m}")
        if m % 2 != 0:
            raise ValueError(f"m must be even, got {m}")

        layers: list[Layer] = []
        ps_roles: dict[int, str] = {}
        ps_counter = 0
        bs_counter = 0

        for column in range(m):
            is_wide = column % 2 == 0
            pair_tops = cls._column_pair_tops(m, column)

            # BS_a -- first beamsplitter of every MZI in the column.
            if pair_tops:
                bs_a: dict[tuple[int, int], int] = {}
                for k in pair_tops:
                    bs_a[(k, k + 1)] = bs_counter
                    bs_counter += 1
                layers.append(BeamsplitterLayer(bs_pairs=bs_a))

            # Phase layer -- an internal PS on BOTH arms of every MZI (this
            # is what distinguishes Bell from Clements), plus the two
            # independent PSs on the idle boundary waveguides of a narrow
            # column. Assigned in ascending waveguide order to honour the
            # module-level indexing convention.
            roles_by_waveguide: dict[int, str] = {}
            for k in pair_tops:
                roles_by_waveguide[k] = "internal"
                roles_by_waveguide[k + 1] = "internal"
            if not is_wide:
                roles_by_waveguide[0] = "independent"
                roles_by_waveguide[m - 1] = "independent"

            ps_layer: dict[int, int] = {}
            for wg in sorted(roles_by_waveguide):
                ps_layer[wg] = ps_counter
                ps_roles[ps_counter] = roles_by_waveguide[wg]
                ps_counter += 1
            layers.append(PhaseShifterLayer(shifter_indices=ps_layer))

            # BS_b -- second beamsplitter of every MZI in the column.
            if pair_tops:
                bs_b: dict[tuple[int, int], int] = {}
                for k in pair_tops:
                    bs_b[(k, k + 1)] = bs_counter
                    bs_counter += 1
                layers.append(BeamsplitterLayer(bs_pairs=bs_b))

        mesh = cls(
            m=m,
            n_PS=ps_counter,
            n_BS=bs_counter,
            layers=layers,
            scheme="bell",
            ps_roles=ps_roles,
        )
        if mesh.n_BS != m * (m - 1):
            raise AssertionError(
                f"bell({m}): n_BS = {mesh.n_BS}, expected {m * (m - 1)}"
            )
        if mesh.n_PS != m * m:
            raise AssertionError(
                f"bell({m}): n_PS = {mesh.n_PS}, expected {m * m}"
            )
        return mesh

    @classmethod
    def clements(cls, m: int) -> "ChipMesh":
        """Construct a Clements rectangular mesh on m modes.

        Retained as a regression reference for the scheme-agnostic model
        and training code; the project default is ``bell``. See the module
        docstring: each MZI row contributes four atomic layers (BS_a,
        PS_int, BS_b, PS_ext).
        """
        if m < 2:
            raise ValueError(f"m must be >= 2, got {m}")
        if m % 2 != 0:
            raise ValueError(f"m must be even, got {m}")

        layers: list[Layer] = []
        ps_roles: dict[int, str] = {}
        ps_counter = 0
        bs_counter = 0

        for mzi_row in range(m):
            pair_tops = cls._column_pair_tops(m, mzi_row)

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
                ps_roles[ps_counter] = "internal"
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
                ps_roles[ps_counter] = "external"
                ps_counter += 1
            layers.append(PhaseShifterLayer(shifter_indices=ps_ext))

        mesh = cls(
            m=m,
            n_PS=ps_counter,
            n_BS=bs_counter,
            layers=layers,
            scheme="clements",
            ps_roles=ps_roles,
        )
        if mesh.n_PS != m * (m - 1):
            raise AssertionError(
                f"clements({m}): n_PS = {mesh.n_PS}, expected {m * (m - 1)}"
            )
        return mesh


#: Mesh schemes ``build_mesh`` can construct. First entry is the default.
MESH_SCHEMES = ("bell", "clements")


def build_mesh(m: int, scheme: str = "bell") -> ChipMesh:
    """Construct a mesh by scheme name.

    Single dispatch point so every script, runner and test names the
    scheme as a string (which is also what lands in the saved JSON config)
    rather than reaching for a classmethod directly.

    Args:
        m:      number of modes (even).
        scheme: one of ``MESH_SCHEMES``. Defaults to ``"bell"``, the
                project default.
    """
    if scheme == "bell":
        return ChipMesh.bell(m)
    if scheme == "clements":
        return ChipMesh.clements(m)
    raise ValueError(
        f"scheme must be one of {MESH_SCHEMES}, got {scheme!r}"
    )
