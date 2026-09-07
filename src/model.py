"""Differentiable forward model: the digital twin of an imperfect linear-optical PIC.

Maps voltage configurations and input-port indices to predicted output intensity
distributions on a Clements mesh, with three trainable parameter blocks and one
frozen seed.

Learnable parameters
--------------------
    C_2_raw  (n_PS, n_PS)  Full crosstalk matrix (V**2 -> phase). Both the
                           diagonal (self-heating) and off-diagonal (inter-PS
                           crosstalk) entries update. Initialised: diagonal =
                           V-IFM seed, off-diagonals = 0.
    R_logit  (n_BS,)       Beamsplitter reflectivity logits. R = sigmoid(R_logit)
                           lies in (0, 1). Initialised to 0 -> R = 0.5.
    T_logit  (m,)          Output transmission logits. T_out = sigmoid(T_logit)
                           lies in (0, 1]. Initialised to 6.0 -> T_out ~ 0.998.

Frozen
------
    c_0      (n_PS,)       Passive phase offsets, registered as a buffer.
                           Refined separately in the phi-IFM stage (CLAUDE.md
                           pitfall 2).
    T_in     -- not in the model. Multiplying column i of U by sqrt(T_in[i])
                scales the entire column by a real constant, which cancels in
                the L2 normalisation in forward(). Its gradient is therefore
                mathematically zero (CLAUDE.md pitfall 1). T_in is recovered
                in the separate ITM stage.

Forward model (CLAUDE.md §4)
----------------------------
    phi    = x @ C_2.T + c_0
    U_0    = product of per-layer matrices, propagation order (input -> output)
    U      = diag(sqrt(T_out)) @ U_0
    p_hat  = |U[:, input_port]|**2 / sum(|U[:, input_port]|**2)

Drive variable: electric power
------------------------------
``x`` is the ELECTRIC POWER dissipated in each heater, in watts -- the
Prakash calibration document's ``x``, and the quantity its phase law
``theta = k*x + b`` is linear in. So ``diag(C_2)`` is that document's
per-shifter ``k`` (rad/W) and ``c_0`` is its ``b`` (rad); the off-diagonal
of ``C_2`` is the thermal crosstalk the document does not model and the
ML stage exists to recover.

This replaces the earlier voltage drive (``phi = V**2 @ C_2.T + c_0``,
after Fyrillas et al.). Two reasons: the chip is specified by a power
budget rather than a voltage one, and a heater's resistance rises as it
warms, so power is not simply proportional to V**2 anyway. The
current-to-power conversion, which is where that nonlinearity lives, is
handled upstream by ``src.power_lookup`` and is deliberately outside the
autograd graph.

The document's theta is this module's ``phi``: same quantity, different
letter.
"""

from __future__ import annotations

import torch
from torch import nn

from src.chip_mesh import BeamsplitterLayer, ChipMesh, PhaseShifterLayer


_REAL_TO_COMPLEX: dict[torch.dtype, torch.dtype] = {
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


class DigitalTwin(nn.Module):
    """Differentiable forward model of an imperfect linear-optical PIC."""

    def __init__(
        self,
        mesh: ChipMesh,
        c_0_seed: torch.Tensor,
        c2_diag_seed: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if c_0_seed.shape != (mesh.n_PS,):
            raise ValueError(
                f"c_0_seed.shape must be ({mesh.n_PS},), got {tuple(c_0_seed.shape)}"
            )
        if c2_diag_seed.shape != (mesh.n_PS,):
            raise ValueError(
                f"c2_diag_seed.shape must be ({mesh.n_PS},), got "
                f"{tuple(c2_diag_seed.shape)}"
            )
        if dtype not in _REAL_TO_COMPLEX:
            raise ValueError(f"dtype must be float32 or float64, got {dtype}")

        self.mesh = mesh
        self.m = mesh.m
        self.n_PS = mesh.n_PS
        self.n_BS = mesh.n_BS
        self._real_dtype = dtype
        self._complex_dtype = _REAL_TO_COMPLEX[dtype]

        self.register_buffer("c_0", c_0_seed.detach().to(dtype).clone())

        c2 = torch.zeros(self.n_PS, self.n_PS, dtype=dtype)
        c2.diagonal().copy_(c2_diag_seed.to(dtype))
        self.C_2_raw = nn.Parameter(c2)

        self.R_logit = nn.Parameter(torch.zeros(self.n_BS, dtype=dtype))
        self.T_logit = nn.Parameter(torch.full((self.m,), 6.0, dtype=dtype))

        # Precompute per-layer index tensors so _build_U pays no
        # ``torch.tensor()`` overhead per forward call. Registered as
        # non-persistent buffers: they ride along on ``.to(device)`` and
        # don't pollute the state_dict (they're deterministic from the
        # mesh, recomputed in every fresh __init__).
        self._layer_kinds: list[str] = []
        for layer_idx, layer in enumerate(mesh.layers):
            if isinstance(layer, PhaseShifterLayer):
                ks = sorted(layer.shifter_indices.keys())
                ps_idxs = [layer.shifter_indices[k] for k in ks]
                self.register_buffer(
                    f"_l{layer_idx}_ks",
                    torch.tensor(ks, dtype=torch.long),
                    persistent=False,
                )
                self.register_buffer(
                    f"_l{layer_idx}_ps_idxs",
                    torch.tensor(ps_idxs, dtype=torch.long),
                    persistent=False,
                )
                self._layer_kinds.append("PS")
            elif isinstance(layer, BeamsplitterLayer):
                pairs = sorted(layer.bs_pairs.keys())
                self.register_buffer(
                    f"_l{layer_idx}_ks_top",
                    torch.tensor(
                        [p[0] for p in pairs], dtype=torch.long
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    f"_l{layer_idx}_ks_bot",
                    torch.tensor(
                        [p[1] for p in pairs], dtype=torch.long
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    f"_l{layer_idx}_bs_idxs",
                    torch.tensor(
                        [layer.bs_pairs[p] for p in pairs],
                        dtype=torch.long,
                    ),
                    persistent=False,
                )
                self._layer_kinds.append("BS")
            else:
                raise TypeError(
                    f"layer {layer_idx}: unknown layer type {type(layer)!r}"
                )

    @property
    def C_2(self) -> torch.Tensor:
        return self.C_2_raw

    @property
    def R(self) -> torch.Tensor:
        return torch.sigmoid(self.R_logit)

    @property
    def T_out(self) -> torch.Tensor:
        return torch.sigmoid(self.T_logit)

    def _build_U(self, phi: torch.Tensor) -> torch.Tensor:
        """Build the lossless chip matrix U_0 from a phase vector.

        Accepts either a single sample or a batch:
            phi of shape (n_PS,)        -> returns U_0 of shape (m, m)
            phi of shape (batch, n_PS)  -> returns U_0 of shape (batch, m, m)

        T_out is NOT applied here -- callers that need it should left-
        multiply by diag(sqrt(self.T_out)) (or, equivalently, scale rows).

        Implementation is vectorised over both the batch dimension and the
        elements within each layer:
          - A PhaseShifterLayer is a diagonal in waveguide space, so
            applying it to U is one elementwise row-scale, not a matmul.
          - A BeamsplitterLayer is independent of phi, so the layer
            matrix L is built once and broadcast against the batched U.
          - Within a layer, every quadrant of every BS (or every PS)
            collapses to a single fancy-indexed assignment.
        """
        single_sample = (phi.dim() == 1)
        if single_sample:
            phi = phi.unsqueeze(0)
        if phi.dim() != 2 or phi.shape[-1] != self.n_PS:
            raise ValueError(
                f"phi must have shape (n_PS,) or (batch, n_PS) with "
                f"n_PS={self.n_PS}, got {tuple(phi.shape)}"
            )

        device = self.c_0.device
        phi = phi.to(device=device, dtype=self._real_dtype)
        batch = phi.shape[0]
        m = self.m
        cdtype = self._complex_dtype

        R = self.R
        sqrt_R = torch.sqrt(R).to(cdtype)
        sqrt_1mR = torch.sqrt(1.0 - R).to(cdtype)
        i_sqrt_1mR = 1j * sqrt_1mR

        # Identity, broadcast across the batch.
        U = (
            torch.eye(m, dtype=cdtype, device=device)
            .unsqueeze(0)
            .expand(batch, m, m)
            .contiguous()
        )

        for layer_idx, kind in enumerate(self._layer_kinds):
            if kind == "PS":
                ks_t = getattr(self, f"_l{layer_idx}_ks")
                ps_idxs_t = getattr(self, f"_l{layer_idx}_ps_idxs")
                # Diagonal L in waveguide space: 1 everywhere, exp(i*phi)
                # at the PS positions for each batch element.
                diag = torch.ones(batch, m, dtype=cdtype, device=device)
                diag[:, ks_t] = torch.exp(1j * phi[:, ps_idxs_t])
                # L @ U with L diagonal = elementwise row-scale.
                U = diag.unsqueeze(-1) * U
            else:  # "BS"
                ks_top = getattr(self, f"_l{layer_idx}_ks_top")
                ks_bot = getattr(self, f"_l{layer_idx}_ks_bot")
                bs_idxs = getattr(self, f"_l{layer_idx}_bs_idxs")
                L = torch.eye(m, dtype=cdtype, device=device)
                # All quadrants of all BSs in this layer in four
                # vectorised assignments.
                L[ks_top, ks_top] = sqrt_R[bs_idxs]
                L[ks_bot, ks_bot] = sqrt_R[bs_idxs]
                L[ks_top, ks_bot] = i_sqrt_1mR[bs_idxs]
                L[ks_bot, ks_top] = i_sqrt_1mR[bs_idxs]
                # Broadcasts to (batch, m, m).
                U = L @ U

        if single_sample:
            return U.squeeze(0)
        return U

    def forward(
        self,
        x: torch.Tensor,
        input_ports: torch.Tensor,
    ) -> torch.Tensor:
        """Map heater powers and input ports to predicted output distributions.

        Args:
            x:           (batch, n_PS) real electric powers, in watts.
            input_ports: (batch,) long input-port indices in [0, m).

        Returns:
            (batch, m) real predicted distribution, each row sums to 1.
        """
        if x.shape[-1] != self.n_PS:
            raise ValueError(
                f"x last dim must be n_PS = {self.n_PS}, got {x.shape[-1]}"
            )
        if input_ports.shape[0] != x.shape[0]:
            raise ValueError(
                f"input_ports batch ({input_ports.shape[0]}) must match x batch "
                f"({x.shape[0]})"
            )

        device = self.c_0.device
        phi = x @ self.C_2.T + self.c_0                 # (batch, n_PS)
        batch = phi.shape[0]
        sqrt_T_out = torch.sqrt(self.T_out).to(self._complex_dtype)

        U_0 = self._build_U(phi)                        # (batch, m, m)
        # diag(sqrt(T_out)) @ U_0: scale every output row by sqrt(T_out[j]).
        U = sqrt_T_out.view(1, -1, 1) * U_0             # (m,) -> (1, m, 1)

        # Gather the input-port column for each sample.
        batch_idx = torch.arange(batch, device=device)
        ports = input_ports.to(device=device, dtype=torch.long)
        columns = U[batch_idx, :, ports]                # (batch, m)

        unnorm = columns.real ** 2 + columns.imag ** 2
        p_hat = unnorm / unnorm.sum(dim=-1, keepdim=True)

        max_err = (p_hat.sum(dim=-1) - 1.0).abs().max().item()
        if max_err > 1e-5:
            raise RuntimeError(
                f"Normalisation invariant violated: max |row_sum - 1| = {max_err}"
            )
        return p_hat
