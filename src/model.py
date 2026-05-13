"""Differentiable forward model (the digital twin)."""

from __future__ import annotations

import torch
from torch import nn

from .chip_mesh import ChipMesh


class DigitalTwin(nn.Module):
    """Differentiable forward model of an imperfect linear-optical PIC.

    Parameters (in the optimiser sense):
        C_2_raw   : (n_PS, n_PS) free, mapped directly to C_2.
        R_logit   : (n_BS,) free, mapped to R via sigmoid (keeps R in (0, 1)).
        T_logit   : (m,)    free, mapped to T_out via sigmoid (keeps T_out in (0, 1]).

    Buffers (not optimised):
        c_0       : (n_PS,) frozen seed from V-IFM.

    Loss-reduction convention assumed elsewhere: mean over batch, sum over modes.
    See CLAUDE.md §5 and §10.7.
    """

    def __init__(
        self,
        mesh: ChipMesh,
        c_0_seed: torch.Tensor,
        c2_diag_seed: torch.Tensor,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    @property
    def C_2(self) -> torch.Tensor:
        """(n_PS, n_PS) crosstalk matrix."""
        raise NotImplementedError

    @property
    def R(self) -> torch.Tensor:
        """(n_BS,) beamsplitter reflectivities in (0, 1)."""
        raise NotImplementedError

    @property
    def T_out(self) -> torch.Tensor:
        """(m,) output transmissions in (0, 1]."""
        raise NotImplementedError

    def forward(
        self,
        V: torch.Tensor,
        input_ports: torch.Tensor,
    ) -> torch.Tensor:
        """Map voltages and input ports to predicted output distributions.

        Args:
            V           : (batch, n_PS) real, applied voltages.
            input_ports : (batch,) long, input port indices in [0, m).

        Returns:
            (batch, m) real, normalised so the last axis sums to 1.
        """
        raise NotImplementedError
