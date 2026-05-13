"""Dataset wrapper for (V, port, p) triples."""

from __future__ import annotations

import torch


class PICDataset(torch.utils.data.Dataset):
    """Holds (V, port, p) triples.

    V     : (N, n_PS) real, applied voltages.
    port  : (N,) int,  input port index in [0, m).
    p     : (N, m) real, measured normalised output distribution (sums to 1).
    """

    def __init__(self, V: torch.Tensor, port: torch.Tensor, p: torch.Tensor) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
