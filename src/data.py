"""Dataset wrapper and data I/O for (x, port, p) triples.

``x`` is the ELECTRIC POWER applied to each heater, in watts -- the drive
variable of the forward model (see ``src.model``). The instrument sets a
current; ``src.power_lookup`` converts. Storing power rather than current
keeps the dataset in the units the phase law is linear in.

Disk layout (see data/README.md):
    x.npy             (N, n_PS) float64  applied heater powers, watts
    port.npy          (N,)      int64    input port index in [0, m)
    p.npy             (N, m)    float64  measured output distribution (sums to 1)
    c_0_seed.npy      (n_PS,)   float64  calibration seed for c_0 (b, rad)
    c2_diag_seed.npy  (n_PS,)   float64  calibration seed for diag(C_2) (k, rad/W)

Synthetic data goes through make_synthetic_dataset(): a ground-truth
DigitalTwin is evaluated on uniformly random powers and input ports, and
the resulting p_hat is taken as the "measured" distribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.model import DigitalTwin
from src.power_lookup import max_power


#: Default top of the synthetic power sweep, watts. The m = 10 chip's
#: 100 mA / 15 V budget over a ~140 ohm heater; see src.power_lookup.
X_MAX = max_power()


class PICDataset(Dataset):
    """Holds (x, port, p) triples.

    Args:
        x    : (N, n_PS) real, applied heater powers in watts.
        port : (N,) int, input-port index in [0, m).
        p    : (N, m) real, normalised output distribution; each row sums to 1.
    """

    def __init__(
        self,
        x: torch.Tensor,
        port: torch.Tensor,
        p: torch.Tensor,
    ) -> None:
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D (N, n_PS), got shape {tuple(x.shape)}")
        if port.dim() != 1:
            raise ValueError(f"port must be 1-D (N,), got shape {tuple(port.shape)}")
        if p.dim() != 2:
            raise ValueError(f"p must be 2-D (N, m), got shape {tuple(p.shape)}")
        N = x.shape[0]
        if port.shape[0] != N or p.shape[0] != N:
            raise ValueError(
                f"x, port, p must share batch dim: got {N}, "
                f"{port.shape[0]}, {p.shape[0]}"
            )
        self.x = x
        self.port = port
        self.p = p

    @property
    def n_PS(self) -> int:
        return self.x.shape[1]

    @property
    def m(self) -> int:
        return self.p.shape[1]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.port[idx], self.p[idx]


def load_dataset(data_dir: str | Path) -> PICDataset:
    """Read x.npy, port.npy, p.npy from data_dir and return a PICDataset."""
    data_dir = Path(data_dir)
    x = torch.from_numpy(np.load(data_dir / "x.npy")).to(torch.float64)
    port = torch.from_numpy(np.load(data_dir / "port.npy")).to(torch.long)
    p = torch.from_numpy(np.load(data_dir / "p.npy")).to(torch.float64)
    return PICDataset(x, port, p)


def save_dataset(dataset: PICDataset, data_dir: str | Path) -> None:
    """Write a PICDataset to x.npy / port.npy / p.npy under data_dir.

    Creates the directory if it does not exist.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "x.npy", dataset.x.detach().cpu().numpy())
    np.save(data_dir / "port.npy", dataset.port.detach().cpu().numpy())
    np.save(data_dir / "p.npy", dataset.p.detach().cpu().numpy())


def load_seeds(data_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read c_0_seed.npy and c2_diag_seed.npy from data_dir.

    Returns:
        (c_0_seed, c2_diag_seed) as float64 1-D tensors, ready to pass to
        DigitalTwin.__init__. In the calibration document's notation these
        are ``b`` (rad) and ``k`` (rad/W).
    """
    data_dir = Path(data_dir)
    c_0 = torch.from_numpy(np.load(data_dir / "c_0_seed.npy")).to(torch.float64)
    c2_diag = torch.from_numpy(np.load(data_dir / "c2_diag_seed.npy")).to(
        torch.float64
    )
    return c_0, c2_diag


def train_test_split(
    dataset: PICDataset,
    *,
    test_frac: float = 0.2,
    seed: int | None = None,
) -> tuple[PICDataset, PICDataset]:
    """Split a PICDataset into disjoint train and test parts.

    Args:
        dataset:   the full dataset.
        test_frac: fraction of samples used for testing.
        seed:      RNG seed for reproducible splits. None = non-deterministic.

    Returns:
        (train_dataset, test_dataset)
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    N = len(dataset)
    n_test = int(round(N * test_frac))
    if n_test < 1 or n_test >= N:
        raise ValueError(
            f"test_frac={test_frac} produces n_test={n_test} from N={N}; "
            "must yield at least 1 sample in each split"
        )
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    perm = torch.randperm(N, generator=gen)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return (
        PICDataset(
            dataset.x[train_idx],
            dataset.port[train_idx],
            dataset.p[train_idx],
        ),
        PICDataset(
            dataset.x[test_idx],
            dataset.port[test_idx],
            dataset.p[test_idx],
        ),
    )


def make_synthetic_dataset(
    model: DigitalTwin,
    n_samples: int,
    *,
    x_max: float = X_MAX,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> PICDataset:
    """Generate an (x, port, p) dataset by evaluating `model` as ground truth.

    Powers are sampled uniformly from [0, x_max); input ports uniformly
    from [0, m). The model's forward pass produces the "measured"
    distribution p. Optional Gaussian noise on p (clipped non-negative and
    renormalised) emulates measurement noise.

    Args:
        model:     a DigitalTwin treated as ground truth.
        n_samples: number of (x, port, p) triples.
        x_max:     upper bound of the uniform power distribution, watts.
                   Defaults to the chip's budget (~1.4 W), which spans
                   about two full fringes.
        noise_std: stddev of additive Gaussian noise on p. 0 (default) is clean.
        seed:      RNG seed. None = non-deterministic.

    Returns:
        PICDataset of size n_samples, on CPU.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if x_max <= 0:
        raise ValueError(f"x_max must be positive, got {x_max}")
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    dtype = model.c_0.dtype
    device = model.c_0.device

    x = torch.rand((n_samples, model.n_PS), generator=gen, dtype=dtype) * x_max
    port = torch.randint(0, model.m, (n_samples,), generator=gen, dtype=torch.long)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        p_clean = model(x.to(device), port.to(device)).cpu()
    if was_training:
        model.train()

    if noise_std > 0:
        noise = torch.randn(p_clean.shape, generator=gen, dtype=dtype) * noise_std
        p = (p_clean + noise).clamp(min=0.0)
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    else:
        p = p_clean

    return PICDataset(x.cpu(), port.cpu(), p)
