"""Forward-model tests on the Bell mesh at the project's target size m = 10.

Mirrors tests/test_model.py, which covers the Clements mesh at m = 4. The
point of duplicating rather than parametrising is that these assert the
Bell-specific shapes (n_PS = m**2 = 100, n_BS = m(m-1) = 90) and run the
unitarity check at the size the dissertation actually uses.
"""

import pytest
import torch

from src.chip_mesh import ChipMesh
from src.model import DigitalTwin


M = 10


@pytest.fixture(scope="module")
def mesh() -> ChipMesh:
    return ChipMesh.bell(M)


def _model(
    mesh: ChipMesh,
    *,
    c2_diag: float = 0.034,
    c0_scale: float = 0.3,
    seed: int = 0,
    dtype=torch.float64,
) -> DigitalTwin:
    """A DigitalTwin seeded the way V-IFM would seed it.

    The defaults matter: seeding ``C_2 = 0`` together with ``c_0 = 0``
    puts every fringe at an extremum where dP/dphi vanishes, so no
    gradient flows at all (see
    ``test_gradients_vanish_at_the_symmetric_point``). Real seeds are
    never degenerate that way, so the gradient tests below use realistic
    ones.
    """
    g = torch.Generator().manual_seed(seed)
    c_0 = torch.randn(mesh.n_PS, generator=g, dtype=dtype) * c0_scale
    diag = torch.full((mesh.n_PS,), c2_diag, dtype=dtype)
    return DigitalTwin(mesh, c_0, diag, dtype=dtype)


def test_parameter_shapes(mesh):
    """C_2 is 100x100, R is 90, T_out is 10."""
    model = _model(mesh)
    assert model.C_2.shape == (100, 100)
    assert model.R.shape == (90,)
    assert model.T_out.shape == (M,)
    assert model.c_0.shape == (100,)


def test_lossless_limit_is_unitary(mesh):
    """CLAUDE.md 3.5 sanity check, on Bell at m = 10.

    With phi = 0 and R = 0.5 everywhere the chip matrix must be unitary.
    This is the check that catches a miswired mesh.
    """
    model = _model(mesh)
    U_0 = model._build_U(torch.zeros(mesh.n_PS, dtype=torch.float64))
    assert U_0.shape == (M, M)
    product = U_0.conj().T @ U_0
    eye = torch.eye(M, dtype=U_0.dtype)
    assert torch.allclose(product, eye, atol=1e-10)


def test_unitary_for_arbitrary_phases_and_reflectivities(mesh):
    """Unitarity must not depend on being at the symmetric operating point."""
    model = _model(mesh)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        model.R_logit.copy_(
            torch.randn(mesh.n_BS, generator=g, dtype=torch.float64) * 0.4
        )
    phi = torch.randn(mesh.n_PS, generator=g, dtype=torch.float64) * 2.0
    U_0 = model._build_U(phi)
    product = U_0.conj().T @ U_0
    eye = torch.eye(M, dtype=U_0.dtype)
    assert torch.allclose(product, eye, atol=1e-10)


def test_every_phase_shifter_changes_the_chip_matrix(mesh):
    """No dead parameters.

    A shifter that cannot influence U would be unidentifiable for the ML
    stage -- its gradient would vanish and it would never be learned. On
    the Bell mesh all 100 must be live, including the 10 independents.
    """
    model = _model(mesh)
    base = model._build_U(torch.zeros(mesh.n_PS, dtype=torch.float64))
    dead = []
    for k in range(mesh.n_PS):
        phi = torch.zeros(mesh.n_PS, dtype=torch.float64)
        phi[k] = 1.0
        if torch.allclose(model._build_U(phi), base, atol=1e-13):
            dead.append(k)
    assert dead == []


def test_forward_shape_and_normalisation(mesh):
    model = _model(mesh)
    batch = 8
    V = torch.rand(batch, mesh.n_PS, dtype=torch.float64) * 14.0
    ports = torch.randint(0, M, (batch,))
    out = model(V, ports)
    assert out.shape == (batch, M)
    assert torch.allclose(
        out.sum(dim=-1), torch.ones(batch, dtype=torch.float64), atol=1e-10
    )


def test_gradient_flow(mesh):
    """All three trainable blocks receive gradient; c_0 stays a buffer."""
    model = _model(mesh)
    V = torch.rand(4, mesh.n_PS, dtype=torch.float64) * 14.0
    ports = torch.randint(0, M, (4,))
    p_hat = model(V, ports)
    p = torch.rand_like(p_hat)
    p = p / p.sum(dim=-1, keepdim=True)
    ((p_hat - p) ** 2).sum(dim=-1).mean().backward()
    assert model.C_2_raw.grad.abs().sum() > 0
    assert model.R_logit.grad.abs().sum() > 0
    assert model.T_logit.grad.abs().sum() > 0
    assert model.c_0.grad is None


def test_independent_shifters_receive_gradient(mesh):
    """The 10 boundary shifters must be learnable, not decorative.

    They carry the phases a full Bell scheme would put in its output
    layer, so a bug that left them dangling would show up here as a zero
    row in dC_2 rather than as a wrong answer.
    """
    model = _model(mesh)
    V = torch.rand(16, mesh.n_PS, dtype=torch.float64) * 14.0
    ports = torch.randint(0, M, (16,))
    p_hat = model(V, ports)
    p = torch.rand_like(p_hat)
    p = p / p.sum(dim=-1, keepdim=True)
    ((p_hat - p) ** 2).sum(dim=-1).mean().backward()
    for idx in mesh.ps_indices_with_role("independent"):
        assert model.C_2_raw.grad[idx].abs().sum() > 0, (
            f"independent shifter {idx} (label {mesh.ps_label(idx)}) "
            f"got no gradient"
        )


def test_gradients_vanish_at_the_symmetric_point(mesh):
    """Seeding C_2 = 0 AND c_0 = 0 is a dead start -- documented, not fixed.

    With both at zero, phi = 0 for every input regardless of the drive.
    Every fringe then sits exactly at an extremum, where dP/dphi = 0, so
    no gradient reaches C_2 at all and training cannot start. This is a
    property of the physics, not a bug: it is why the protocol seeds
    diag(C_2) and c_0 from a calibration stage before the ML stage runs,
    and why the seeds must never both be zeroed.

    Note the mesh itself is fine here -- every shifter still moves U (see
    test_every_phase_shifter_changes_the_chip_matrix). It is only the
    derivative at this particular operating point that vanishes.
    """
    degenerate = DigitalTwin(
        mesh,
        torch.zeros(mesh.n_PS, dtype=torch.float64),
        torch.zeros(mesh.n_PS, dtype=torch.float64),
        dtype=torch.float64,
    )
    V = torch.rand(8, mesh.n_PS, dtype=torch.float64) * 14.0
    ports = torch.randint(0, M, (8,))
    p_hat = degenerate(V, ports)
    p = torch.rand_like(p_hat)
    p = p / p.sum(dim=-1, keepdim=True)
    ((p_hat - p) ** 2).sum(dim=-1).mean().backward()
    assert degenerate.C_2_raw.grad.abs().max() == 0.0
