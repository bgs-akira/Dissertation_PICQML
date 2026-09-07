"""Tests for the DigitalTwin forward model.

Note on T_in: there is no T_in parameter in this model. Multiplying column i
of the chip matrix by sqrt(T_in[i]) scales the entire column by a real
constant, which cancels in the L2 normalisation in forward(). Its gradient
is therefore mathematically zero (CLAUDE.md §10 pitfall 1). T_in is recovered
in the separate ITM stage and is not learned here. test_T_in_has_no_effect
is implicit and intentionally omitted.
"""

import math

import pytest
import torch

from src.chip_mesh import ChipMesh
from src.model import DigitalTwin


def _make_model(m: int = 4, c2_diag_value: float = 0.034) -> DigitalTwin:
    mesh = ChipMesh.clements(m)
    c_0 = torch.zeros(mesh.n_PS)
    c2_diag = torch.full((mesh.n_PS,), c2_diag_value)
    return DigitalTwin(mesh, c_0, c2_diag)


def test_init_shapes():
    """Parameter and buffer shapes match the mesh."""
    m = 4
    mesh = ChipMesh.clements(m)
    model = DigitalTwin(mesh, torch.zeros(mesh.n_PS), torch.zeros(mesh.n_PS))
    assert model.C_2.shape == (mesh.n_PS, mesh.n_PS)
    assert model.R.shape == (mesh.n_BS,)
    assert model.T_out.shape == (m,)
    assert model.c_0.shape == (mesh.n_PS,)


def test_c_0_is_buffer_not_parameter():
    """c_0 is a buffer (no grad). Trainable params: C_2_raw, R_logit, T_logit."""
    model = _make_model()
    param_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    assert param_names == {"C_2_raw", "R_logit", "T_logit"}
    assert "c_0" in buffer_names
    assert not model.c_0.requires_grad


def test_R_initialisation():
    """R_logit = 0 -> R = 0.5 within float precision."""
    model = _make_model()
    assert torch.allclose(model.R, torch.full_like(model.R, 0.5))


def test_T_out_initialisation():
    """T_logit = 6.0 -> T_out ~ sigmoid(6) ~ 0.998."""
    model = _make_model()
    expected = 1.0 / (1.0 + math.exp(-6.0))
    assert torch.allclose(
        model.T_out, torch.full_like(model.T_out, expected), atol=1e-6
    )


def test_forward_output_shape():
    """forward returns shape (batch, m)."""
    m = 4
    model = _make_model(m)
    batch = 8
    V = torch.rand(batch, model.n_PS)
    ports = torch.randint(0, m, (batch,))
    out = model(V, ports)
    assert out.shape == (batch, m)


def test_forward_normalisation():
    """Output rows sum to 1 within 1e-5."""
    model = _make_model(4)
    V = torch.rand(8, model.n_PS)
    ports = torch.randint(0, 4, (8,))
    out = model(V, ports)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_lossless_limit_unitary():
    """With c_0=0, C_2=0, R=0.5, T_out~1, the chip matrix U_0 is unitary.

    Verified directly via U_0.conj().T @ U_0 ~= I (CLAUDE.md §3.5 sanity check).
    """
    m = 4
    mesh = ChipMesh.clements(m)
    model = DigitalTwin(mesh, torch.zeros(mesh.n_PS), torch.zeros(mesh.n_PS))
    # R_logit = 0 already gives R = 0.5 from the constructor defaults.
    with torch.no_grad():
        model.T_logit.fill_(1e6)  # T_out -> 1.0 numerically
    phi = torch.zeros(mesh.n_PS)
    U_0 = model._build_U(phi)
    assert U_0.shape == (m, m)
    product = U_0.conj().T @ U_0
    eye = torch.eye(m, dtype=U_0.dtype)
    assert torch.allclose(product, eye, atol=1e-4)


def test_gradient_flow():
    """Trainable parameters get non-zero gradients; c_0 (buffer) does not."""
    model = _make_model(4)
    V = torch.rand(2, model.n_PS)
    ports = torch.randint(0, 4, (2,))
    p_hat = model(V, ports)
    p_true = torch.rand_like(p_hat)
    p_true = p_true / p_true.sum(dim=-1, keepdim=True)
    loss = ((p_hat - p_true) ** 2).sum(dim=-1).mean()
    loss.backward()
    assert model.C_2_raw.grad is not None
    assert model.C_2_raw.grad.abs().sum() > 0
    assert model.R_logit.grad is not None
    assert model.R_logit.grad.abs().sum() > 0
    assert model.T_logit.grad is not None
    assert model.T_logit.grad.abs().sum() > 0
    assert model.c_0.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_propagation():
    """forward runs on GPU after .to('cuda')."""
    model = _make_model(4).to("cuda")
    V = torch.rand(2, model.n_PS, device="cuda")
    ports = torch.randint(0, 4, (2,), device="cuda")
    out = model(V, ports)
    assert out.device.type == "cuda"
    assert out.shape == (2, 4)
