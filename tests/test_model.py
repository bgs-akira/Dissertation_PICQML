"""Tests for the DigitalTwin forward model."""

import pytest


def test_lossless_limit_is_unitary():
    """With c_0=0, C_2=0, R=0.5, T_out=1, U_0.conj().T @ U_0 ~= I_m.

    See CLAUDE.md §3.5 (sanity check) and §10.11.
    """
    pytest.skip("not implemented")


def test_output_distribution_sums_to_one():
    """forward(V, port).sum(dim=-1) == 1 within float tolerance."""
    pytest.skip("not implemented")


def test_c0_has_no_gradient():
    """c_0 is a buffer; loss.backward() must not produce a gradient for it."""
    pytest.skip("not implemented")


def test_trainable_params_have_gradients():
    """C_2_raw, R_logit, T_logit all receive non-zero gradients after one step."""
    pytest.skip("not implemented")


def test_r_and_tout_stay_in_unit_interval():
    """Sigmoid parameterisation keeps R in (0,1) and T_out in (0,1]."""
    pytest.skip("not implemented")
