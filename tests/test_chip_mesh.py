"""Tests for ChipMesh topology construction."""

import pytest


def test_clements_layer_count():
    """A Clements mesh on m modes has 2m alternating columns."""
    pytest.skip("not implemented")


def test_clements_disjoint_waveguides_per_column():
    """Within any single column, no waveguide appears in two BS pairs or two PS slots."""
    pytest.skip("not implemented")


def test_clements_global_indices_unique():
    """Every PS gets a unique global index in [0, n_PS); same for BS."""
    pytest.skip("not implemented")


def test_clements_counts_match_paper():
    """For m = 12: n_BS = 132, n_PS matches the convention chosen in chip_mesh.py."""
    pytest.skip("not implemented")
