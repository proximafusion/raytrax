"""Tests for 3D plotting functions."""

import pytest

from raytrax.plot.plot3d import plot_flux_surface_3d

from ..fixtures import w7x_magnetic_configuration

pytest.importorskip("pyvista")


def test_plot_flux_surface_3d_w7x(w7x_magnetic_configuration):
    """Test that plot_flux_surface_3d works with W7-X configuration."""
    plotter = plot_flux_surface_3d(w7x_magnetic_configuration, rho_value=1.0)
    assert plotter is not None
