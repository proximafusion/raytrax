"""Tests for 3D plotting functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from raytrax.plot.plot3d import (
    plot_b_surface_3d,
    plot_beam_profile_3d,
    plot_flux_surface_3d,
)
from raytrax.types import BeamProfile

from ..fixtures import w7x_magnetic_configuration, w7x_wout

pytest.importorskip("pyvista")


def test_plot_flux_surface_3d_w7x(w7x_magnetic_configuration):
    """Test that plot_flux_surface_3d works with W7-X configuration."""
    plotter = plot_flux_surface_3d(w7x_magnetic_configuration, rho_value=1.0)
    assert plotter is not None


def test_plot_b_surface_3d_returns_plotter(w7x_magnetic_configuration):
    """plot_b_surface_3d returns a PyVista plotter without error."""
    B_res = 140e9 / (2.0 * 27.99e9)  # ≈ 2.502 T — 2nd harmonic resonance
    plotter = plot_b_surface_3d(w7x_magnetic_configuration, b_value=B_res)
    assert plotter is not None


def test_plot_b_surface_3d_accepts_existing_plotter(w7x_magnetic_configuration):
    """plot_b_surface_3d adds to a supplied plotter rather than creating a new one."""
    import pyvista as pv

    p = pv.Plotter(off_screen=True)
    result = plot_b_surface_3d(w7x_magnetic_configuration, b_value=2.4, plotter=p)
    assert result is p


def _straight_beam_profile(n: int = 20) -> BeamProfile:
    """Minimal straight-line BeamProfile for testing."""
    s = jnp.linspace(0.0, 1.0, n)
    # Straight line along x from (5, 0, 0) to (6, 0, 0)
    pos = jnp.stack([5.0 + s, jnp.zeros(n), jnp.zeros(n)], axis=-1)
    zeros = jnp.zeros(n)
    return BeamProfile(
        position=pos,
        arc_length=s,
        refractive_index=jnp.ones((n, 3)),
        optical_depth=zeros,
        absorption_coefficient=zeros,
        electron_density=zeros,
        electron_temperature=zeros,
        magnetic_field=jnp.ones((n, 3)) * 2.5,
        normalized_effective_radius=s,
        linear_power_density=jnp.exp(-((s - 0.5) ** 2) / 0.05),  # Gaussian peak
    )


def test_plot_beam_profile_3d_returns_plotter():
    """plot_beam_profile_3d returns a PyVista plotter without error."""
    plotter = plot_beam_profile_3d(_straight_beam_profile())
    assert plotter is not None


def test_plot_beam_profile_3d_accepts_existing_plotter():
    """plot_beam_profile_3d adds to a supplied plotter rather than creating a new one."""
    import pyvista as pv

    p = pv.Plotter(off_screen=True)
    result = plot_beam_profile_3d(_straight_beam_profile(), plotter=p)
    assert result is p
