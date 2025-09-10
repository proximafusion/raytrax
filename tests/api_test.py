"""Tests for the raytrax.api module."""

import jax.numpy as jnp
import numpy as np

from raytrax.api import get_interpolator_for_equilibrium, trace
from raytrax.types import Beam, RadialProfiles
from tests.fixtures import w7x_wout


def test_get_interpolator_for_equilibrium_w7x(w7x_wout):
    """Test the get_interpolator_for_equilibrium function with the W7-X equilibrium."""
    # Generate the interpolator
    interpolator = get_interpolator_for_equilibrium(w7x_wout)

    # Check that the interpolator has the expected attributes
    assert hasattr(interpolator, "rphiz")
    assert hasattr(interpolator, "magnetic_field")
    assert hasattr(interpolator, "rho")

    # Check the shapes match
    # The actual shape is (n_r, n_phi, n_z, 3) for rphiz and magnetic_field
    # and (n_r, n_phi, n_z) for rho
    assert interpolator.rphiz.shape == (45, 50, 55, 3)
    assert interpolator.magnetic_field.shape == (45, 50, 55, 3)
    assert interpolator.rho.shape == (45, 50, 55)

    # Check that non-NaN rho values are within expected range [0, 1]
    # We expect some NaN values for points outside the plasma
    valid_rho = interpolator.rho[~jnp.isnan(interpolator.rho)]
    assert valid_rho.size > 0, "All rho values are NaN"
    assert jnp.all(valid_rho >= 0.0)
    assert jnp.all(valid_rho <= 1.0)

    # Verify that the magnetic field has non-zero values
    assert not jnp.all(interpolator.magnetic_field == 0.0)

    # Test that the rphiz coordinates make sense physically
    # For W7-X, R (major radius) should be around 5.5m
    r_values = interpolator.rphiz[..., 0]  # First component is r
    assert jnp.min(r_values) > 0.0  # Should be positive
    assert jnp.max(r_values) < 7.0  # Should be less than ~7m for W7-X

    # Z values should be within reasonable range for W7-X
    z_values = interpolator.rphiz[..., 2]  # Third component is z
    assert jnp.min(z_values) > -1.5  # Lower bound
    assert jnp.max(z_values) < 1.5  # Upper bound


def test_trace_w7x_beam(w7x_wout):
    """Test the trace function with W7-X equilibrium and a specific beam position/direction."""
    # Beam position
    r = 6.55803
    phi = -6.56692
    z = -0.1
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    position = jnp.array([x, y, z])

    # Beam direction starting from W7-X aiming angles
    alpha = np.deg2rad(4.59997)
    beta = np.deg2rad(20.4)
    d_r = -np.cos(alpha) * np.cos(beta)
    d_phi = np.cos(alpha) * np.sin(beta)
    d_z = np.sin(alpha)
    dir_x = d_r * np.cos(phi) - d_phi * np.sin(phi)
    dir_y = d_r * np.sin(phi) + d_phi * np.cos(phi)
    dir_z = d_z
    direction = np.array([dir_x, dir_y, dir_z])
    direction = direction / np.linalg.norm(direction)

    rho = np.linspace(0, 1, 45)
    quadratic_profile = (1 - rho) ** 2
    electron_temperature = 5 * quadratic_profile # 5 keV on axis
    electron_density = 0.75 * quadratic_profile # 0.75e20/m³ on axis
    radial_profiles = RadialProfiles(
        rho=rho,
        electron_density=electron_density,
        electron_temperature=electron_temperature,
    )

    beam = Beam(
        position=position,
        direction=direction,
        frequency=140e9,
        mode="O",
    )

    interpolator = get_interpolator_for_equilibrium(w7x_wout)
    result = trace(interpolator, radial_profiles, beam)

    assert hasattr(result, "beam_profile")
    assert hasattr(result, "radial_profile")
    assert result.beam_profile is not None
    assert result.radial_profile is not None
