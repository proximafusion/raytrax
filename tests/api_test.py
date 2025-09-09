"""Tests for the raytrax.api module."""

import jax.numpy as jnp

from raytrax.api import get_interpolator_for_equilibrium
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
