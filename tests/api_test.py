"""Tests for the raytrax.api module."""

import jax.numpy as jnp
import numpy as np

from raytrax.api import trace, _bin_power_deposition
from raytrax.equilibrium.interpolate import MagneticConfiguration
from raytrax.types import Beam, RadialProfiles
from tests.fixtures import w7x_wout


def test_from_vmec_wout_w7x(w7x_wout):
    """Test the MagneticConfiguration.from_vmec_wout classmethod with the W7-X equilibrium."""
    # Generate the interpolator
    interpolator = MagneticConfiguration.from_vmec_wout(w7x_wout)

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

    # Check that non-NaN rho values are non-negative
    # Note: rho > 1 is expected for points outside the LCMS (in the vacuum region)
    valid_rho = interpolator.rho[~jnp.isnan(interpolator.rho)]
    assert valid_rho.size > 0, "All rho values are NaN"
    assert jnp.all(valid_rho >= 0.0)
    # Check that we have some points inside the plasma (rho < 1)
    assert jnp.any(valid_rho < 1.0), "No points found inside plasma (rho < 1)"
    # Check that we have some points outside the plasma (rho > 1)
    assert jnp.any(valid_rho > 1.0), "No points found outside plasma (rho > 1)"

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
    r = 6.50866
    phi = np.deg2rad(-6.56378)
    z = 0.38
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    position = jnp.array([x, y, z])

    # Beam direction starting from W7-X aiming angles
    alpha = np.deg2rad(15.7)
    beta = np.deg2rad(19.7001)
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
    electron_temperature = 5 * quadratic_profile  # 5 keV on axis
    electron_density = 0.75 * quadratic_profile  # 0.75e20/m³ on axis
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

    interpolator = MagneticConfiguration.from_vmec_wout(w7x_wout)
    result = trace(interpolator, radial_profiles, beam)

    assert hasattr(result, "beam_profile")
    assert hasattr(result, "radial_profile")
    assert result.beam_profile is not None
    assert result.radial_profile is not None


def test_bin_power_deposition():
    """Test the _bin_power_deposition helper function."""
    rho_grid = jnp.linspace(0.0, 1.0, 11)  # 11 points: 0.0, 0.1, ..., 1.0
    dvolume_drho = jnp.linspace(1.0, 2.0, 11)

    # Non-monotonic trajectory: enters at rho=0.9, reaches rho=0.3, exits to rho=0.8.
    # Crosses rho≈0.5 twice.
    rho_trajectory = jnp.array([0.9, 0.7, 0.5, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    optical_depth = jnp.linspace(0.0, 2.0, 9)  # tau: 0 → 2 along trajectory
    arc_length = jnp.linspace(0.0, 1.0, 9)

    result = _bin_power_deposition(
        rho_grid, dvolume_drho, arc_length, rho_trajectory, optical_depth
    )

    assert result.shape == rho_grid.shape
    assert jnp.all(jnp.isfinite(result))
    assert result[5] > 0  # rho=0.5 bin should have deposited power

    # Power conservation: sum(dP/dV * dV) = exp(-tau[0]) - exp(-tau[-1])
    edges = jnp.concatenate(
        [rho_grid[:1], 0.5 * (rho_grid[:-1] + rho_grid[1:]), rho_grid[-1:]]
    )
    total_deposited = jnp.sum(result * dvolume_drho * jnp.diff(edges))
    expected_total = float(jnp.exp(-optical_depth[0]) - jnp.exp(-optical_depth[-1]))
    np.testing.assert_allclose(total_deposited, expected_total, rtol=1e-5)


def test_bin_power_deposition_power_conservation():
    """Total deposited power equals exp(-tau_0) - exp(-tau_final) exactly."""
    rho_grid = jnp.linspace(0.0, 1.0, 200)
    dvolume_drho = jnp.ones(200)

    # Monotone trajectory: enters at rho=0.9, exits at rho=0.1
    rho_trajectory = jnp.linspace(0.9, 0.1, 20)
    optical_depth = jnp.linspace(0.0, 3.0, 20)  # tau_final = 3
    arc_length = jnp.linspace(0.0, 1.0, 20)

    result = _bin_power_deposition(
        rho_grid, dvolume_drho, arc_length, rho_trajectory, optical_depth
    )

    assert result.shape == rho_grid.shape

    traversed = (rho_grid >= 0.1) & (rho_grid <= 0.9)
    assert jnp.all(
        result[traversed] > 0
    ), "Bins along trajectory path should be populated"

    edges = jnp.concatenate(
        [rho_grid[:1], 0.5 * (rho_grid[:-1] + rho_grid[1:]), rho_grid[-1:]]
    )
    total_deposited = jnp.sum(result * dvolume_drho * jnp.diff(edges))
    expected_total = 1.0 - float(jnp.exp(-optical_depth[-1]))  # 1 - exp(-3)
    np.testing.assert_allclose(total_deposited, expected_total, rtol=1e-5)


def test_bin_power_with_padded_zeros():
    """Padded entries (optical_depth=inf after trajectory end) contribute zero power."""
    rho_grid = jnp.linspace(0.0, 1.0, 11)
    dvolume_drho = jnp.ones(11)

    # 5 valid points followed by 5 inf-padded slots (diffrax fills unused slots with inf)
    rho_valid = jnp.array([0.8, 0.6, 0.4, 0.3, 0.5])
    tau_valid = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
    s_valid = jnp.linspace(0.0, 1.0, 5)
    rho_padded = jnp.zeros(5)
    tau_padded = jnp.full(5, jnp.inf)  # diffrax fills unused sol.ys slots with inf
    s_padded = jnp.full(5, jnp.inf)

    rho_trajectory = jnp.concatenate([rho_valid, rho_padded])
    optical_depth = jnp.concatenate([tau_valid, tau_padded])
    arc_length = jnp.concatenate([s_valid, s_padded])

    result_padded = _bin_power_deposition(
        rho_grid, dvolume_drho, arc_length, rho_trajectory, optical_depth
    )
    result_trimmed = _bin_power_deposition(
        rho_grid, dvolume_drho, s_valid, rho_valid, tau_valid
    )

    assert result_padded.shape == rho_grid.shape
    assert jnp.all(jnp.isfinite(result_padded))
    # Padded inf entries must not change the result
    np.testing.assert_allclose(result_padded, result_trimmed, rtol=1e-5)
