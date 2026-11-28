import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from raytrax.plot.plot2d import (
    interpolate_rz_slice,
    plot_electron_density_rz,
    plot_magnetic_field_rz,
)
from raytrax.types import RadialProfiles


def test_interpolate_rz_slice_w7x(w7x_magnetic_configuration):
    """Test that interpolate_rz_slice works for W7-X stellarator."""
    phi = 0.0
    n_r, n_z = 100, 120

    slice_data = interpolate_rz_slice(w7x_magnetic_configuration, phi, n_r=n_r, n_z=n_z)

    # Check shapes
    assert slice_data.R.shape == (n_r, n_z)
    assert slice_data.Z.shape == (n_r, n_z)
    assert slice_data.B.shape == (n_r, n_z)
    assert slice_data.rho.shape == (n_r, n_z)

    # Check that B values are positive where defined (not NaN)
    valid_B = slice_data.B[~np.isnan(slice_data.B)]
    assert len(valid_B) > 0
    assert np.all(valid_B > 0)

    # Check that rho is in reasonable range
    valid_rho = slice_data.rho[~np.isnan(slice_data.rho)]
    assert len(valid_rho) > 0
    assert np.all(valid_rho >= 0)


def test_interpolate_rz_slice_w7x_different_phi(w7x_magnetic_configuration):
    """Test that interpolation works at different phi values for W7-X."""
    # W7-X has 5 field periods, fundamental domain is [0, 2π/5]
    phi_0 = 0.0
    phi_mid = np.pi / 5  # Middle of fundamental domain

    slice_0 = interpolate_rz_slice(w7x_magnetic_configuration, phi=phi_0)
    slice_mid = interpolate_rz_slice(w7x_magnetic_configuration, phi=phi_mid)

    # Should get valid data at both positions
    assert np.any(~np.isnan(slice_0.B))
    assert np.any(~np.isnan(slice_mid.B))

    # Due to toroidal variation, B should differ
    valid_both = ~np.isnan(slice_0.B) & ~np.isnan(slice_mid.B)
    if np.any(valid_both):
        # Check that at least some values are different (stellarator asymmetry)
        assert not np.allclose(slice_0.B[valid_both], slice_mid.B[valid_both])


def test_axisymmetric_slice_at_any_angle(tokamak_magnetic_configuration):
    """A rotation does not change a tokamak's poloidal slice."""
    config = tokamak_magnetic_configuration
    ref = interpolate_rz_slice(config, 0.0, n_r=24, n_z=26)
    for phi in [np.pi / 2, 2 * np.pi, -np.pi / 2]:
        rotated = interpolate_rz_slice(config, phi, n_r=24, n_z=26)
        np.testing.assert_allclose(rotated.B, ref.B)
        np.testing.assert_allclose(rotated.rho, ref.rho)


def test_w7x_slice_field_period_and_stellarator_mirror(w7x_magnetic_configuration):
    """Equivalent planes repeat after a field period and reflect in Z at -phi."""
    config = w7x_magnetic_configuration
    period = 2 * np.pi / config.nfp
    phi = period / 4
    ref = interpolate_rz_slice(config, phi, n_r=24, n_z=26)
    for angle, reflected in [(phi + period, False), (-phi, True)]:
        other = interpolate_rz_slice(config, angle, n_r=24, n_z=26)
        np.testing.assert_array_equal(other.Z, ref.Z)
        np.testing.assert_allclose(other.B, ref.B[:, ::-1] if reflected else ref.B)
        np.testing.assert_allclose(
            other.rho, ref.rho[:, ::-1] if reflected else ref.rho
        )


def test_plot_magnetic_field_rz_w7x(w7x_magnetic_configuration):
    """Test that plot_magnetic_field_rz works for W7-X."""
    fig, ax = plt.subplots()
    result = plot_magnetic_field_rz(w7x_magnetic_configuration, phi=0.0, ax=ax)

    # Should return the axes object
    assert result is ax

    # Check that axis labels are set
    assert ax.get_xlabel() == "R [m]"
    assert ax.get_ylabel() == "Z [m]"

    plt.close(fig)


def test_plot_electron_density_rz_w7x(w7x_magnetic_configuration):
    """Test that plot_electron_density_rz works for W7-X."""
    # Create simple parabolic profile
    rho = jnp.linspace(0, 1, 50)
    ne = 2.0 * (1 - rho**2)
    te = 4.0 * (1 - rho**2)
    profiles = RadialProfiles(rho=rho, electron_density=ne, electron_temperature=te)

    fig, ax = plt.subplots()
    result = plot_electron_density_rz(
        w7x_magnetic_configuration, profiles, phi=0.0, ax=ax
    )

    # Should return the axes object
    assert result is ax

    plt.close(fig)
