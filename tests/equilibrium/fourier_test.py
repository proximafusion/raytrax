import jax.numpy as jnp
import numpy as np

from raytrax.equilibrium.fourier import (
    dvolume_drho,
    evaluate_magnetic_field_on_toroidal_grid,
    evaluate_rphiz_on_toroidal_grid,
)


def test_evaluate_rphiz_on_toroidal_grid(torus_wout):
    """Test the evaluate_rphiz_on_toroidal_grid function."""
    rho_theta_phi = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, 1, 8),
            jnp.linspace(0, 2 * jnp.pi, 6),
            jnp.linspace(0, 2 * jnp.pi, 7),
            indexing="ij",
        ),
        axis=-1,
    )
    assert rho_theta_phi.shape == (8, 6, 7, 3)
    rphiz = evaluate_rphiz_on_toroidal_grid(torus_wout, rho_theta_phi)
    assert rphiz.shape == (8, 6, 7, 3)
    assert np.all(np.isfinite(rphiz))

    phi = rho_theta_phi[..., 2]
    np.testing.assert_allclose(rphiz[..., 1], phi, rtol=0, atol=1e-16)

    rho = rho_theta_phi[..., 0]
    theta = rho_theta_phi[..., 1]
    major_radius = 2.0
    minor_radius = 0.5
    r_expected = major_radius + rho * jnp.cos(theta) * minor_radius
    np.testing.assert_allclose(rphiz[..., 0], r_expected, rtol=0, atol=1e-15)


def test_evaluate_magnetic_field_on_toroidal_grid(torus_wout):
    rho_theta_phi = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, 1, 8),
            jnp.linspace(0, 2 * jnp.pi, 6),
            jnp.linspace(0, 2 * jnp.pi, 7),
            indexing="ij",
        ),
        axis=-1,
    )
    bfield = evaluate_magnetic_field_on_toroidal_grid(torus_wout, rho_theta_phi)
    assert bfield.shape == (8, 6, 7, 3)
    assert np.all(np.isfinite(bfield))

    rho = rho_theta_phi[..., 0]
    theta = rho_theta_phi[..., 1]
    phi = rho_theta_phi[..., 2]
    major_radius = 2.0
    minor_radius = 0.5
    r = major_radius + rho * jnp.cos(theta) * minor_radius
    bfield_expected_xy = 0.7 * jnp.stack(
        [
            -r * jnp.sin(phi),
            r * jnp.cos(phi),
        ],
        axis=-1,
    )
    # z component should be zero
    np.testing.assert_allclose(bfield[..., 2], 0.0, rtol=0, atol=1e-6)
    # xy components
    np.testing.assert_allclose(bfield[..., :2], bfield_expected_xy, rtol=0, atol=1e-14)


def test_dvolume_drho_torus(torus_wout):
    """Test the dvolume_drho function with a simple torus equilibrium."""
    # Test with various rho values
    rho = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

    dv_drho = dvolume_drho(torus_wout, rho)

    assert dv_drho.shape == rho.shape
    assert jnp.all(jnp.isfinite(dv_drho))

    # dV/dρ = (2π)² g₀₀ × 2ρ.  For a torus g₀₀ = R₀ r₀²/2 = 0.25 (constant),
    # so dV/dρ = 2π² ρ exactly.
    np.testing.assert_allclose(dv_drho, 2 * np.pi**2 * rho, rtol=1e-3)


def test_dvolume_drho_torus_volume_integral(torus_wout):
    """Integrating dV/dρ over [0, 1] must recover the exact torus volume."""
    n_points = 1000
    rho = jnp.linspace(0.0, 1.0, n_points)
    dv_drho = dvolume_drho(torus_wout, rho)
    total_volume = float(jnp.trapezoid(dv_drho, rho))
    # Analytical: V = 2π² R₀ r₀² = 2π² × 2 × 0.25 = π²
    expected = 2 * np.pi**2 * 2.0 * 0.5**2
    np.testing.assert_allclose(total_volume, expected, rtol=1e-3)


def test_dvolume_drho_w7x_integration(w7x_wout):
    """Test that integrating dV/drho gives a reasonable total volume for W7-X."""
    n_points = 1000
    rho = jnp.linspace(0.001, 0.99, n_points)  # Avoid exact boundaries

    dv_drho = dvolume_drho(w7x_wout, rho)

    drho = rho[1] - rho[0]
    total_volume = jnp.trapezoid(dv_drho, dx=drho)

    expected_volume_min = 20.0  # m³ (conservative lower bound)
    expected_volume_max = 35.0  # m³ (conservative upper bound)
    assert total_volume > expected_volume_min, (
        f"Volume {total_volume:.2f} m³ is too small"
    )
    assert total_volume < expected_volume_max, (
        f"Volume {total_volume:.2f} m³ is too large"
    )
    assert jnp.isfinite(total_volume), "Total volume is not finite"

    assert total_volume > 0, "Total volume should be positive"


def test_extrapolation_beyond_lcms(torus_wout):
    """Test that extrapolation works correctly for rho > 1."""
    # Test with rho values extending beyond LCMS
    rho_theta_phi = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, 1.2, 10),  # Extend to rho=1.2
            jnp.linspace(0, 2 * jnp.pi, 6),
            jnp.linspace(0, 2 * jnp.pi, 7),
            indexing="ij",
        ),
        axis=-1,
    )

    # Test rphiz extrapolation
    rphiz = evaluate_rphiz_on_toroidal_grid(torus_wout, rho_theta_phi)
    assert rphiz.shape == (10, 6, 7, 3)
    assert np.all(np.isfinite(rphiz)), "Extrapolated rphiz contains NaN or Inf"

    # Check that flux surfaces expand outward for rho > 1
    # Compare rho=1.0 vs rho=1.2 at the same (theta, phi)
    idx_rho_1_0 = 8  # rho ≈ 1.07
    idx_rho_1_2 = 9  # rho = 1.2
    r_at_1_0 = rphiz[idx_rho_1_0, :, :, 0]
    r_at_1_2 = rphiz[idx_rho_1_2, :, :, 0]
    z_at_1_0 = rphiz[idx_rho_1_0, :, :, 2]
    z_at_1_2 = rphiz[idx_rho_1_2, :, :, 2]

    # Surfaces should expand: distance from axis should increase
    # For a torus, axis is at (R0, Z0) = (2.0, 0.0)
    dist_1_0 = jnp.sqrt((r_at_1_0 - 2.0) ** 2 + z_at_1_0**2)
    dist_1_2 = jnp.sqrt((r_at_1_2 - 2.0) ** 2 + z_at_1_2**2)
    assert jnp.all(dist_1_2 >= dist_1_0), "Flux surfaces should expand for rho > 1"

    # Test magnetic field extrapolation
    bfield = evaluate_magnetic_field_on_toroidal_grid(torus_wout, rho_theta_phi)
    assert bfield.shape == (10, 6, 7, 3)
    assert np.all(np.isfinite(bfield)), "Extrapolated B field contains NaN or Inf"

    # Magnetic field should be continuous across rho=1 boundary
    b_mag = jnp.linalg.norm(bfield, axis=-1)
    # Compare rho just below 1.0 vs just above
    idx_below = 7  # rho ≈ 0.93
    idx_above = 9  # rho = 1.2
    assert jnp.allclose(b_mag[idx_below], b_mag[idx_above], rtol=0.2), (
        "B field should be continuous across LCMS"
    )
