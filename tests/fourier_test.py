import jax.numpy as jnp
import numpy as np

from raytrax.fourier import (
    evaluate_magnetic_field_on_toroidal_grid,
    evaluate_rphiz_on_toroidal_grid,
    dvolume_drho,
)

from .fixtures import torus_wout, w7x_wout



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
    np.testing.assert_allclose(bfield[..., :2], bfield_expected_xy, rtol=0, atol=1e-15)


def test_dvolume_drho_torus(torus_wout):
    """Test the dvolume_drho function with a simple torus equilibrium."""
    # Test with various rho values
    rho = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    dv_drho = dvolume_drho(torus_wout, rho)
    
    # Check shape and basic properties
    assert dv_drho.shape == rho.shape
    assert jnp.all(jnp.isfinite(dv_drho))
    assert jnp.all(dv_drho > 0)  # Volume derivative should be positive
    
    # For our test fixture, g_{0,0} varies linearly from 0.1 to 1.0
    # so the interpolated values should be reasonable
    expected_min = (2 * jnp.pi)**2 * 0.1
    expected_max = (2 * jnp.pi)**2 * 1.0
    assert jnp.all(dv_drho >= expected_min * 0.9)  # Allow some tolerance
    assert jnp.all(dv_drho <= expected_max * 1.1)


def test_dvolume_drho_w7x_integration(w7x_wout):
    """Test that integrating dV/drho gives a reasonable total volume for W7-X."""
    n_points = 1000
    rho = jnp.linspace(0.001, 0.99, n_points)  # Avoid exact boundaries
    
    dv_drho = dvolume_drho(w7x_wout, rho)
    
    drho = rho[1] - rho[0]
    total_volume = jnp.trapezoid(dv_drho, dx=drho)
    
    expected_volume_min = 20.0   # m³ (conservative lower bound)
    expected_volume_max = 35.0   # m³ (conservative upper bound)
    assert total_volume > expected_volume_min, f"Volume {total_volume:.2f} m³ is too small"
    assert total_volume < expected_volume_max, f"Volume {total_volume:.2f} m³ is too large"
    assert jnp.isfinite(total_volume), "Total volume is not finite"
    
    assert total_volume > 0, "Total volume should be positive"
