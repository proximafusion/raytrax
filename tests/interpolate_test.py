import jax.numpy as jnp
import numpy as np

from raytrax.fourier import evaluate_rphiz_on_toroidal_grid
from raytrax.interpolate import (
    cylindrical_grid_for_equilibrium,
    interpolate_toroidal_to_cylindrical_grid,
)

from .fixtures import torus_wout


def test_interpolate_toroidal_to_cylindrical_grid(torus_wout):
    rho_theta_phi = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, 1, 8),
            jnp.linspace(0, 2 * jnp.pi, 6),
            jnp.linspace(0, 2 * jnp.pi, 7),
            indexing="ij",
        ),
        axis=-1,
    )
    rphiz_toroidal = evaluate_rphiz_on_toroidal_grid(torus_wout, rho_theta_phi)
    rmin = np.min(rphiz_toroidal[..., 0])
    rmax = np.max(rphiz_toroidal[..., 0])
    zmin = np.min(rphiz_toroidal[..., 2])
    zmax = np.max(rphiz_toroidal[..., 2])
    rz_cylindrical = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(rmin, rmax, 4),
            jnp.linspace(zmin, zmax, 5),
            indexing="ij",
        ),
        axis=-1,
    )
    values_cylindrical = interpolate_toroidal_to_cylindrical_grid(
        rphiz_toroidal=rphiz_toroidal,
        rz_cylindrical=rz_cylindrical,
        value_toroidal=jnp.ones((8, 6, 7, 3)),
    )
    assert values_cylindrical.shape == (4, 7, 5, 3)
    # some of them will be NaN, but not all
    assert np.any(np.isfinite(values_cylindrical))
    # all values should be either NaN or 1.0
    np.testing.assert_allclose(
        values_cylindrical[np.isfinite(values_cylindrical)], 1.0, rtol=0, atol=1e-15
    )


def test_cylindrical_grid_for_equilibrium(torus_wout):
    """Test that cylindrical_grid_for_equilibrium works correctly."""
    n_rho = 10
    n_theta = 8
    n_phi = 6
    n_r = 7
    n_z = 9

    grid = cylindrical_grid_for_equilibrium(
        equilibrium=torus_wout,
        n_rho=n_rho,
        n_theta=n_theta,
        n_phi=n_phi,
        n_r=n_r,
        n_z=n_z,
    )

    assert grid.shape == (n_r, n_phi, n_z, 7)
    assert np.any(np.isfinite(grid))

    rphiz_data = grid[..., 0]
    field_data = grid[..., 1]

    assert np.any(np.isfinite(rphiz_data))
    assert np.any(np.isfinite(field_data))

    r_values = grid[..., 0, 0]
    z_values = grid[..., 0, 2]

    assert np.all(r_values[np.isfinite(r_values)] > 0)

    major_radius = 2.0
    minor_radius = 0.5
    finite_r = r_values[np.isfinite(r_values)]
    finite_z = z_values[np.isfinite(z_values)]
    if len(finite_r) > 0:
        assert np.min(finite_r) >= major_radius - minor_radius
        assert np.max(finite_r) <= major_radius + minor_radius

    if len(finite_z) > 0:
        assert np.min(finite_z) >= -minor_radius
        assert np.max(finite_z) <= minor_radius
