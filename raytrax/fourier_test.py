from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from raytrax.fourier import (
    evaluate_magnetic_field_on_toroidal_grid,
    evaluate_rphiz_on_toroidal_grid,
    interpolate_toroidal_to_cylindrical_grid,
)

jax.config.update("jax_enable_x64", True)


@dataclass
class TestWout:
    rmnc: jax.Array
    zmns: jax.Array
    xm: jax.Array
    xn: jax.Array
    bsupumnc: jax.Array
    bsupvmnc: jax.Array
    xm_nyq: jax.Array
    xn_nyq: jax.Array
    ns: int
    lasym: bool = False


@pytest.fixture
def torus_wout():
    """Fixture for a torus shaped Wout-like object."""
    n_surfaces = 5
    major_radius = 2.0
    minor_radius = 0.5
    rmnc = np.zeros((n_surfaces, 2))
    rmnc[:, 0] = major_radius
    rmnc[:, 1] = np.sqrt(np.linspace(0, 1, n_surfaces)) * minor_radius

    xm = np.array([0, 1])
    xn = np.array([0, 0])
    zmns = np.zeros((n_surfaces, 2))
    zmns[:, 1] = np.sqrt(np.linspace(0, 1, n_surfaces)) * minor_radius

    xm_nyq = np.array([0, 1])
    xn_nyq = np.array([0, 0])
    bsupumnc = np.zeros((n_surfaces - 1, 2))
    bsupvmnc = np.zeros((n_surfaces - 1, 2))
    bsupvmnc[:, 0] = 0.7
    return TestWout(
        rmnc=jnp.array(rmnc),
        zmns=jnp.array(zmns),
        xm=jnp.array(xm),
        xn=jnp.array(xn),
        bsupumnc=jnp.array(bsupumnc),
        bsupvmnc=jnp.array(bsupvmnc),
        xm_nyq=jnp.array(xm_nyq),
        xn_nyq=jnp.array(xn_nyq),
        ns=5,
        lasym=False,
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
    np.testing.assert_allclose(bfield[..., :2], bfield_expected_xy, rtol=0, atol=1e-15)


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
