from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from raytrax.fourier import evaluate_rphiz_on_toroidal_grid


@dataclass
class TestWout:
    rmnc: np.ndarray
    zmns: np.ndarray
    xm: np.ndarray
    xn: np.ndarray
    lasym: bool = False


@pytest.fixture
def torus_wout():
    """Fixture for a torus shaped Wout-like object."""
    n_surfaces = 5
    major_radius = 2.0
    minor_radius = 0.5
    rmnc = np.zeros((2, n_surfaces))
    rmnc[0] = major_radius
    rmnc[1] = np.linspace(0, 1, n_surfaces) * minor_radius

    xm = np.array([0, 1])
    xn = np.array([0, 0])
    zmns = np.zeros((2, n_surfaces))
    zmns[1] = np.linspace(0, 1, n_surfaces) * minor_radius
    return TestWout(rmnc=rmnc, zmns=zmns, xm=xm, xn=xn, lasym=False)


def test_evaluate_rphiz_on_toroidal_grid(torus_wout):
    """Test the evaluate_rphiz_on_toroidal_grid function."""
    s_theta_phi = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, 1, 5),
            jnp.linspace(0, 2 * jnp.pi, 6),
            jnp.linspace(0, 2 * jnp.pi, 7),
            indexing="ij",
        ),
        axis=-1,
    )
    assert s_theta_phi.shape == (5, 6, 7, 3)
    rphiz = evaluate_rphiz_on_toroidal_grid(torus_wout, s_theta_phi)
    assert rphiz.shape == (5, 6, 7, 3)
    # some of them will be NaN, but hopefully not all
    assert np.any(np.isfinite(rphiz))

    phi = s_theta_phi[..., 2]
    np.testing.assert_allclose(rphiz[..., 1], phi, rtol=0, atol=1e-16)

    rho = s_theta_phi[..., 0]
    theta = s_theta_phi[..., 1]
    major_radius = 2.0
    minor_radius = 0.5
    r_expected = major_radius + rho * jnp.cos(theta) * minor_radius
    np.testing.assert_allclose(rphiz[..., 0], r_expected, rtol=0, atol=1e-16)