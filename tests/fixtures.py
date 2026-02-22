from dataclasses import dataclass
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vmecpp import VmecWOut

from raytrax.data import get_w7x_wout
from raytrax.interpolate import MagneticConfiguration


@dataclass
class TestWout:
    rmnc: jax.Array
    zmns: jax.Array
    xm: jax.Array
    xn: jax.Array
    gmnc: jax.Array
    gmns: jax.Array
    bsupumnc: jax.Array
    bsupvmnc: jax.Array
    xm_nyq: jax.Array
    xn_nyq: jax.Array
    ns: int
    nfp: int
    lasym: bool = False


@pytest.fixture
def torus_wout():
    """Fixture for a torus shaped Wout-like object."""
    n_surfaces = 5
    major_radius = 2.0
    minor_radius = 0.5
    rmnc = np.zeros((2, n_surfaces))
    rmnc[0] = major_radius
    rmnc[1] = np.sqrt(np.linspace(0, 1, n_surfaces)) * minor_radius

    xm = np.array([0, 1])
    xn = np.array([0, 0])
    zmns = np.zeros((2, n_surfaces))
    zmns[1] = np.sqrt(np.linspace(0, 1, n_surfaces)) * minor_radius

    xm_nyq = np.array([0, 1])
    xn_nyq = np.array([0, 0])
    bsupumnc = np.zeros((2, n_surfaces))
    bsupvmnc = np.zeros((2, n_surfaces))
    bsupvmnc[0] = 0.7

    # Add gmnc and gmns arrays for volume calculations
    gmnc = np.zeros((2, n_surfaces))
    gmnc[0] = np.linspace(0.1, 1.0, n_surfaces)  # g_{0,0} mode varies radially
    gmns = np.zeros((2, n_surfaces))

    return TestWout(
        rmnc=jnp.array(rmnc),
        zmns=jnp.array(zmns),
        xm=jnp.array(xm),
        xn=jnp.array(xn),
        gmnc=jnp.array(gmnc),
        gmns=jnp.array(gmns),
        bsupumnc=jnp.array(bsupumnc),
        bsupvmnc=jnp.array(bsupvmnc),
        xm_nyq=jnp.array(xm_nyq),
        xn_nyq=jnp.array(xn_nyq),
        ns=5,
        nfp=1,
        lasym=False,
    )


@pytest.fixture
def w7x_wout():
    """Fixture for the W7-X equilibrium."""
    return get_w7x_wout()


@pytest.fixture
def w7x_travis_wout():
    """Fixture for the W7-X equilibrium from TRAVIS NetCDF file.

    This loads the exact same equilibrium file that TRAVIS uses, allowing
    direct comparison between TRAVIS and raytrax calculations.

    Set the TRAVIS_W7X_WOUT environment variable to point to the wout file:
        export TRAVIS_W7X_WOUT=/path/to/wout-w7x-hm13.nc

    Raises:
        pytest.skip: If TRAVIS_W7X_WOUT is not set or file doesn't exist
    """
    wout_path_str = os.environ.get("TRAVIS_W7X_WOUT")
    if not wout_path_str:
        pytest.skip(
            "TRAVIS_W7X_WOUT environment variable not set. "
            "Set it to the path of the TRAVIS W7-X wout file for integration tests."
        )

    wout_path = Path(wout_path_str).expanduser()
    if not wout_path.exists():
        pytest.skip(f"TRAVIS W7-X equilibrium file not found at {wout_path}")

    return VmecWOut.from_wout_file(str(wout_path))


@pytest.fixture
def tokamak_magnetic_configuration():
    """Analytic tokamak MagneticConfiguration with circular cross-section.

    B_phi = B0 * R0 / R  (vacuum toroidal field, 1/R dependence)
    B_R = B_Z = 0
    rho = sqrt((R - R0)^2 + Z^2) / a

    Parameters: R0=3.0 m, a=1.0 m, B0=2.5 T.
    """
    R0, a, B0 = 3.0, 1.0, 2.5
    n_R, n_Z = 50, 60

    R_grid = jnp.linspace(R0 - 1.5 * a, R0 + 1.5 * a, n_R)
    Z_grid = jnp.linspace(-1.5 * a, 1.5 * a, n_Z)
    R_2d, Z_2d = jnp.meshgrid(R_grid, Z_grid, indexing="ij")

    B_R = jnp.zeros_like(R_2d)
    B_phi = B0 * R0 / R_2d
    B_Z = jnp.zeros_like(R_2d)

    rho_2d = jnp.sqrt((R_2d - R0) ** 2 + Z_2d**2) / a

    phi_grid = jnp.array([0.0])
    rphiz = jnp.stack(
        jnp.meshgrid(R_grid, phi_grid, Z_grid, indexing="ij"),
        axis=-1,
    )
    magnetic_field = jnp.stack([B_R, B_phi, B_Z], axis=-1)[:, jnp.newaxis, :, :]
    rho = rho_2d[:, jnp.newaxis, :]

    rho_1d = jnp.linspace(0, 1, 200)
    dvolume_drho = 4.0 * jnp.pi**2 * R0 * a**2 * rho_1d

    return MagneticConfiguration(
        rphiz=rphiz,
        magnetic_field=magnetic_field,
        rho=rho,
        nfp=1,
        is_stellarator_symmetric=False,
        rho_1d=rho_1d,
        dvolume_drho=dvolume_drho,
        is_axisymmetric=True,
    )
