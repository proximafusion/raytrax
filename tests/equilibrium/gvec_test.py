# ruff: noqa: E402
"""Unit tests for the GVEC equilibrium importer.

Since ``pygvec`` is an optional dependency that may not be installed in CI,
we test the importer against a minimal mock ``State`` object that satisfies
the same interface as ``gvec.State.evaluate()``.

The mock represents a tokamak-like circular torus with major radius R0 and
minor radius a, for which all quantities have exact analytic forms.  This
lets us verify the coordinate transformations and the Jacobian integration
independently of the actual GVEC solver.
"""

import sys
import types

# Create a dummy 'gvec' module so the importer doesn't raise ImportError
dummy_gvec = types.ModuleType("gvec")
sys.modules["gvec"] = dummy_gvec

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from raytrax.equilibrium.gvec import (
    GvecGridResolution,
    _compute_dvolume_drho_gvec,
    _nfp_from_state,
    magnetic_configuration_from_gvec,
)
from raytrax.equilibrium.interpolate import CylindricalGridResolution

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Analytic circular torus mock
# ---------------------------------------------------------------------------
# Geometry: R(rho, theta) = R0 + a*rho*cos(theta), Z(rho, theta) = a*rho*sin(theta)
# Jacobian sqrt(g) = a^2 * rho * (R0 + a*rho*cos(theta))
# dV/drho = 2*pi * nfp * int_0^{2pi/nfp} dzeta * int_0^{2pi} dtheta * |sqrt(g)|
#         = 2*pi * nfp * (2*pi/nfp) * 2*pi * a^2 * rho * R0   (leading order in a/R0)
#         = (2*pi)^2 * 2*pi * a^2 * rho * R0   ... but we use numerical integration below.
# Exact: dV/drho = (2*pi)^2 * a^2 * rho * R0  (all phi values contribute equally for axisymmetry)

R0 = 10.0  # major radius [m]
a = 1.0  # minor radius [m]
NFP = 5


class _MockGvecState:
    """Minimal mock of gvec.State for testing purposes.

    Implements a circular torus with major radius R0 and minor radius a.
    GVEC coordinate convention: X1=R, X2=Z, zeta=toroidal angle (opposite sign to VMEC phi).
    B field chosen as a simple analytically known form: B = B0 / R * hat(phi).
    """

    nfp: int = NFP
    B0: float = 5.0  # [T]

    def evaluate(self, quantity: str, rho, theta, zeta) -> np.ndarray:
        rho = np.asarray(rho)
        theta = np.asarray(theta)
        zeta = np.asarray(zeta)

        # Circular torus geometry
        R = R0 + a * rho * np.cos(theta)
        Z = a * rho * np.sin(theta)

        if quantity == "X1":
            return R
        elif quantity == "X2":
            return Z
        elif quantity == "sqrtg":
            # Jacobian of (rho, theta, zeta) -> (R, Z, phi)
            return a**2 * rho * R
        elif quantity == "B1":
            # B_R = 0 for pure toroidal field
            return np.zeros_like(rho)
        elif quantity == "B2":
            # B_Z = 0 for pure toroidal field
            return np.zeros_like(rho)
        elif quantity == "B3":
            # B_zeta = B0 * R / R (normalized), but in GVEC B3 is B_zeta component
            # For a pure toroidal field: B_phi = B0 / R in physical units
            # B3 in flux coords: B^zeta * g_zz ... simplified here to B0/R
            return self.B0 / R
        else:
            raise ValueError(f"Unknown quantity: {quantity!r}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_nfp_from_state_attribute():
    """_nfp_from_state reads nfp from the standard attribute."""
    state = _MockGvecState()
    assert _nfp_from_state(state) == NFP


def test_nfp_from_state_fallback():
    """_nfp_from_state also accepts nfperiod (older GVEC versions)."""

    class OldState:
        nfperiod = 3

    assert _nfp_from_state(OldState()) == 3


def test_nfp_from_state_missing():
    """_nfp_from_state raises AttributeError if no known attribute exists."""

    class BadState:
        pass

    with pytest.raises(AttributeError, match="nfp"):
        _nfp_from_state(BadState())


def test_dvolume_drho_circular_torus():
    """Numerical Jacobian integral matches analytic formula for a circular torus.

    For a circular torus (axisymmetric) with major radius R0 and minor radius a:
        dV/drho = (2*pi)^2 * a^2 * rho * R0   (exact, no a/R0 approximation needed
        since the phi integral of cos(theta) vanishes).
    """
    state = _MockGvecState()
    rho_1d = np.linspace(0.0, 1.0, 20)
    n_theta = 60
    n_zeta = 30

    dv = _compute_dvolume_drho_gvec(state, rho_1d, n_theta, n_zeta, NFP)

    expected = (2 * np.pi) ** 2 * a**2 * rho_1d * R0
    # Trapezoidal rule should be accurate to < 1% for these smooth integrands.
    np.testing.assert_allclose(dv, expected, rtol=1e-2)


def test_magnetic_configuration_from_gvec_shape():
    """magnetic_configuration_from_gvec returns arrays with correct shapes."""
    state = _MockGvecState()
    grid = GvecGridResolution(
        cylindrical=CylindricalGridResolution(
            n_r=10, n_z=12, n_phi=8, n_rho_profile=20
        ),
        n_rho=8,
        n_theta=10,
    )
    mag = magnetic_configuration_from_gvec(state, nfp=NFP, grid=grid)

    n_r, n_phi, n_z = 10, 8, 12
    assert mag.rphiz.shape == (n_r, n_phi, n_z, 3)
    assert mag.magnetic_field.shape == (n_r, n_phi, n_z, 3)
    assert mag.rho.shape == (n_r, n_phi, n_z)
    assert mag.rho_1d.shape == (20,)
    assert mag.dvolume_drho.shape == (20,)
    assert mag.nfp == NFP
    assert mag.is_stellarator_symmetric is True


def test_magnetic_configuration_from_gvec_rho_range():
    """rho values on the cylindrical grid are in [0, rho_max]."""
    state = _MockGvecState()
    grid = GvecGridResolution(
        cylindrical=CylindricalGridResolution(
            n_r=10, n_z=12, n_phi=8, n_rho_profile=20
        ),
        n_rho=8,
        n_theta=10,
        rho_max=1.2,
    )
    mag = magnetic_configuration_from_gvec(state, nfp=NFP, grid=grid)

    rho_finite = jnp.where(jnp.isfinite(mag.rho), mag.rho, 0.0)
    assert float(jnp.max(rho_finite)) <= 1.21  # small margin for interpolation


def test_magnetic_configuration_from_gvec_toroidal_angle_sign():
    """phi in rphiz is correctly mapped to [0, phi_max] despite GVEC sign convention."""
    state = _MockGvecState()
    grid = GvecGridResolution(
        cylindrical=CylindricalGridResolution(n_r=5, n_z=5, n_phi=4, n_rho_profile=10),
        n_rho=5,
        n_theta=6,
    )
    mag = magnetic_configuration_from_gvec(state, nfp=NFP, grid=grid)

    phi_vals = mag.rphiz[..., 1]

    # phi should be in [0, pi/nfp]
    phi_max = np.pi / NFP
    assert float(jnp.min(phi_vals)) >= -1e-10
    np.testing.assert_allclose(float(jnp.max(phi_vals)), phi_max, rtol=1e-5)


def test_missing_pygvec_raises_helpful_error():
    """ImportError for missing pygvec gives a helpful install hint."""
    import sys

    del sys.modules["gvec"]

    with pytest.raises(ImportError, match="pip install raytrax\\[gvec\\]"):
        magnetic_configuration_from_gvec(_MockGvecState(), nfp=NFP)

    # restore the dummy for the rest of the run if needed
    sys.modules["gvec"] = dummy_gvec
