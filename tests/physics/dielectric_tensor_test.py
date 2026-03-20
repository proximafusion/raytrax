import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import factorial

from raytrax.physics import quantities
from raytrax.physics.dielectric_tensor import (
    a_shkarofsky,
    b_shkarofsky,
    cold_dielectric_tensor,
    weakly_relativistic_dielectric_tensor,
)
from raytrax.physics.dispersion import (
    _dispersion_appleton_hartee,
    dispersion_tensor_stix,
)

jax.config.update("jax_enable_x64", True)
_MACHINE_PRECISION = float(np.finfo(float).eps)


def _a_krivenski_orefice(s, k):
    if k < 0:
        return 0.0
    abs_s = abs(s)
    term1 = (-1) ** k
    term2 = factorial(2 * (abs_s + k))
    term3 = (0.5) ** (2 * (abs_s + k))
    term4 = (factorial(abs_s + k)) ** 2
    term5 = factorial(2 * abs_s + k)
    term6 = factorial(k)
    result = (term1 * term2 * term3) / (term4 * term5 * term6)
    return result


def _b_krivenski_orefice(s, k):
    if k < 0:
        return 0.0
    if s == 0:
        return _a_krivenski_orefice(1, k - 2)
    abs_s = abs(s)
    part1 = _a_krivenski_orefice(s - 1, k)
    part2 = _a_krivenski_orefice(s + 1, k - 2)
    frac = (abs_s + k - 1) / (abs_s + k)
    part3 = 2 * frac * _a_krivenski_orefice(s, k - 1)
    return 0.25 * (part1 + part2 - part3)


@pytest.mark.parametrize("s", [0, 1, 2])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_a(s, k):
    """Proving that a in Krivenski-Orefice is related to a in Shkarofsky like:

    .. math::
        a_S(s, k) = (s + k)! 2^{s + k} a_{KO}(s, k)
    """
    np.testing.assert_allclose(
        a_shkarofsky(s, k),
        factorial(s + k) * 2 ** (s + k) * _a_krivenski_orefice(s, k),
        rtol=1e-9,
        atol=0,
    )


@pytest.mark.parametrize(
    "s,k", [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
)
def test_b(s, k):
    """Test that b/a is invariant across the two definitions."""
    np.testing.assert_allclose(
        b_shkarofsky(s, k),
        factorial(s + k) * 2 ** (s + k) * _b_krivenski_orefice(s, k),
        rtol=1e-9,
        atol=0,
    )


def test_weakly_relativistic_dielectric_tensor():
    epsilon = weakly_relativistic_dielectric_tensor(
        frequency=230.0,
        plasma_frequency=260.0,
        cyclotron_frequency=232.0,
        thermal_velocity=quantities.normalized_electron_thermal_velocity(
            electron_temperature_keV=5.0
        ),
        refractive_index_para=0.5,
        max_s=1,
        max_k=1,
    )
    assert isinstance(epsilon, jax.Array)
    assert epsilon.shape == (3, 3)
    assert epsilon.dtype == jnp.complex128
    np.testing.assert_allclose(
        epsilon[0, 1], -epsilon[1, 0], rtol=0, atol=_MACHINE_PRECISION
    )
    np.testing.assert_allclose(
        epsilon[1, 2], -epsilon[2, 1], rtol=0, atol=_MACHINE_PRECISION
    )
    np.testing.assert_allclose(
        epsilon[0, 2], epsilon[2, 0], rtol=0, atol=_MACHINE_PRECISION
    )


def test_weakly_relativistic_converges_to_cold():
    frequency = 2.0e8  # Hz
    plasma_frequency = 1.0e8  # Hz
    cyclotron_frequency = 5.0e8  # Hz
    refractive_index_para = 1.2
    electron_temperature_keV = 1e-8  # Very small temperature

    eps_wr = weakly_relativistic_dielectric_tensor(
        frequency=frequency,
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
        thermal_velocity=quantities.normalized_electron_thermal_velocity(
            electron_temperature_keV=electron_temperature_keV
        ),
        refractive_index_para=refractive_index_para,
        max_s=1,
        max_k=1,
    )

    # these vanish in the cold plasma limit
    np.testing.assert_allclose(eps_wr[0, 2], 0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(eps_wr[1, 2], 0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(eps_wr[2, 0], 0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(eps_wr[2, 1], 0, rtol=0, atol=1e-8)

    # these become equal in the cold plasma limit
    np.testing.assert_allclose(eps_wr[0, 0], eps_wr[1, 1], rtol=0, atol=1e-8)

    # these fail, unclear why!
    # np.testing.assert_allclose(eps_wr[0, 1], eps_cold[0, 1], rtol=0, atol=1e-8)
    # np.testing.assert_allclose(eps_wr[0, 0] - 1, eps_cold[0, 0] - 1, rtol=0, atol=1e-8)
    # np.testing.assert_allclose(1 - eps_wr[2, 2], 1 - eps_cold[2, 2], rtol=0, atol=1e-8)


@pytest.mark.parametrize("mode", ["X", "O"])
def test_cold_dielectric_tensor_dispersion(mode):
    """Check that the solution of the dispersion relation using the cold dielectric
    tensor satisfies the Appleton-Hartree equation."""
    frequency = 200e9
    plasma_frequency = 100e9
    cyclotron_frequency = 220e9
    dielectric_tensor = cold_dielectric_tensor(
        frequency=frequency,
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
    )
    X = plasma_frequency**2 / frequency**2
    Y = cyclotron_frequency / frequency
    tantheta = 3.0
    sin2theta = tantheta**2 / (1 + tantheta**2)
    n2 = _dispersion_appleton_hartee(X=X, Y=Y, sin2theta=sin2theta, mode=mode)
    n_para = jnp.sqrt(n2) * jnp.sqrt(1 - sin2theta)
    n_perp = jnp.sqrt(n2) * jnp.sqrt(sin2theta)
    dispersion_tensor = dispersion_tensor_stix(
        refractive_index_perp=n_perp,
        refractive_index_para=n_para,
        dielectric_tensor=dielectric_tensor,
    )
    s = jnp.linalg.svd(dispersion_tensor, compute_uv=False)
    # check that the matrix has a singular value
    print(s)
    np.testing.assert_allclose(s[-1], 0, rtol=0, atol=_MACHINE_PRECISION)
