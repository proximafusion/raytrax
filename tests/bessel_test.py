import jax
import jax.numpy as jnp
import numpy as np
import pytest
from raytrax.bessel import jv_jax, kv_jax, kve_jax
from scipy.special import jv as scipy_jv
from scipy.special import kv as scipy_kv
from scipy.special import kve as scipy_kve

jax.config.update("jax_enable_x64", True)
_MACHINE_PRECISION = float(jnp.finfo(float).eps)


@pytest.mark.parametrize("v", [0, 1, 5, -1])
@pytest.mark.parametrize("z", [0.0, 1.0, 2.0, 5.0, 10.0])
def test_jv_jax_real(v, z):
    z_complex = jnp.float64(z)
    np.testing.assert_allclose(
        jv_jax(v, z_complex), scipy_jv(v, z_complex), rtol=0, atol=_MACHINE_PRECISION
    )


@pytest.mark.parametrize("v", [0, 1, 5])
def test_jv_jax_derivative(v):
    z = jnp.linspace(-10, 10, 2000, dtype=jnp.float64)
    dx = z[1] - z[0]

    jv_scalar = lambda zi: jv_jax(v, zi)
    value = jv_scalar(z)

    derivative = jax.vmap(jax.grad(jv_scalar))(z)

    expected_derivative = (value[2:] - value[:-2]) / (2 * dx)

    np.testing.assert_allclose(derivative[1:-1], expected_derivative, rtol=0, atol=0.01)


@pytest.mark.parametrize("v", [0, 1, 5, -1])
@pytest.mark.parametrize("z", [1.0, 2.0, 5.0, 10.0])  # Note: kv not defined at z=0
def test_kv_jax_real(v, z):
    z_complex = jnp.float64(z)
    np.testing.assert_allclose(
        kv_jax(v, z_complex), scipy_kv(v, z_complex), rtol=0, atol=_MACHINE_PRECISION
    )


@pytest.mark.parametrize("v", [0, 1, 5])
def test_kv_jax_derivative(v):
    # Avoiding small z values where kv can diverge
    z = jnp.linspace(0.5, 10, 2000, dtype=jnp.float64)
    dx = z[1] - z[0]

    kv_scalar = lambda zi: kv_jax(v, zi)
    value = kv_scalar(z)

    derivative = jax.vmap(jax.grad(kv_scalar))(z)

    # Finite difference for comparison
    expected_derivative = (value[2:] - value[:-2]) / (2 * dx)

    # Check only for the inner points and with relatively loose tolerance
    np.testing.assert_allclose(derivative[1:-1], expected_derivative, rtol=1e-3, atol=0)


@pytest.mark.parametrize("v", [0, 1, 5, -1])
@pytest.mark.parametrize("z", [1.0, 2.0, 5.0, 10.0])  # Note: kve not defined at z=0
def test_kve_jax_real(v, z):
    z_complex = jnp.float64(z)
    np.testing.assert_allclose(
        kve_jax(v, z_complex), scipy_kve(v, z_complex), rtol=0, atol=_MACHINE_PRECISION
    )


@pytest.mark.parametrize("v", [0, 1, 5])
def test_kve_jax_derivative(v):
    # Test with a range that includes large values where kve is useful
    z = jnp.linspace(0.5, 20, 2000, dtype=jnp.float64)
    dx = z[1] - z[0]

    kve_scalar = lambda zi: kve_jax(v, zi)
    value = kve_scalar(z)

    derivative = jax.vmap(jax.grad(kve_scalar))(z)

    # Finite difference for comparison
    expected_derivative = (value[2:] - value[:-2]) / (2 * dx)

    # Check only for the inner points with relatively loose tolerance
    # Higher order Bessel functions have steeper derivatives near small z, so need higher atol
    np.testing.assert_allclose(derivative[1:-1], expected_derivative, rtol=1e-3, atol=5e2)
