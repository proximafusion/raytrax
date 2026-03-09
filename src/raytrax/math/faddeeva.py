"""Plasma dispersion function Z(w) and its derivative, computed via the Faddeeva function."""

import jax
import jax.numpy as jnp
from scipy.special import wofz as scipy_wofz


@jax.custom_jvp
def wofz_jax(z):
    """Compute the Faddeeva function w(z) for a complex argument z.

    Parameters:
    z : complex
        The complex argument for which to compute the Faddeeva function.

    Returns:
    complex
        The value of the Faddeeva function at z.
    """
    # Determine shape and dtype
    shape = jnp.shape(z)
    dtype = jnp.complex128

    # Use pure_callback for scalar or batched complex values
    return jax.pure_callback(
        scipy_wofz,
        jax.ShapeDtypeStruct(shape=shape, dtype=dtype),
        z,
        vmap_method="sequential",
    )


@wofz_jax.defjvp
def wofz_jax_jvp(primals, tangents):
    """Custom JVP rule for the Faddeeva function."""
    (z,) = primals
    (dz,) = tangents
    w_z = wofz_jax(z)
    w_prime = 2j / jnp.sqrt(jnp.pi) - 2 * z * w_z
    return w_z, w_prime * dz


def plasma_dispersion_function(z):
    """Compute the plasma dispersion function Z for a complex argument z.

    Args:
        z: The complex argument for which to compute the plasma dispersion function.
    """
    return 1j * jnp.sqrt(jnp.pi) * wofz_jax(z)


def plasma_dispersion_function_derivative(z):
    """Compute the derivative of the plasma dispersion function Z'.

    Args:
        z: The complex argument for which to compute the derivative.
    """
    return -2 * (1 + z * plasma_dispersion_function(z))
