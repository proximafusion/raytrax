"""Relativistic Maxwell-Jüttner electron distribution function and its energy derivative."""

import jax
import jax.numpy as jnp

from raytrax.math import bessel
from raytrax.types import ScalarFloat


def maxwell_juettner_distribution(
    lorentz_factor: ScalarFloat,
    thermal_velocity: ScalarFloat,
    K2_scaled: ScalarFloat | None = None,
) -> ScalarFloat:
    """Compute the Maxwell-Juettner distribution function.

    Normalized so that int f(gamma(u)) d^3u = 1, where d^3u = d^3p/(m_0 c)^3.

    Uses the scaled Bessel function kve(2, mu) = K_2(mu)*exp(mu) to avoid
    underflow for large mu (low temperature). The distribution is rewritten as:
        f = mu / (4*pi * K2_scaled) * exp(-mu * (gamma - 1))

    Args:
        lorentz_factor: Lorentz factor of the particle (gamma = 1 / sqrt(1 - v^2/c^2))
        thermal_velocity: Electron thermal velocity normalized to the speed of light
            (v_th / c)
        K2_scaled: Optional pre-computed kve(2, \\mu) = K_2(\\mu) * exp(\\mu), where
            \\mu = 2/v_th^2.  When calling inside a vmap loop where thermal_velocity
            is constant, compute this once before the loop to avoid evaluating the
            Bessel function on every iteration::

                mu = 2 / thermal_velocity**2
                K2_scaled = bessel.kve_jax(2, mu)

            If None (default), K2_scaled is computed internally.

    Returns:
        The value of the Maxwell-Juettner distribution function at the given Lorentz
        factor.
    """
    # mu = m_0 c^2 / T_e = m_0 c^2 / (1/2 m_0 v_th^2) = 2 / (v_th / c)^2
    mu = 2 / thermal_velocity**2
    k2 = bessel.kve_jax(2, mu) if K2_scaled is None else K2_scaled
    return mu / (4 * jnp.pi * k2) * jnp.exp(-mu * (lorentz_factor - 1))


maxwell_juettner_distribution_dgamma = jax.grad(
    maxwell_juettner_distribution, argnums=0
)


def maxwell_juettner_distribution_dgamma_precomputed(
    lorentz_factor: ScalarFloat,
    thermal_velocity: ScalarFloat,
    K2_scaled: ScalarFloat,
) -> ScalarFloat:
    """df/d\\gamma of the Maxwell-Juettner distribution with pre-computed K2_scaled.

    Equivalent to calling maxwell_juettner_distribution_dgamma(lorentz_factor,
    thermal_velocity, K2_scaled).  The separate name signals the pre-computation
    pattern: when thermal_velocity is constant across a vmap loop, compute K2_scaled
    once before the loop::

        mu = 2 / thermal_velocity**2
        K2_scaled = bessel.kve_jax(2, mu)

    and pass it here so that the Bessel evaluation is not repeated for every
    loop iteration.

    Args:
        lorentz_factor: Lorentz factor \\gamma.
        thermal_velocity: Electron thermal velocity normalized to c (v_th/c).
        K2_scaled: Pre-computed kve(2, \\mu) = K_2(\\mu) * exp(\\mu), where
            \\mu = 2/v_th^2.

    Returns:
        df/d\\gamma = -\\mu * f(\\gamma)
    """
    return maxwell_juettner_distribution_dgamma(
        lorentz_factor, thermal_velocity, K2_scaled
    )
