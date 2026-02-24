import jax
import jax.numpy as jnp
import jaxtyping as jt
from raytrax import bessel

ScalarFloat = float | jt.Float[jax.Array, " "]


def maxwell_juettner_distribution(
    lorentz_factor: ScalarFloat,
    thermal_velocity: ScalarFloat,
) -> float:
    """Compute the Maxwell-Juettner distribution function.

    Normalized so that ∫ f(γ(u)) d³u = 1, where d³u = d³p/(m₀c)³.

    Uses the scaled Bessel function kve(2, mu) = K2(mu)*exp(mu) to avoid
    underflow for large mu (low temperature). The distribution is rewritten as:
        f = mu / (4π * K2_scaled) * exp(-mu * (gamma - 1))

    Args:
        lorentz_factor: Lorentz factor of the particle (gamma = 1 / sqrt(1 - v^2/c^2))
        thermal_velocity: Electron thermal velocity normalized to the speed of light
            (v_th / c)

    Returns:
        The value of the Maxwell-Juettner distribution function at the given Lorentz
        factor.
    """
    # mu = m_0 c^2 / T_e = m_0 c^2 / (1/2 m_0 v_th^2) = 2 / (v_th / c)^2
    mu = 2 / thermal_velocity**2
    K2_scaled = bessel.kve_jax(2, mu)
    return mu / (4 * jnp.pi * K2_scaled) * jnp.exp(-mu * (lorentz_factor - 1))


maxwell_juettner_distribution_dgamma = jax.grad(
    maxwell_juettner_distribution, argnums=0
)
