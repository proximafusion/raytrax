"""Fundamental plasma quantities: electron plasma frequency, cyclotron frequency, and thermal velocity."""

import jax.numpy as jnp
from scipy import constants

from raytrax.types import ScalarFloat


def electron_plasma_frequency(electron_density_1e20_per_m3: ScalarFloat) -> ScalarFloat:
    """Compute the electron plasma frequency.

    Args:
        electron_density_1e20_per_m3: The electron density in 10^20 m^-3.

    Returns:
        The plasma frequency in Hz (not GHz!).
    """
    electron_density = electron_density_1e20_per_m3 * 1e20
    return jnp.sqrt(
        constants.e**2 * electron_density / (constants.m_e * constants.epsilon_0)
    ) / (2 * jnp.pi)


def electron_cyclotron_frequency(magnetic_field_strength: ScalarFloat) -> ScalarFloat:
    """Compute the cold electron cyclotron frequency.

    Args:
        magnetic_field_strength: The magnetic field strength in Tesla.

    Returns:
        The cyclotron frequency in Hz (not GHz!).
    """
    return constants.e * magnetic_field_strength / constants.m_e / (2 * jnp.pi)


def normalized_electron_thermal_velocity(
    electron_temperature_keV: ScalarFloat,
) -> ScalarFloat:
    """Compute the electron thermal velocity normalized to the speed of light.

    Args:
        electron_temperature_keV: The electron temperature in keV.

    Returns:
        The dimensionless thermal velocity normalized to the speed of light.
    """
    # Thermal velocity is sqrt(2 * T / m_e) (where T is in Joules) since T = 1/2 m v^2
    # we normalize to the speed of light
    T = electron_temperature_keV * constants.e * 1e3
    v = jnp.sqrt(2 * T / constants.m_e)
    return v / constants.c
