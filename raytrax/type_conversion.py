"""Functions for converting between different data types in Raytrax."""

import jax.numpy as jnp

from .fourier import dvolume_drho
from .ray import RayState, RayQuantities
from .types import BeamProfile, RadialProfile, WoutLike


def ray_states_to_beam_profile(
    states: list[RayState], quantities: list[RayQuantities]
) -> BeamProfile:
    """Convert a list of RayState objects to a BeamProfile object."""
    position = jnp.array([state.position for state in states])
    arc_length = jnp.array([state.arc_length for state in states])
    refractive_index = jnp.array([state.refractive_index for state in states])
    optical_depth = jnp.array([state.optical_depth for state in states])
    absorption_coefficient = jnp.array([q.absorption_coefficient for q in quantities])
    electron_density = jnp.array([q.electron_density for q in quantities])
    electron_temperature = jnp.array([q.electron_temperature for q in quantities])
    magnetic_field = jnp.array([q.magnetic_field for q in quantities])
    linear_power_density = jnp.array([q.linear_power_density for q in quantities])
    return BeamProfile(
        position=position,
        arc_length=arc_length,
        refractive_index=refractive_index,
        optical_depth=optical_depth,
        absorption_coefficient=absorption_coefficient,
        electron_density=electron_density,
        electron_temperature=electron_temperature,
        magnetic_field=magnetic_field,
        linear_power_density=linear_power_density,
    )


def ray_states_to_radial_profile(
    states: list[RayState], quantities: list[RayQuantities], equilibrium: WoutLike
) -> RadialProfile:
    """Convert a list of RayState objects to a RadialProfile object.

    Args:
        states: List of ray states along the trajectory
        quantities: List of ray quantities along the trajectory
        equilibrium: The MHD equilibrium (needed for volume calculation)

    Returns:
        RadialProfile
    """
    rho = jnp.array([q.normalized_effective_radius for q in quantities])
    dP_ds = jnp.array([q.linear_power_density for q in quantities])
    arc_length = jnp.array([state.arc_length for state in states])

    ds = jnp.diff(arc_length)
    drho_ds = jnp.diff(rho) / ds
    drho_ds_padded = jnp.pad(drho_ds, (1, 1), mode="edge")
    drho_ds_avg = 0.5 * (drho_ds_padded[:-1] + drho_ds_padded[1:])

    dV_drho = dvolume_drho(equilibrium, rho)
    dP_drho = dP_ds / jnp.abs(drho_ds_avg)
    dP_dV = dP_drho / dV_drho

    return RadialProfile(rho=rho, volumetric_power_density=dP_dV)
