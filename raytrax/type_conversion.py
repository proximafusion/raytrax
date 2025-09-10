"""Functions for converting between different data types in Raytrax."""

import jax.numpy as jnp

from .ray import RayState, RayQuantities
from .types import BeamProfile, RadialProfile


def ray_states_to_beam_profile(states: list[RayState], quantities: list[RayQuantities]) -> BeamProfile:
    """Convert a list of RayState objects to a BeamProfile object."""
    position = jnp.array([state.position for state in states])
    arc_length = jnp.array([state.arc_length for state in states])
    refractive_index = jnp.array([state.refractive_index for state in states])
    optical_depth = jnp.array([state.optical_depth for state in states])
    absorption_coefficient = jnp.array([q.absorption_coefficient for q in quantities])
    electron_density = jnp.array([q.electron_density for q in quantities])
    electron_temperature = jnp.array([q.electron_temperature for q in quantities])
    magnetic_field = jnp.array([q.magnetic_field for q in quantities])
    return BeamProfile(
        position=position,
        arc_length=arc_length,
        refractive_index=refractive_index,
        optical_depth=optical_depth,
        absorption_coefficient=absorption_coefficient,
        electron_density=electron_density,
        electron_temperature=electron_temperature,
        magnetic_field=magnetic_field,
    )


def ray_states_to_radial_profile(states: list[RayState], quantities: list[RayQuantities]) -> RadialProfile:
    """Convert a list of RayState objects to a RadialProfile object."""
    # Placeholder implementation: in a real implementation, this would project
    # the deposition profile onto the radial coordinate.
    rho = jnp.array([jnp.linalg.norm(state.position) for state in states])
    return RadialProfile(rho=rho)
