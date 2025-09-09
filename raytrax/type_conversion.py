"""Functions for converting between different data types in Raytrax."""

import jax.numpy as jnp

from .ray import RayState
from .types import BeamProfile, RadialProfile


def ray_states_to_beam_profile(states: list[RayState]) -> BeamProfile:
    """Convert a list of RayState objects to a BeamProfile object."""
    positions = jnp.array([state.position for state in states])
    arc_lengths = jnp.array([state.arc_length for state in states])
    refractive_indices = jnp.array([state.refractive_index for state in states])
    optical_depths = jnp.array([state.optical_depth for state in states])
    return BeamProfile(
        position=positions,
        arc_length=arc_lengths,
        refractive_index=refractive_indices,
        optical_depth=optical_depths,
    )


def ray_states_to_radial_profile(states: list[RayState]) -> RadialProfile:
    """Convert a list of RayState objects to a RadialProfile object."""
    # Placeholder implementation: in a real implementation, this would project
    # the deposition profile onto the radial coordinate.
    rho = jnp.array([jnp.linalg.norm(state.position) for state in states])
    return RadialProfile(rho=rho)
