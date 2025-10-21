"""Main functions to interact with Raytrax."""

import jax.numpy as jnp

from .interpolate import (
    build_magnetic_field_interpolator,
    build_rho_interpolator,
    build_electron_density_profile_interpolator,
    build_electron_temperature_profile_interpolator,
    cylindrical_grid_for_equilibrium,
)
from .ray import RaySetting, RayState
from .solver import solve, compute_additional_quantities
from .type_conversion import ray_states_to_beam_profile, ray_states_to_radial_profile
from .types import (
    Beam,
    EquilibriumInterpolator,
    RadialProfiles,
    TracingResult,
    WoutLike,
)


def get_interpolator_for_equilibrium(equilibrium: WoutLike) -> EquilibriumInterpolator:
    """Generate interpolators for the given MHD equilibrium.

    Args:
        equilibrium: an MHD equilibrium compatible with `vmecpp.VmecWOut`

    Returns:
        An EquilibriumInterpolator object containing interpolation data.
    """
    # TODO add settings for grid resolution
    interpolated_array = cylindrical_grid_for_equilibrium(
        equilibrium=equilibrium, n_rho=40, n_theta=45, n_phi=50, n_r=45, n_z=55
    )
    rphiz = interpolated_array[..., :3]
    rho = interpolated_array[..., 3]
    magnetic_field = interpolated_array[..., 4:]
    return EquilibriumInterpolator(
        rphiz=rphiz, 
        magnetic_field=magnetic_field, 
        rho=rho, 
        equilibrium=equilibrium
    )


def trace(
    equilibrium_interpolator: EquilibriumInterpolator,
    radial_profiles: RadialProfiles,
    beam: Beam,
) -> TracingResult:
    """Solve the ray tracing equations for a given beam given an MHD equilibrium.

    Args:
        equilibrium: an MHD equilibrium compatible with `vmecpp.VmecWOut`
        beam: A Beam object containing the initial conditions of the beam.

    Returns:
        A TracingResult object containing the results of the tracing.
    """
    # Use the beam direction as the initial refractive index direction
    initial_state = RayState(
        position=jnp.asarray(beam.position),
        refractive_index=jnp.asarray(beam.direction),
        optical_depth=jnp.array(0.0),
        arc_length=jnp.array(0.0),
    )
    setting = RaySetting(frequency=beam.frequency, mode=beam.mode)
    magnetic_field_interpolator = build_magnetic_field_interpolator(
        equilibrium_interpolator
    )
    rho_interpolator = build_rho_interpolator(equilibrium_interpolator)
    electron_density_profile_interpolator = build_electron_density_profile_interpolator(radial_profiles)
    electron_temperature_profile_interpolator = build_electron_temperature_profile_interpolator(radial_profiles)
    ray_states = solve(
        state=initial_state,
        setting=setting,
        magnetic_field_interpolator=magnetic_field_interpolator,
        rho_interpolator=rho_interpolator,
        electron_density_profile_interpolator=electron_density_profile_interpolator,
        electron_temperature_profile_interpolator=electron_temperature_profile_interpolator,
    )
    additional_quantities = compute_additional_quantities(
        ray_states=ray_states,
        setting=setting,
        magnetic_field_interpolator=magnetic_field_interpolator,
        rho_interpolator=rho_interpolator,
        electron_density_profile_interpolator=electron_density_profile_interpolator,
        electron_temperature_profile_interpolator=electron_temperature_profile_interpolator,
    )
    beam_profile = ray_states_to_beam_profile(ray_states, additional_quantities)
    radial_profile = ray_states_to_radial_profile(
        ray_states, additional_quantities, equilibrium_interpolator.equilibrium
    )
    return TracingResult(beam_profile=beam_profile, radial_profile=radial_profile)
