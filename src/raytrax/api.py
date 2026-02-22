"""Main functions to interact with Raytrax."""

import jax.numpy as jnp

from .interpolate import (
    build_magnetic_field_interpolator,
    build_rho_interpolator,
    build_electron_density_profile_interpolator,
    build_electron_temperature_profile_interpolator,
    MagneticConfiguration,
)
from .ray import RaySetting
from .solver import trace_jitted
from .types import (
    Beam,
    BeamProfile,
    Interpolators,
    RadialProfile,
    RadialProfiles,
    TraceResult,
)


def trace(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
    beam: Beam,
) -> TraceResult:
    """Trace a single beam through the plasma.

    Args:
        magnetic_configuration: Magnetic configuration with gridded data
        radial_profiles: Radial profiles of plasma parameters
        beam: Beam initial conditions (position, direction, frequency, mode)

    Returns:
        TraceResult with beam profile and radial deposition profile.
    """
    setting = RaySetting(frequency=beam.frequency, mode=beam.mode)

    interpolators = Interpolators(
        magnetic_field=build_magnetic_field_interpolator(magnetic_configuration),
        rho=build_rho_interpolator(magnetic_configuration),
        electron_density=build_electron_density_profile_interpolator(radial_profiles),
        electron_temperature=build_electron_temperature_profile_interpolator(
            radial_profiles
        ),
        is_axisymmetric=magnetic_configuration.is_axisymmetric,
    )

    result = trace_jitted(
        jnp.asarray(beam.position),
        jnp.asarray(beam.direction),
        setting,
        interpolators,
        magnetic_configuration.nfp,
        magnetic_configuration.rho_1d,
        magnetic_configuration.dvolume_drho,
    )

    # Trim padded buffer to valid entries
    n = int(jnp.sum(jnp.isfinite(result.arc_length)).item())

    beam_profile = BeamProfile(
        position=result.ode_state[:n, :3],
        arc_length=result.arc_length[:n],
        refractive_index=result.ode_state[:n, 3:6],
        optical_depth=result.ode_state[:n, 6],
        absorption_coefficient=result.absorption_coefficient[:n],
        electron_density=result.electron_density[:n],
        electron_temperature=result.electron_temperature[:n],
        magnetic_field=result.magnetic_field[:n],
        normalized_effective_radius=result.normalized_effective_radius[:n],
        linear_power_density=result.linear_power_density[:n],
    )
    radial_profile = RadialProfile(
        rho=result.normalized_effective_radius[:n],
        volumetric_power_density=result.volumetric_power_density[:n],
    )
    return TraceResult(beam_profile=beam_profile, radial_profile=radial_profile)
