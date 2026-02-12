"""Main functions to interact with Raytrax."""

import jax.numpy as jnp

from .interpolate import (
    build_magnetic_field_interpolator,
    build_rho_interpolator,
    build_electron_density_profile_interpolator,
    build_electron_temperature_profile_interpolator,
)
from .ray import RaySetting
from .solver import trace_jitted
from .types import (
    Beam,
    BeamProfile,
    Interpolators,
    MagneticConfiguration,
    RadialProfile,
    RadialProfiles,
    TracingResult,
)


def trace(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
    beam: Beam,
) -> TracingResult:
    """Trace a single beam through the plasma.

    Args:
        magnetic_configuration: Magnetic configuration with gridded data
        radial_profiles: Radial profiles of plasma parameters
        beam: Beam initial conditions (position, direction, frequency, mode)

    Returns:
        TracingResult with beam profile and radial deposition profile.
    """
    setting = RaySetting(frequency=beam.frequency, mode=beam.mode)

    interpolators = Interpolators(
        magnetic_field=build_magnetic_field_interpolator(magnetic_configuration),
        rho=build_rho_interpolator(magnetic_configuration),
        electron_density=build_electron_density_profile_interpolator(radial_profiles),
        electron_temperature=build_electron_temperature_profile_interpolator(
            radial_profiles
        ),
    )

    ts, ys, B_all, rho_all, ne_all, te_all, alpha_all, P_all, dP_dV = trace_jitted(
        jnp.asarray(beam.position),
        jnp.asarray(beam.direction),
        setting,
        interpolators,
        magnetic_configuration.nfp,
        magnetic_configuration.rho_1d,
        magnetic_configuration.dvolume_drho,
    )

    # Trim padded buffer to valid entries
    n = int(jnp.sum(jnp.isfinite(ts)).item())

    beam_profile = BeamProfile(
        position=ys[:n, :3],
        arc_length=ts[:n],
        refractive_index=ys[:n, 3:6],
        optical_depth=ys[:n, 6],
        absorption_coefficient=alpha_all[:n],
        electron_density=ne_all[:n],
        electron_temperature=te_all[:n],
        magnetic_field=B_all[:n],
        normalized_effective_radius=rho_all[:n],
        linear_power_density=P_all[:n],
    )
    radial_profile = RadialProfile(
        rho=rho_all[:n],
        volumetric_power_density=dP_dV[:n],
    )
    return TracingResult(beam_profile=beam_profile, radial_profile=radial_profile)
