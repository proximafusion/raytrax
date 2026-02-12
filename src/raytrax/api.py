"""Main functions to interact with Raytrax."""

import jax.numpy as jnp

from .fourier import dvolume_drho
from .interpolate import (
    build_magnetic_field_interpolator,
    build_rho_interpolator,
    build_electron_density_profile_interpolator,
    build_electron_temperature_profile_interpolator,
    cylindrical_grid_for_equilibrium,
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
    WoutLike,
)


def get_interpolator_for_equilibrium(
    equilibrium: WoutLike,
    magnetic_field_scale: float = 1.0,
) -> MagneticConfiguration:
    """Generate interpolators for the given MHD equilibrium.

    Args:
        equilibrium: an MHD equilibrium compatible with `vmecpp.VmecWOut`
        magnetic_field_scale: Factor to multiply all magnetic field values by.
            Use this to normalize the field strength, e.g. to match TRAVIS's
            ``B0_normalization_type at angle on magn.axis`` setting.

    Returns:
        A MagneticConfiguration object containing interpolation data.
    """
    # TODO add settings for grid resolution
    interpolated_array = cylindrical_grid_for_equilibrium(
        equilibrium=equilibrium, n_rho=40, n_theta=45, n_phi=50, n_r=45, n_z=55
    )
    rphiz = interpolated_array[..., :3]
    rho = interpolated_array[..., 3]
    magnetic_field = interpolated_array[..., 4:] * magnetic_field_scale

    # Compute volume derivative on 1D radial grid
    rho_1d = jnp.linspace(0, 1, 200)
    dv_drho = dvolume_drho(equilibrium, rho_1d)

    return MagneticConfiguration(
        rphiz=rphiz,
        magnetic_field=magnetic_field,
        rho=rho,
        nfp=equilibrium.nfp,
        stellarator_symmetric=not equilibrium.lasym,
        rho_1d=rho_1d,
        dvolume_drho=dv_drho,
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
