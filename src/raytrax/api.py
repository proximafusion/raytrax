"""Main functions to interact with Raytrax."""

import interpax
import jax
import jax.numpy as jnp
import jaxtyping as jt

from .equilibrium.interpolate import (
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
    TraceBuffers,
    TraceResult,
)

_N_INTERP = 2000  # dense arc-length samples used for smooth radial binning


def _bin_power_deposition(
    rho_grid: jt.Float[jax.Array, " nrho"],
    dvolume_drho: jt.Float[jax.Array, " nrho"],
    arc_length: jt.Float[jax.Array, " n"],
    rho_trajectory: jt.Float[jax.Array, " n"],
    optical_depth: jt.Float[jax.Array, " n"],
) -> jt.Float[jax.Array, " nrho"]:
    """Compute differentiable volumetric power deposition profile from ray trajectory.

    The ODE solver produces O(50) sparse trajectory points.  Directly binning
    them yields a jagged staircase profile.  Instead we:

    1. Sanitize inf padding (diffrax fills unused sol.ys slots with inf).
    2. Linearly interpolate rho(s) and τ(s) to _N_INTERP uniformly-spaced
       arc-length samples — no bandwidth choice needed, smoothness follows
       from density.
    3. Use exact overlap-based binning on the dense samples.
    """
    # --- 1. Sanitize inf padding -------------------------------------------
    # diffrax fills unused sol.ys slots with inf.  Push padded arc_length
    # entries to s_max + 1 (strictly beyond the query range of s_fine) so that
    # jnp.interp never picks up the padded rho/tau values.  tau_final is used
    # to fill padded optical_depth so dP = 0 at the boundary.
    tau_final = jnp.max(jnp.where(jnp.isfinite(optical_depth), optical_depth, 0.0))
    optical_depth = jnp.where(jnp.isfinite(optical_depth), optical_depth, tau_final)

    s_max = jnp.max(jnp.where(jnp.isfinite(arc_length), arc_length, 0.0))
    arc_length = jnp.where(jnp.isfinite(arc_length), arc_length, s_max + 1.0)

    # Any finite fill for rho at padded slots — those slots are never queried.
    rho_trajectory = jnp.where(jnp.isfinite(rho_trajectory), rho_trajectory, 0.0)

    # --- 2. Dense linear interpolation along arc length --------------------
    s_fine = jnp.linspace(arc_length[0], s_max, _N_INTERP)
    rho_fine = interpax.interp1d(s_fine, arc_length, rho_trajectory, method="linear")
    tau_fine = interpax.interp1d(s_fine, arc_length, optical_depth, method="linear")

    # --- 3. Exact overlap-based binning on dense samples -------------------
    # dP_i = exp(-τ_i) − exp(-τ_{i+1}), clamped to ≥ 0.
    dP = jnp.maximum(jnp.exp(-tau_fine[:-1]) - jnp.exp(-tau_fine[1:]), 0.0)

    edges = jnp.concatenate(
        [rho_grid[:1], 0.5 * (rho_grid[:-1] + rho_grid[1:]), rho_grid[-1:]]
    )
    bin_lo = edges[:-1]  # (nrho,)
    bin_hi = edges[1:]  # (nrho,)

    rho_a = rho_fine[:-1]  # (_N_INTERP-1,)
    rho_b = rho_fine[1:]
    seg_lo = jnp.minimum(rho_a, rho_b)[:, None]
    seg_hi = jnp.maximum(rho_a, rho_b)[:, None]

    overlap = jnp.maximum(
        0.0, jnp.minimum(seg_hi, bin_hi) - jnp.maximum(seg_lo, bin_lo)
    )  # (_N_INTERP-1, nrho)

    seg_len = (seg_hi - seg_lo)[:, 0]

    # For zero-length (tangential) segments, fall back to a hat kernel at the
    # midpoint so weights remain finite and differentiable.
    bin_width = rho_grid[1] - rho_grid[0]
    rho_mid = 0.5 * (rho_a + rho_b)
    hat = jnp.maximum(
        0.0, 1.0 - jnp.abs(rho_mid[:, None] - rho_grid[None, :]) / bin_width
    )
    hat = hat / jnp.maximum(hat.sum(axis=1, keepdims=True), 1e-30)

    overlap_weights = overlap / jnp.where(seg_len > 0, seg_len, 1.0)[:, None]
    weights = jnp.where((seg_len > 0)[:, None], overlap_weights, hat)

    power_per_bin = dP @ weights  # (nrho,)

    dV = dvolume_drho * (bin_hi - bin_lo)
    return power_per_bin / jnp.maximum(dV, 1e-30)


def _run_trace(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
    beam: Beam,
) -> tuple[TraceBuffers, jax.Array]:
    """Build interpolators and run the JIT-compiled ODE solve."""
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
    return trace_jitted(
        jnp.asarray(beam.position),
        jnp.asarray(beam.direction),
        setting,
        interpolators,
        magnetic_configuration.nfp,
        magnetic_configuration.rho_1d,
        magnetic_configuration.dvolume_drho,
    )


def trace(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
    beam: Beam,
    trim: bool = True,
) -> TraceResult:
    """Trace a single beam through the plasma.

    Args:
        magnetic_configuration: Magnetic configuration with gridded data
        radial_profiles: Radial profiles of plasma parameters
        beam: Beam initial conditions (position, direction, frequency, mode)
        trim: If True (default), trim the output to the valid trajectory length.
            Set trim=False for gradient-based optimization. The returned
            BeamProfile then contains padded arrays (4097 slots) that are fully
            differentiable w.r.t. beam.position and beam.direction. Padded
            entries have linear_power_density=0 and optical_depth=inf
            (diffrax fills unused buffer slots with inf). Loss functions like::

                jnp.max(jnp.where(jnp.isfinite(od), od, 0.0))
                jnp.sum(result.beam_profile.linear_power_density * weights)

            give the correct answer without trimming.

    Returns:
        TraceResult with beam profile and radial deposition profile.
    """
    result, num_accepted_steps = _run_trace(
        magnetic_configuration, radial_profiles, beam
    )

    if not trim:
        beam_profile = BeamProfile(
            position=result.ode_state[:, :3],
            arc_length=result.arc_length,
            refractive_index=result.ode_state[:, 3:6],
            optical_depth=result.ode_state[:, 6],
            absorption_coefficient=result.absorption_coefficient,
            electron_density=result.electron_density,
            electron_temperature=result.electron_temperature,
            magnetic_field=result.magnetic_field,
            normalized_effective_radius=result.normalized_effective_radius,
            linear_power_density=result.linear_power_density,
        )
        # Padded optical_depth entries are inf (diffrax fills unused slots with inf).
        # _bin_power_deposition sanitizes inf before computing dP.
        power_binned = _bin_power_deposition(
            magnetic_configuration.rho_1d,
            magnetic_configuration.dvolume_drho,
            result.arc_length,
            result.normalized_effective_radius,
            result.ode_state[:, 6],
        )
        return TraceResult(
            beam_profile=beam_profile,
            radial_profile=RadialProfile(
                rho=magnetic_configuration.rho_1d,
                volumetric_power_density=power_binned,
            ),
        )

    # Slot 0 is the antenna position (SaveAt t0=True); accepted steps follow.
    n = num_accepted_steps.item() + 1

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

    power_binned = _bin_power_deposition(
        magnetic_configuration.rho_1d,
        magnetic_configuration.dvolume_drho,
        result.arc_length[:n],
        result.normalized_effective_radius[:n],
        result.ode_state[:n, 6],
    )

    return TraceResult(
        beam_profile=beam_profile,
        radial_profile=RadialProfile(
            rho=magnetic_configuration.rho_1d,
            volumetric_power_density=power_binned,
        ),
    )
