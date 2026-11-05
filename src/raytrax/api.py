"""Main functions to interact with Raytrax."""

import warnings

import interpax
import jax
import jax.numpy as jnp
import jaxtyping as jt

from .equilibrium.interpolate import (
    MagneticConfiguration,
    build_electron_density_profile_interpolator,
    build_electron_temperature_profile_interpolator,
    build_magnetic_field_interpolator,
    build_rho_interpolator,
)
from .tracer.buffers import Interpolators, TraceBuffers
from .tracer.ray import RaySetting
from .tracer.solver import _ARC_LENGTH_TOLERANCE, trace_jitted
from .types import (
    Beam,
    BeamProfile,
    RadialProfile,
    RadialProfiles,
    TraceResult,
    TracerSettings,
)

_N_INTERP = 2000  # dense arc-length samples used for smooth radial binning


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n (minimum 4).

    Used to bucket the valid trajectory length to a small set of fixed sizes,
    limiting XLA recompilation count to O(log(max_steps)) ≈ 12 while keeping
    the number of spline knots close to the actual trajectory length.
    diffrax fills arc_length[n:] with inf, so arc_length[:bucket] has exactly
    the right inf-padding structure that _bin_power_deposition already handles.
    """
    return max(4, 1 << max(0, (n - 1).bit_length()))


_MIN_BIN_VOLUME = 1e-30
"""Lower bound for the volume of a radial bin (the bin at the axis has dV/drho = 0)."""


def _bin_volumes(
    rho_grid: jt.Float[jax.Array, " nrho"],
    dvolume_drho: jt.Float[jax.Array, " nrho"],
) -> jt.Float[jax.Array, " nrho"]:
    """Volume of the radial bins centred on ``rho_grid``, floored at ``_MIN_BIN_VOLUME``."""
    # Integrate dV/drho, linear between grid points, over each half bin. Sampling it
    # at the bin centre instead gives the axis bin dV/drho(0) = 0, i.e. no volume.
    eighth = jnp.diff(rho_grid) / 8
    upper_half = eighth * (3 * dvolume_drho[:-1] + dvolume_drho[1:])
    lower_half = eighth * (3 * dvolume_drho[1:] + dvolume_drho[:-1])
    volumes = (
        jnp.zeros_like(dvolume_drho).at[:-1].add(upper_half).at[1:].add(lower_half)
    )
    return jnp.maximum(volumes, _MIN_BIN_VOLUME)


@jax.jit
def _bin_power_deposition(
    rho_grid: jt.Float[jax.Array, " nrho"],
    dvolume_drho: jt.Float[jax.Array, " nrho"],
    arc_length: jt.Float[jax.Array, " n"],
    rho_trajectory: jt.Float[jax.Array, " n"],
    optical_depth: jt.Float[jax.Array, " n"],
) -> jt.Float[jax.Array, " nrho"]:
    """Compute differentiable volumetric power deposition profile from ray trajectory.

    The ODE solver produces up to max_steps+1 arc-length points; unvisited slots
    (after an early-exit event) are left as inf by diffrax.  Instead of binning
    directly we:

    1. Sanitize inf padding (diffrax fills unused sol.ys slots with inf).
    2. Interpolate rho(s) (cubic) and τ(s) (linear) to _N_INTERP uniformly-spaced
       arc-length samples — no bandwidth choice needed, smoothness follows
       from density. τ uses linear to preserve monotonicity; rho uses cubic
       for smooth deposition profiles.
    3. Use exact overlap-based binning on the dense samples.

    ``rho_grid`` must be uniformly spaced.
    """
    # --- 1. Sanitize inf padding -------------------------------------------
    # diffrax fills unused sol.ys slots with inf.  Push padded arc_length
    # entries to s_max + 1 (strictly beyond the query range of s_fine) so that
    # jnp.interp never picks up the padded rho/tau values.  tau_final is used
    # to fill padded optical_depth so dP = 0 at the boundary.
    tau_final = jnp.max(jnp.where(jnp.isfinite(optical_depth), optical_depth, 0.0))
    optical_depth = jnp.where(jnp.isfinite(optical_depth), optical_depth, tau_final)

    s_max = jnp.max(jnp.where(jnp.isfinite(arc_length), arc_length, 0.0))

    # Degenerate case: only the initial point is present (n_valid=1, s_max=0).
    # Padding formula s_max*(2+arange) would produce duplicate zero knots,
    # corrupting the spline.  Use s_max_safe=1 as a stand-in so all arithmetic
    # stays finite; the result is masked to zero at the end via jnp.where so
    # no spurious deposition is ever returned.  Using jnp.where (not a Python
    # `if`) keeps the function JIT-traceable.
    is_degenerate = s_max == 0
    s_max_safe = jnp.where(is_degenerate, 1.0, s_max)

    # Push padded entries well beyond s_max with *distinct* monotone values.
    # All-equal padding (s_max + 1) creates duplicate cubic knots that corrupt
    # the spline within [0, s_max].  Spacing by s_max per slot pushes each
    # padded knot to 2s_max, 3s_max, … so their influence on [0, s_max] is O(1/k²).
    arc_length = jnp.where(
        jnp.isfinite(arc_length),
        arc_length,
        s_max_safe * (2.0 + jnp.arange(arc_length.shape[0])),
    )

    # Pad rho with the last valid value, not 0: using 0 would pull the cubic
    # spline's boundary derivative at s_max toward zero, distorting the profile.
    n_valid_rho = jnp.sum(jnp.isfinite(rho_trajectory), dtype=jnp.int32)
    rho_last = rho_trajectory[jnp.maximum(n_valid_rho - 1, 0)]
    rho_trajectory = jnp.where(jnp.isfinite(rho_trajectory), rho_trajectory, rho_last)

    # --- 2. Dense interpolation along arc length ---------------------
    # tau uses linear (not cubic) to preserve monotonicity: cubic undershoots
    # below zero, making exp(-tau) > 1 and inflating the total binned power.
    s_fine = jnp.linspace(arc_length[0], s_max_safe, _N_INTERP)
    rho_fine = interpax.interp1d(s_fine, arc_length, rho_trajectory, method="cubic")
    tau_fine = interpax.interp1d(s_fine, arc_length, optical_depth, method="linear")

    # --- 3. Exact overlap-based binning on dense samples -------------------
    # dP_i = exp(-τ_i) − exp(-τ_{i+1}), clamped to ≥ 0.
    dP = jnp.maximum(jnp.exp(-tau_fine[:-1]) - jnp.exp(-tau_fine[1:]), 0.0)
    n_bins = rho_grid.shape[0]
    bin_width = rho_grid[1] - rho_grid[0]
    rho_a = rho_fine[:-1]
    rho_b = rho_fine[1:]
    seg_len = jnp.abs(rho_b - rho_a)
    has_len = seg_len > 0

    # Each segment spreads dP uniformly over its rho range. The bins are uniform and
    # centred on rho_grid, so in u = (rho - rho_0) / Δρ + 1/2 bin k is [k, k + 1):
    # a segment fills its first and last bin partially and the bins between fully,
    # which costs O(segments + bins) instead of a dense (segment × bin) matrix.
    # Every term is at most dP, so no large partial sums cancel.
    def to_u(rho: jax.Array) -> jax.Array:
        return jnp.clip((rho - rho_grid[0]) / bin_width + 0.5, 0.5, n_bins - 0.5)

    u_lo = to_u(jnp.minimum(rho_a, rho_b))
    u_hi = to_u(jnp.maximum(rho_a, rho_b))
    first = jnp.minimum(jnp.floor(u_lo).astype(int), n_bins - 1)
    last = jnp.minimum(jnp.floor(u_hi).astype(int), n_bins - 1)
    per_bin = jnp.where(has_len, dP * bin_width / jnp.where(has_len, seg_len, 1.0), 0.0)
    same = first == last
    power_per_bin = (
        jnp.zeros(n_bins)
        .at[first]
        .add(per_bin * jnp.where(same, u_hi - u_lo, first + 1 - u_lo))
        .at[last]
        .add(jnp.where(same, 0.0, per_bin * (u_hi - last)))
    )
    full = jnp.where(last > first + 1, per_bin, 0.0)
    power_per_bin += jnp.cumsum(
        jnp.zeros(n_bins + 1).at[first + 1].add(full).at[last].add(-full)
    )[:-1]

    # Zero-length (tangential) segments go to the two nearest grid points with a
    # hat kernel, renormalized over the points that exist, so weights stay finite.
    v = (0.5 * (rho_a + rho_b) - rho_grid[0]) / bin_width
    left = jnp.floor(v).astype(int)
    hat_index = jnp.stack([left, left + 1])
    hat_weight = jnp.stack([1 - (v - left), v - left])
    hat_weight = jnp.where((hat_index >= 0) & (hat_index < n_bins), hat_weight, 0.0)
    hat_power = jnp.where(has_len, 0.0, dP) / jnp.maximum(hat_weight.sum(axis=0), 1e-30)
    power_per_bin = power_per_bin.at[hat_index].add(hat_weight * hat_power, mode="drop")

    result = power_per_bin / _bin_volumes(rho_grid, dvolume_drho)
    # Mask to zero for the degenerate single-point case (s_max=0).
    return jnp.where(is_degenerate, jnp.zeros_like(result), result)


_MIN_DEPOSITED_FRACTION = 1e-12
"""Deposited power fraction below which the deposition statistics are undefined."""


@jax.jit
def _deposition_stats(
    power_binned: jt.Float[jax.Array, " nrho"],
    rho_1d: jt.Float[jax.Array, " nrho"],
    dvolume_drho: jt.Float[jax.Array, " nrho"],
) -> tuple[jt.Float[jax.Array, ""], jt.Float[jax.Array, ""]]:
    """Compute power-weighted mean and standard deviation of power deposition in ρ.

    Returns (rho_mean, rho_std), both differentiable, or NaN if no power is
    deposited.
    """
    # fraction of total power in each bin
    power_per_bin = power_binned * _bin_volumes(rho_1d, dvolume_drho)
    total = jnp.sum(power_per_bin)
    has_power = total > _MIN_DEPOSITED_FRACTION
    safe_total = jnp.where(has_power, total, 1.0)
    rho_mean = jnp.sum(rho_1d * power_per_bin) / safe_total
    rho_var = jnp.sum((rho_1d - rho_mean) ** 2 * power_per_bin) / safe_total
    rho_std = jnp.sqrt(jnp.maximum(rho_var, 1e-30))
    return (
        jnp.where(has_power, rho_mean, jnp.nan),
        jnp.where(has_power, rho_std, jnp.nan),
    )


@jax.jit
def _run_trace(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
    beam: Beam,
    settings: TracerSettings,
) -> tuple[TraceBuffers, jax.Array]:
    """Build interpolators and run the JIT-compiled ODE solve."""
    setting = RaySetting(
        frequency=beam.frequency, mode=beam.mode, max_harmonic=beam.max_harmonic
    )
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
        settings,
    )


def trace(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
    beam: Beam,
    trim: bool = True,
    settings: TracerSettings = TracerSettings(),
) -> TraceResult:
    """Trace a single beam through the plasma.

    Args:
        magnetic_configuration: Magnetic configuration with gridded data
        radial_profiles: Radial profiles of plasma parameters
        beam: Beam initial conditions
        trim: If `True` (default), trim the output to the valid trajectory length.
            Set this to `False` when using automatic differentiation.
            Note: this parameter may be removed in a future release.
        settings: ODE solver settings (tolerances, step sizes, step budget,
            termination thresholds). Defaults to :class:`TracerSettings` with
            sensible values.

    Returns:
        TraceResult with beam profile and radial deposition profile.

    Warns:
        RuntimeWarning: If `trim=True` and the solve stopped before the ray left the
            plasma or was absorbed. With `trim=False` only `TraceResult.incomplete`
            is set, so the call stays traceable under `jax.jit` and `jax.grad`.
    """
    result, n_valid = _run_trace(
        magnetic_configuration, radial_profiles, beam, settings
    )

    if not trim:
        # n_valid unused: trim=False returns the full padded buffer.
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
            linear_power_density=result.linear_power_density * beam.power,
        )
        # Padded optical_depth entries are inf (diffrax fills unused slots with inf).
        # _bin_power_deposition sanitizes inf before computing dP.
        od = result.ode_state[:, 6]
        tau_final = jnp.max(jnp.where(jnp.isfinite(od), od, 0.0))
        absorbed_fraction = 1.0 - jnp.exp(-tau_final)
        power_binned = _bin_power_deposition(
            magnetic_configuration.rho_1d,
            magnetic_configuration.dvolume_drho,
            result.arc_length,
            result.normalized_effective_radius,
            od,
        )
        rho_mean, rho_std = _deposition_stats(
            power_binned,
            magnetic_configuration.rho_1d,
            magnetic_configuration.dvolume_drho,
        )
        return TraceResult(
            beam_profile=beam_profile,
            radial_profile=RadialProfile(
                rho=magnetic_configuration.rho_1d,
                volumetric_power_density=power_binned * beam.power,
            ),
            absorbed_power=absorbed_fraction * beam.power,
            absorbed_power_fraction=absorbed_fraction,
            optical_depth=tau_final,
            deposition_rho_mean=rho_mean,
            deposition_rho_std=rho_std,
            incomplete=result.incomplete,
        )

    # Trim to valid (finite) arc-length entries; diffrax leaves unvisited
    # SaveAt slots as inf after an early-exit event.
    n = int(n_valid.item())

    # Slice on the host: eager JAX slicing to a new length n compiles a program per
    # op and length, so every new trajectory length would recompile.
    result = jax.device_get(result)
    if bool(result.incomplete):
        s_end = float(result.arc_length[n - 1])
        if n == settings.max_steps + 1:
            cause = f"ran out of steps (TracerSettings.max_steps={settings.max_steps})"
        elif s_end >= settings.max_arc_length - _ARC_LENGTH_TOLERANCE:
            cause = (
                f"reached TracerSettings.max_arc_length={settings.max_arc_length:g} m"
            )
        else:
            cause = "failed (e.g. a non-finite state)"
        warnings.warn(
            f"The ODE solver {cause} at arc length {s_end:.3g} m, before the ray left "
            "the plasma or was absorbed, so the absorbed power may be underestimated.",
            RuntimeWarning,
            stacklevel=2,
        )

    beam_profile = BeamProfile(
        position=jnp.asarray(result.ode_state[:n, :3]),
        arc_length=jnp.asarray(result.arc_length[:n]),
        refractive_index=jnp.asarray(result.ode_state[:n, 3:6]),
        optical_depth=jnp.asarray(result.ode_state[:n, 6]),
        absorption_coefficient=jnp.asarray(result.absorption_coefficient[:n]),
        electron_density=jnp.asarray(result.electron_density[:n]),
        electron_temperature=jnp.asarray(result.electron_temperature[:n]),
        magnetic_field=jnp.asarray(result.magnetic_field[:n]),
        normalized_effective_radius=jnp.asarray(result.normalized_effective_radius[:n]),
        linear_power_density=jnp.asarray(
            result.linear_power_density[:n] * float(beam.power)
        ),
    )

    # Clamp: _next_power_of_two(n) can exceed the buffer length (e.g. 4097→8192
    # for the default max_steps=4096).
    n_bucket = min(_next_power_of_two(n), result.arc_length.shape[0])
    power_binned = _bin_power_deposition(
        magnetic_configuration.rho_1d,
        magnetic_configuration.dvolume_drho,
        result.arc_length[:n_bucket],
        result.normalized_effective_radius[:n_bucket],
        result.ode_state[:n_bucket, 6],
    )

    tau_final = jnp.asarray(result.ode_state[n - 1, 6])
    absorbed_fraction = 1.0 - jnp.exp(-tau_final)
    rho_mean, rho_std = _deposition_stats(
        power_binned,
        magnetic_configuration.rho_1d,
        magnetic_configuration.dvolume_drho,
    )
    return TraceResult(
        beam_profile=beam_profile,
        radial_profile=RadialProfile(
            rho=magnetic_configuration.rho_1d,
            volumetric_power_density=power_binned * beam.power,
        ),
        absorbed_power=absorbed_fraction * beam.power,
        absorbed_power_fraction=absorbed_fraction,
        optical_depth=tau_final,
        deposition_rho_mean=rho_mean,
        deposition_rho_std=rho_std,
        incomplete=jnp.asarray(result.incomplete),
    )
