"""Tests for the raytrax.api module."""

import functools
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from raytrax.api import (
    _bin_power_deposition,
    _bin_volumes,
    _deposition_stats,
    _next_power_of_two,
    trace,
)
from raytrax.equilibrium.fourier import dvolume_drho
from raytrax.equilibrium.interpolate import MagneticConfiguration
from raytrax.examples import w7x
from raytrax.physics import dispersion, quantities
from raytrax.types import Beam, RadialProfiles, TracerSettings


def test_from_vmec_wout_w7x(w7x_wout):
    """Test the MagneticConfiguration.from_vmec_wout classmethod with the W7-X equilibrium."""
    # Generate the interpolator
    interpolator = MagneticConfiguration.from_vmec_wout(w7x_wout)

    # Check that the interpolator has the expected attributes
    assert hasattr(interpolator, "rphiz")
    assert hasattr(interpolator, "magnetic_field")
    assert hasattr(interpolator, "rho")

    # Check the shapes match
    # The actual shape is (n_r, n_phi, n_z, 3) for rphiz and magnetic_field
    # and (n_r, n_phi, n_z) for rho
    assert interpolator.rphiz.shape == (95, 50, 105, 3)
    assert interpolator.magnetic_field.shape == (95, 50, 105, 3)
    assert interpolator.rho.shape == (95, 50, 105)

    # Check that non-NaN rho values are non-negative
    # Note: rho > 1 is expected for points outside the LCMS (in the vacuum region)
    valid_rho = interpolator.rho[~jnp.isnan(interpolator.rho)]
    assert valid_rho.size > 0, "All rho values are NaN"
    assert jnp.all(valid_rho >= 0.0)
    # Check that we have some points inside the plasma (rho < 1)
    assert jnp.any(valid_rho < 1.0), "No points found inside plasma (rho < 1)"
    # Check that we have some points outside the plasma (rho > 1)
    assert jnp.any(valid_rho > 1.0), "No points found outside plasma (rho > 1)"

    # Verify that the magnetic field has non-zero values
    assert not jnp.all(interpolator.magnetic_field == 0.0)

    # Test that the rphiz coordinates make sense physically
    # For W7-X, R (major radius) should be around 5.5m
    r_values = interpolator.rphiz[..., 0]  # First component is r
    assert jnp.min(r_values) > 0.0  # Should be positive
    assert jnp.max(r_values) < 7.0  # Should be less than ~7m for W7-X

    # Z values should be within reasonable range for W7-X
    z_values = interpolator.rphiz[..., 2]  # Third component is z
    assert jnp.min(z_values) > -1.5  # Lower bound
    assert jnp.max(z_values) < 1.5  # Upper bound


def test_trace_w7x_beam(w7x_wout):
    """Test the trace function with W7-X equilibrium and a specific beam position/direction."""
    # Beam position
    r = 6.50866
    phi = np.deg2rad(-6.56378)
    z = 0.38
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    position = jnp.array([x, y, z])

    # Beam direction starting from W7-X aiming angles
    alpha = np.deg2rad(15.7)
    beta = np.deg2rad(19.7001)
    d_r = -np.cos(alpha) * np.cos(beta)
    d_phi = np.cos(alpha) * np.sin(beta)
    d_z = np.sin(alpha)
    dir_x = d_r * np.cos(phi) - d_phi * np.sin(phi)
    dir_y = d_r * np.sin(phi) + d_phi * np.cos(phi)
    dir_z = d_z
    direction = np.array([dir_x, dir_y, dir_z])
    direction = direction / np.linalg.norm(direction)

    rho = np.linspace(0, 1, 45)
    quadratic_profile = (1 - rho) ** 2
    electron_temperature = 5 * quadratic_profile  # 5 keV on axis
    electron_density = 0.75 * quadratic_profile  # 0.75e20/m³ on axis
    radial_profiles = RadialProfiles(
        rho=rho,
        electron_density=electron_density,
        electron_temperature=electron_temperature,
    )

    beam = Beam(
        position=position,
        direction=direction,
        frequency=140e9,
        mode="O",
        power=1e6,
    )

    interpolator = MagneticConfiguration.from_vmec_wout(w7x_wout)
    result = trace(interpolator, radial_profiles, beam)

    assert hasattr(result, "beam_profile")
    assert hasattr(result, "radial_profile")
    assert result.beam_profile is not None
    assert result.radial_profile is not None


@pytest.mark.parametrize("central_temperature_keV", [1.5, 5.0])
def test_default_tolerances_resolve_narrow_deposition(
    w7x_magnetic_configuration, central_temperature_keV
):
    """The default solver settings place a narrow W7-X X2 deposition within 1e-3 in rho.

    A perpendicular launch absorbs within about 1 cm of arc length, which is the hardest
    case for the step-size controller. The reference is the same trace at rtol = 1e-9,
    which is converged to below 1e-6 in rho.
    """
    rho = np.linspace(0, 1, 101)
    profiles = RadialProfiles(
        rho=rho,
        electron_density=0.6 * (0.1 + 0.9 * (1 - rho**2) ** 0.5),
        electron_temperature=central_temperature_keV
        * (0.05 + 0.95 * (1 - rho**2) ** 1.5),
    ).with_zero_density_at_boundary(0.05)
    beam = Beam(
        position=np.asarray(w7x.PortA.D1.cartesian),
        direction=np.asarray(
            w7x.w7x_aiming_angles_to_direction(5.0, 0.0, w7x.PortA.D1.phi_deg)
        ),
        frequency=140e9,
        mode="X",
        power=1e6,
    )

    result = trace(w7x_magnetic_configuration, profiles, beam)
    reference = trace(
        w7x_magnetic_configuration,
        profiles,
        beam,
        settings=TracerSettings(
            relative_tolerance=1e-9, absolute_tolerance=1e-11, max_steps=8192
        ),
    )

    np.testing.assert_allclose(
        result.deposition_rho_mean, reference.deposition_rho_mean, rtol=0, atol=1e-3
    )
    np.testing.assert_allclose(
        result.deposition_rho_std, reference.deposition_rho_std, rtol=0, atol=1e-3
    )


def test_ray_stays_on_dispersion_surface_with_finite_edge_density(
    tokamak_magnetic_configuration,
):
    """With ne(rho=1) > 0 the ray stays on the cold dispersion surface in the plasma."""
    rho = jnp.linspace(0, 1, 40)
    profiles = RadialProfiles(
        rho=rho,
        electron_density=0.4 * (1 - rho**2) + 0.2,
        electron_temperature=3.0 * (1 - rho**2) + 0.1,
    )
    direction = jnp.array([-1.0, 0.3, 0.2])
    beam = Beam(
        position=jnp.array([3.0 + 1.3, 0.0, 0.0]),  # in vacuum, rho = 1.3
        direction=direction / jnp.linalg.norm(direction),
        frequency=140e9,
        mode="O",
        power=1e6,
    )
    result = trace(tokamak_magnetic_configuration, profiles, beam)
    bp = result.beam_profile

    n_vec = np.asarray(bp.refractive_index)
    b_vec = np.asarray(bp.magnetic_field)
    ne = np.asarray(bp.electron_density)
    b_hat = b_vec / np.linalg.norm(b_vec, axis=1, keepdims=True)
    n_para = np.sum(n_vec * b_hat, axis=1)
    n_perp = np.linalg.norm(n_vec - n_para[:, None] * b_hat, axis=1)
    n2_dispersion = dispersion.dispersion_cold(
        refractive_index_perp=n_perp,
        refractive_index_para=n_para,
        frequency=beam.frequency,
        cyclotron_frequency=quantities.electron_cyclotron_frequency(
            np.linalg.norm(b_vec, axis=1)
        ),
        plasma_frequency=quantities.electron_plasma_frequency(ne),
        mode="O",
    )
    in_plasma = ne > 0.1
    assert in_plasma.sum() > 5
    h = np.sum(n_vec**2, axis=1) - np.asarray(n2_dispersion)
    # The step-size control rejects steps that jump over the density ramp.
    np.testing.assert_array_less(np.abs(h[in_plasma]), 1e-3)


def test_finite_edge_density_matches_converged_trace(tokamak_magnetic_configuration):
    """At default settings, a ray entering a plasma with ne(rho=1) > 0 matches a
    converged trace, and R N_phi, conserved in an axisymmetric field, stays constant.
    """
    rho = jnp.linspace(0, 1, 40)
    profiles = RadialProfiles(
        rho=rho,
        electron_density=0.4 * (1 - rho**2) + 0.2,
        electron_temperature=3.0 * (1 - rho**2) + 0.1,
    )
    direction = jnp.array([-1.0, 0.3, 0.2])
    beam = Beam(
        position=jnp.array([3.0 + 1.3, 0.0, 0.0]),
        direction=direction / jnp.linalg.norm(direction),
        frequency=140e9,
        mode="O",
        power=1e6,
    )
    result = trace(tokamak_magnetic_configuration, profiles, beam)
    converged = trace(
        tokamak_magnetic_configuration,
        profiles,
        beam,
        settings=TracerSettings(
            relative_tolerance=1e-7, absolute_tolerance=1e-9, max_step_size=0.005
        ),
    )
    assert not result.incomplete and not converged.incomplete
    np.testing.assert_allclose(result.optical_depth, converged.optical_depth, rtol=5e-3)

    position = np.asarray(result.beam_profile.position)
    n_vec = np.asarray(result.beam_profile.refractive_index)
    r_n_phi = position[:, 0] * n_vec[:, 1] - position[:, 1] * n_vec[:, 0]
    np.testing.assert_allclose(r_n_phi, r_n_phi[0], rtol=0, atol=1e-4)


@pytest.mark.parametrize(
    "mode, angle_deg", [("O", 0.0), ("O", 25.0), ("X", 0.0), ("X", 25.0)]
)
def test_ray_is_reflected_at_its_cutoff(
    tokamak_magnetic_configuration, mode, angle_deg
):
    """An O-mode ray reaching the O cutoff X = 1 and an X-mode ray reaching the R
    cutoff X = 1 - Y turn around and leave the plasma on the side they entered, at
    default settings as well as converged ones, perpendicular and oblique to B.

    140 GHz in the analytic tokamak: the cutoffs lie well outside the second-harmonic
    resonance at R = 3 m, so nothing is absorbed.
    """
    R0 = 3.0
    rho = jnp.linspace(0, 1, 40)
    profiles = RadialProfiles(
        rho=rho,
        electron_density=0.3 + 3.2 * (1 - rho**2),
        electron_temperature=1.0 * jnp.ones_like(rho),
    )
    angle = np.deg2rad(angle_deg)
    beam = Beam(
        position=jnp.array([R0 + 1.3, 0.0, 0.0]),  # in vacuum, rho = 1.3
        # B is toroidal: the tilt out of the R direction towards phi sets N_par.
        direction=jnp.array([-np.cos(angle), np.sin(angle), 0.0]),
        frequency=140e9,
        mode=mode,
        power=1e6,
    )
    converged = trace(
        tokamak_magnetic_configuration,
        profiles,
        beam,
        settings=TracerSettings(
            relative_tolerance=1e-7,
            absolute_tolerance=1e-9,
            max_step_size=0.002,
            max_steps=65536,
        ),
    )
    result = trace(tokamak_magnetic_configuration, profiles, beam)
    for traced in (result, converged):
        assert not traced.incomplete
        bp = traced.beam_profile
        position = np.asarray(bp.position)
        # Reflected: the ray ends on the low-field side where it started.
        assert np.hypot(position[-1, 0], position[-1, 1]) > R0 + 1.0
        X = (
            np.asarray(quantities.electron_plasma_frequency(bp.electron_density))
            / beam.frequency
        ) ** 2
        Y = (
            np.asarray(
                quantities.electron_cyclotron_frequency(
                    np.linalg.norm(bp.magnetic_field, axis=1)
                )
            )
            / beam.frequency
        )
        n_vec = np.asarray(bp.refractive_index)
        b_hat = np.asarray(bp.magnetic_field)
        b_hat = b_hat / np.linalg.norm(b_hat, axis=1, keepdims=True)
        n_para2 = np.sum(n_vec * b_hat, axis=1) ** 2
        # At oblique incidence the X-mode turns around where (1 - Y) N_par^2 = R, i.e.
        # below the perpendicular R cutoff X = 1 - Y.
        cutoff = 1.0 if mode == "O" else (1.0 - Y) * (1.0 - n_para2)
        # The ray gets close to its cutoff but does not cross it.
        assert np.max(X - cutoff) > -0.05
        np.testing.assert_array_less(X - cutoff, 2e-3)
        r_n_phi = position[:, 0] * n_vec[:, 1] - position[:, 1] * n_vec[:, 0]
        np.testing.assert_allclose(r_n_phi, r_n_phi[0], rtol=0, atol=1e-4)
    end_distance = np.linalg.norm(
        np.asarray(result.beam_profile.position[-1])
        - np.asarray(converged.beam_profile.position[-1])
    )
    assert end_distance < 1e-3


def _make_tokamak_beam_and_profiles(tokamak_magnetic_configuration):
    """Shared setup for TracerSettings tests using the analytic tokamak fixture."""
    R0 = 3.0
    rho = jnp.linspace(0, 1, 40)
    profiles = RadialProfiles(
        rho=rho,
        electron_density=0.5 * (1 - rho**2),
        electron_temperature=3.0 * (1 - rho**2),
    )
    beam = Beam(
        position=jnp.array([R0 + 0.8, 0.0, 0.0]),
        direction=jnp.array([-1.0, 0.0, 0.0]),
        frequency=140e9,
        mode="O",
        power=1e6,
    )
    return beam, profiles


def test_tracer_settings_non_default_runs_under_jit(tokamak_magnetic_configuration):
    """Non-default TracerSettings are accepted by trace() and produce finite results.

    trace() calls trace_jitted internally, which is @jax.jit, so this exercises
    that all TracerSettings fields are valid JAX pytree leaves.
    """
    beam, profiles = _make_tokamak_beam_and_profiles(tokamak_magnetic_configuration)
    settings = TracerSettings(
        relative_tolerance=1e-5,
        absolute_tolerance=1e-7,
        max_step_size=0.02,
        max_arc_length=15.0,
    )

    result = trace(tokamak_magnetic_configuration, profiles, beam, settings=settings)

    assert jnp.isfinite(result.absorbed_power)
    assert jnp.isfinite(result.optical_depth)


def test_step_limit_is_reported_instead_of_silently_truncating(
    tokamak_magnetic_configuration,
):
    """A trace that runs out of ODE steps is flagged instead of reporting ~0 power.

    The X-mode ray at the second harmonic is fully absorbed in about 50 steps.
    """
    _, profiles = _make_tokamak_beam_and_profiles(tokamak_magnetic_configuration)
    beam = Beam(
        position=jnp.array([3.8, 0.0, 0.0]),
        direction=jnp.array([-1.0, 0.0, 0.0]),
        frequency=140e9,
        mode="X",
        power=1e6,
    )
    short = TracerSettings(max_steps=16)

    with pytest.warns(RuntimeWarning, match="max_steps=16"):
        truncated = trace(
            tokamak_magnetic_configuration, profiles, beam, settings=short
        )
    assert bool(truncated.incomplete)
    assert truncated.beam_profile.arc_length.shape == (17,)

    # trim=False must stay traceable: the flag is a traced array, not a Python bool.
    padded = jax.jit(functools.partial(trace, trim=False, settings=short))(
        tokamak_magnetic_configuration, profiles, beam
    )
    assert bool(padded.incomplete)

    complete = trace(
        tokamak_magnetic_configuration,
        profiles,
        beam,
        settings=TracerSettings(max_steps=128),
    )
    assert not bool(complete.incomplete)
    assert float(complete.absorbed_power_fraction) > 0.99
    # The ray stops where exp(-tau) reaches 1e-3, not at the end of the step past it.
    np.testing.assert_allclose(complete.optical_depth, np.log(1e3), rtol=1e-6)


def test_ray_leaving_the_plasma_stops_at_the_exit_condition(w7x_wout):
    """A ray leaving the plasma with a long step stops exactly at rho = 1.05.

    The last step of this ray goes from rho = 1.04 to rho = 1.12, and rho is not
    monotonic along it, so the root find must stay inside the step.
    """
    # B scaled to 3 T on axis.
    configuration = MagneticConfiguration.from_vmec_wout(
        w7x_wout, magnetic_field_scale=1.1792
    )
    phi = np.deg2rad(352.876)
    beam = Beam(
        position=jnp.array([6.6192 * np.cos(phi), 6.6192 * np.sin(phi), 0.4963]),
        direction=jnp.array([-0.89116, 0.44952, -0.06135]),
        frequency=45.4e9,
        mode="X",
        power=1e6,
    )
    rho = np.linspace(0, 1, 45)
    profiles = RadialProfiles(
        rho=rho,
        electron_density=0.1 * (1 - rho**2),
        electron_temperature=8.0 * (1 - rho**2),
    )

    result = trace(configuration, profiles, beam)

    assert not bool(result.incomplete)
    assert np.all(np.diff(result.beam_profile.arc_length) > 0)
    np.testing.assert_allclose(
        result.beam_profile.normalized_effective_radius[-1], 1.05, rtol=1e-6
    )


def test_tracer_settings_max_arc_length_limits_trajectory(
    tokamak_magnetic_configuration,
):
    """A short max_arc_length stops the ray at that length and flags it incomplete."""
    beam, profiles = _make_tokamak_beam_and_profiles(tokamak_magnetic_configuration)

    result_default = trace(tokamak_magnetic_configuration, profiles, beam)
    with pytest.warns(RuntimeWarning, match="max_arc_length"):
        result_short = trace(
            tokamak_magnetic_configuration,
            profiles,
            beam,
            settings=TracerSettings(max_arc_length=0.5),
        )
    assert bool(result_short.incomplete)
    assert not bool(result_default.incomplete)

    default_length = float(result_default.beam_profile.arc_length[-1])
    short_length = float(result_short.beam_profile.arc_length[-1])
    assert short_length < default_length
    # The event root find ends the ray at the cap itself, not at the end of the
    # step that crossed it.
    np.testing.assert_allclose(short_length, 0.5, rtol=0, atol=1e-6)


def test_deposition_stats():
    """Mean and std are weighted by the deposited power and normalised by its sum."""
    rho = jnp.linspace(0.0, 1.0, 11)
    dvolume_drho = jnp.ones(11)
    edges = jnp.concatenate([rho[:1], 0.5 * (rho[:-1] + rho[1:]), rho[-1:]])
    dV = jnp.diff(edges)

    # All power in the bin at rho = 0.3: mean is 0.3, independent of the amount.
    for amount in (1e-6, 0.5):
        power_binned = jnp.zeros(11).at[3].set(amount / dV[3])
        mean, std = _deposition_stats(power_binned, rho, dvolume_drho)
        np.testing.assert_allclose(float(mean), 0.3, atol=1e-12)
        assert float(std) < 1e-6

    # Equal power at rho = 0.2 and 0.6.
    power_binned = jnp.zeros(11).at[2].set(0.1 / dV[2]).at[6].set(0.1 / dV[6])
    mean, std = _deposition_stats(power_binned, rho, dvolume_drho)
    np.testing.assert_allclose(float(mean), 0.4, atol=1e-12)
    np.testing.assert_allclose(float(std), 0.2, atol=1e-12)

    # No deposited power.
    mean, std = _deposition_stats(jnp.zeros(11), rho, dvolume_drho)
    assert np.isnan(float(mean)) and np.isnan(float(std))


def test_deposition_stats_with_power_at_axis():
    """Power deposited in the axis bin, where dV/drho = 0, enters the statistics."""
    rho_grid = jnp.linspace(0.0, 1.0, 11)
    dvolume_drho = 2.0 * rho_grid  # dV/drho vanishes on the axis

    # Ray segment passing through the axis: 0.15 -> 0.0 -> 0.15.
    rho_trajectory = jnp.array([0.15, 0.1, 0.05, 0.0, 0.05, 0.1, 0.15])
    optical_depth = jnp.linspace(0.0, 1.0, 7)
    arc_length = jnp.linspace(0.0, 0.3, 7)
    power_binned = _bin_power_deposition(
        rho_grid, dvolume_drho, arc_length, rho_trajectory, optical_depth
    )
    mean, std = _deposition_stats(power_binned, rho_grid, dvolume_drho)
    assert np.isfinite(float(mean)) and np.isfinite(float(std))
    # Uniform deposition in rho on [0, 0.15]; bins centred at 0, 0.1 (0.05 width each
    # on [0, 0.05] and [0.05, 0.15]) give the mean (0.05 * 0 + 0.1 * 0.1) / 0.15.
    np.testing.assert_allclose(float(mean), 0.1 * 0.1 / 0.15, rtol=0.05)

    # All power in the axis bin.
    power_axis = jnp.zeros(11).at[0].set(0.5 / 1e-30)
    mean, _ = _deposition_stats(power_axis, rho_grid, dvolume_drho)
    np.testing.assert_allclose(float(mean), 0.0, atol=1e-12)


def test_bin_volumes_add_up_to_the_vmec_volume(w7x_wout):
    """The radial bins hold the plasma volume, and the axis bin is not empty."""
    rho_grid = jnp.linspace(0.0, 1.0, 200)
    volumes = _bin_volumes(rho_grid, dvolume_drho(w7x_wout, rho_grid))
    np.testing.assert_allclose(volumes.sum(), w7x_wout.volume_p, rtol=1e-4)
    # dV/drho ∝ rho near the axis, so the axis bin [0, Δ/2] holds 1/8 of the next.
    np.testing.assert_allclose(volumes[0] / volumes[1], 1 / 8, rtol=1e-2)


def test_bin_power_deposition():
    """Test the _bin_power_deposition helper function."""
    rho_grid = jnp.linspace(0.0, 1.0, 11)  # 11 points: 0.0, 0.1, ..., 1.0
    dvolume_drho = jnp.linspace(1.0, 2.0, 11)

    # Non-monotonic trajectory: enters at rho=0.9, reaches rho=0.3, exits to rho=0.8.
    # Crosses rho≈0.5 twice.
    rho_trajectory = jnp.array([0.9, 0.7, 0.5, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    optical_depth = jnp.linspace(0.0, 2.0, 9)  # tau: 0 → 2 along trajectory
    arc_length = jnp.linspace(0.0, 1.0, 9)

    result = _bin_power_deposition(
        rho_grid, dvolume_drho, arc_length, rho_trajectory, optical_depth
    )

    assert result.shape == rho_grid.shape
    assert jnp.all(jnp.isfinite(result))
    assert result[5] > 0  # rho=0.5 bin should have deposited power

    # Power conservation: sum(dP/dV * dV) = exp(-tau[0]) - exp(-tau[-1])
    edges = jnp.concatenate(
        [rho_grid[:1], 0.5 * (rho_grid[:-1] + rho_grid[1:]), rho_grid[-1:]]
    )
    total_deposited = jnp.sum(result * dvolume_drho * jnp.diff(edges))
    expected_total = float(jnp.exp(-optical_depth[0]) - jnp.exp(-optical_depth[-1]))
    np.testing.assert_allclose(total_deposited, expected_total, rtol=1e-5)


def test_bin_power_deposition_power_conservation():
    """Total deposited power equals exp(-tau_0) - exp(-tau_final) exactly."""
    rho_grid = jnp.linspace(0.0, 1.0, 200)
    dvolume_drho = jnp.ones(200)

    # Monotone trajectory: enters at rho=0.9, exits at rho=0.1
    rho_trajectory = jnp.linspace(0.9, 0.1, 20)
    optical_depth = jnp.linspace(0.0, 3.0, 20)  # tau_final = 3
    arc_length = jnp.linspace(0.0, 1.0, 20)

    result = _bin_power_deposition(
        rho_grid, dvolume_drho, arc_length, rho_trajectory, optical_depth
    )

    assert result.shape == rho_grid.shape

    traversed = (rho_grid >= 0.1) & (rho_grid <= 0.9)
    assert jnp.all(result[traversed] > 0), (
        "Bins along trajectory path should be populated"
    )

    edges = jnp.concatenate(
        [rho_grid[:1], 0.5 * (rho_grid[:-1] + rho_grid[1:]), rho_grid[-1:]]
    )
    total_deposited = jnp.sum(result * dvolume_drho * jnp.diff(edges))
    expected_total = 1.0 - float(jnp.exp(-optical_depth[-1]))  # 1 - exp(-3)
    np.testing.assert_allclose(total_deposited, expected_total, rtol=1e-5)


def test_bin_power_with_padded_zeros():
    """Padded entries (optical_depth=inf after trajectory end) contribute zero power."""
    rho_grid = jnp.linspace(0.0, 1.0, 11)
    dvolume_drho = jnp.ones(11)

    # 5 valid points followed by 5 inf-padded slots (diffrax fills unused slots with inf)
    rho_valid = jnp.array([0.8, 0.6, 0.4, 0.3, 0.5])
    tau_valid = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
    s_valid = jnp.linspace(0.0, 1.0, 5)
    rho_padded = jnp.zeros(5)
    tau_padded = jnp.full(5, jnp.inf)  # diffrax fills unused sol.ys slots with inf
    s_padded = jnp.full(5, jnp.inf)

    rho_trajectory = jnp.concatenate([rho_valid, rho_padded])
    optical_depth = jnp.concatenate([tau_valid, tau_padded])
    arc_length = jnp.concatenate([s_valid, s_padded])

    result_padded = _bin_power_deposition(
        rho_grid, dvolume_drho, arc_length, rho_trajectory, optical_depth
    )
    result_trimmed = _bin_power_deposition(
        rho_grid, dvolume_drho, s_valid, rho_valid, tau_valid
    )

    assert result_padded.shape == rho_grid.shape
    assert jnp.all(jnp.isfinite(result_padded))
    # Padded inf entries must not significantly change the result.
    # A global cubic spline has different boundary conditions when padded
    # points are present, so perfect agreement is not achievable; 5% is fine.
    np.testing.assert_allclose(result_padded, result_trimmed, rtol=0.05)


def test_bin_power_full_padded_arrays_match_trimmed():
    """inf-padded tail in all three arrays does not change the deposited power.

    This regression test covers the sanitization path in _bin_power_deposition.
    trace(trim=True) passes power-of-2 bucketed slices (not the full max_steps+1
    buffer), so the padded region is small (at most bucket-n entries of inf);
    this test exercises the same sanitization with a larger pad to confirm
    robustness. diffrax fills arc_length, ode_state (→ optical_depth), and
    normalized_effective_radius with inf for every unused slot.
    """
    rho_grid = jnp.linspace(0.0, 1.0, 20)
    dvolume_drho = jnp.ones(20)

    n_valid = 8
    n_pad = 42  # simulate a large fixed-size buffer with many unused slots

    rho_valid = jnp.linspace(0.9, 0.2, n_valid)
    tau_valid = jnp.linspace(0.0, 2.5, n_valid)
    s_valid = jnp.linspace(0.0, 0.7, n_valid)

    # All three arrays are inf in the padded region — exactly as diffrax produces.
    rho_trajectory = jnp.concatenate([rho_valid, jnp.full(n_pad, jnp.inf)])
    optical_depth = jnp.concatenate([tau_valid, jnp.full(n_pad, jnp.inf)])
    arc_length = jnp.concatenate([s_valid, jnp.full(n_pad, jnp.inf)])

    result_full = _bin_power_deposition(
        rho_grid, dvolume_drho, arc_length, rho_trajectory, optical_depth
    )
    result_trimmed = _bin_power_deposition(
        rho_grid, dvolume_drho, s_valid, rho_valid, tau_valid
    )

    assert result_full.shape == rho_grid.shape
    assert jnp.all(jnp.isfinite(result_full)), (
        "inf in rho_trajectory (padded slots) must be sanitized before binning"
    )

    # Power conservation: total deposited power is the same regardless of padding.
    edges = jnp.concatenate(
        [rho_grid[:1], 0.5 * (rho_grid[:-1] + rho_grid[1:]), rho_grid[-1:]]
    )
    dV = dvolume_drho * jnp.diff(edges)
    total_full = float(jnp.sum(result_full * dV))
    total_trimmed = float(jnp.sum(result_trimmed * dV))
    np.testing.assert_allclose(total_full, total_trimmed, rtol=0.02)


def test_next_power_of_two():
    """_next_power_of_two returns a small set of fixed sizes to cap XLA recompilations.

    The function itself may return a value larger than the actual buffer length
    (e.g. 4097 → 8192); the caller is responsible for clamping to the buffer
    size so that full-length traces don't produce a unique shape.
    """
    assert _next_power_of_two(1) == 4
    assert _next_power_of_two(4) == 4
    assert _next_power_of_two(5) == 8
    assert _next_power_of_two(8) == 8
    assert _next_power_of_two(9) == 16
    assert _next_power_of_two(4096) == 4096
    # 4097 exceeds the last power-of-two bucket; the caller clamps to buffer length.
    assert _next_power_of_two(4097) == 8192
    # Verify the caller-side clamp: min(_next_power_of_two(4097), 4097) == 4097
    assert min(_next_power_of_two(4097), 4097) == 4097


def test_bin_power_bucketed_slice_matches_trimmed():
    """Power-of-2 bucketed slice gives same deposition as exact trimmed slice.

    trace(trim=True) now passes result.arc_length[:_next_power_of_two(n)] instead of
    either result.arc_length[:n] (variable shape → recompilation) or result.arc_length
    (full 4097 knots → spline perturbation). The bucketed slice has at most
    bucket - n extra inf knots, which the sanitization step pushes well beyond s_max.
    """
    rho_grid = jnp.linspace(0.0, 1.0, 20)
    dvolume_drho = jnp.ones(20)

    n_valid = 13  # not a power of two → bucket = 16
    n_bucket = _next_power_of_two(n_valid)  # 16
    n_total = 64  # simulate a larger fixed buffer

    assert n_bucket == 16

    rho_valid = jnp.linspace(0.85, 0.15, n_valid)
    tau_valid = jnp.linspace(0.0, 3.0, n_valid)
    s_valid = jnp.linspace(0.0, 1.2, n_valid)

    # Build a full diffrax-style buffer: valid entries then inf padding.
    rho_buf = jnp.concatenate([rho_valid, jnp.full(n_total - n_valid, jnp.inf)])
    tau_buf = jnp.concatenate([tau_valid, jnp.full(n_total - n_valid, jnp.inf)])
    s_buf = jnp.concatenate([s_valid, jnp.full(n_total - n_valid, jnp.inf)])

    result_bucketed = _bin_power_deposition(
        rho_grid,
        dvolume_drho,
        s_buf[:n_bucket],
        rho_buf[:n_bucket],
        tau_buf[:n_bucket],
    )
    result_trimmed = _bin_power_deposition(
        rho_grid,
        dvolume_drho,
        s_valid,
        rho_valid,
        tau_valid,
    )

    assert result_bucketed.shape == rho_grid.shape
    assert jnp.all(jnp.isfinite(result_bucketed))

    edges = jnp.concatenate(
        [rho_grid[:1], 0.5 * (rho_grid[:-1] + rho_grid[1:]), rho_grid[-1:]]
    )
    dV = dvolume_drho * jnp.diff(edges)
    total_bucketed = float(jnp.sum(result_bucketed * dV))
    total_trimmed = float(jnp.sum(result_trimmed * dV))
    # Bucketed slice has only 3 extra inf knots → negligible spline perturbation.
    np.testing.assert_allclose(total_bucketed, total_trimmed, rtol=0.01)


def test_bin_power_deposition_single_point_returns_zeros():
    """n_valid=1 (s_max=0) returns an all-zero deposition profile without crashing.

    When the ODE solver terminates immediately (e.g., beam starts outside the
    plasma), arc_length = [0.0] and s_max = 0.  The padding formula
    s_max*(2+arange) would collapse all padded knots to 0, producing duplicate
    spline knots and corrupting the interpolation.  _bin_power_deposition must
    detect s_max==0 and return zeros instead of attempting the spline.
    """
    rho_grid = jnp.linspace(0.0, 1.0, 20)
    dvolume_drho = jnp.ones(20)

    # Single valid point (what diffrax produces for an immediately-terminated trace)
    # followed by the inf padding that _next_power_of_two(1)==4 would expose.
    arc_length = jnp.array([0.0, jnp.inf, jnp.inf, jnp.inf])
    rho_trajectory = jnp.array([0.5, jnp.inf, jnp.inf, jnp.inf])
    optical_depth = jnp.array([0.0, jnp.inf, jnp.inf, jnp.inf])

    result = _bin_power_deposition(
        rho_grid, dvolume_drho, arc_length, rho_trajectory, optical_depth
    )

    assert result.shape == rho_grid.shape
    assert jnp.all(result == 0.0), "Single-point trace must return zero deposition"


def test_bin_power_large_bucket_profile_matches_trimmed():
    """n=2049 → bucket=4096 (2047 padding slots) preserves total power and profile shape.

    Catches rho-padding bugs: using rho=0 for padding knots distorts the cubic
    spline's boundary derivative at s_max, creating artifacts near the trajectory end.
    """
    rho_grid = jnp.linspace(0.0, 1.0, 20)
    dvolume_drho = jnp.ones(20)

    n_valid = 2049  # just above a bucket boundary → bucket = 4096
    n_bucket = _next_power_of_two(n_valid)
    assert n_bucket == 4096

    # Last rho is 0.45 (not 0), so zero-padding would distort the boundary derivative.
    rho_valid = jnp.linspace(0.9, 0.45, n_valid)
    tau_valid = jnp.linspace(0.0, 3.5, n_valid)
    s_valid = jnp.linspace(0.0, 2.5, n_valid)

    n_pad = n_bucket - n_valid  # 2047
    rho_buf = jnp.concatenate([rho_valid, jnp.full(n_pad, jnp.inf)])
    tau_buf = jnp.concatenate([tau_valid, jnp.full(n_pad, jnp.inf)])
    s_buf = jnp.concatenate([s_valid, jnp.full(n_pad, jnp.inf)])

    result_bucketed = _bin_power_deposition(
        rho_grid, dvolume_drho, s_buf, rho_buf, tau_buf
    )
    result_trimmed = _bin_power_deposition(
        rho_grid, dvolume_drho, s_valid, rho_valid, tau_valid
    )

    assert result_bucketed.shape == rho_grid.shape
    assert jnp.all(jnp.isfinite(result_bucketed))

    edges = jnp.concatenate(
        [rho_grid[:1], 0.5 * (rho_grid[:-1] + rho_grid[1:]), rho_grid[-1:]]
    )
    dV = dvolume_drho * jnp.diff(edges)
    total_bucketed = float(jnp.sum(result_bucketed * dV))
    total_trimmed = float(jnp.sum(result_trimmed * dV))

    # Total power and per-bin profile must match (latter catches spline boundary artifacts).
    np.testing.assert_allclose(total_bucketed, total_trimmed, rtol=0.01)
    np.testing.assert_allclose(result_bucketed, result_trimmed, rtol=0.05, atol=1e-6)


def test_x1_ray_absorbs_after_crossing_parallel_light_line(w7x_magnetic_configuration):
    """Step-2 validation sample 928 must absorb before approaching a cold resonance.

    Previously absorption vanished at |N_parallel| = 1, leaving the ray to
    approach divergent N and exhaust its step budget. Scale the native W7-X
    field to approximately 2.5 T, as in the validation scan.
    """
    configuration = replace(
        w7x_magnetic_configuration,
        magnetic_field=w7x_magnetic_configuration.magnetic_field * 0.982442,
    )
    rho = jnp.linspace(0.0, 1.0, 501)
    profiles = RadialProfiles(
        rho=rho,
        electron_density=0.2529599084569189
        * (
            0.032558900989367645
            + (1 - 0.032558900989367645)
            * (1 - rho**3.8262163882617397) ** 3.0092203020039183
        ),
        electron_temperature=8.219955542614228
        * (
            0.02921659320420252
            + (1 - 0.02921659320420252)
            * (1 - rho**2.478202073624697) ** 3.1351784534530545
        ),
    )
    beam = Beam(
        position=jnp.array(
            [6.2912875965327215, 0.7658132158020339, 0.5523136511803072]
        ),
        direction=jnp.array(
            [-0.5910383845967915, 0.8017814550004639, -0.08843147827868825]
        ),
        frequency=62.951748777242386e9,
        mode="X",
        power=1e6,
    )
    result = trace(configuration, profiles, beam)
    bp = result.beam_profile
    n_parallel = jnp.sum(
        bp.refractive_index * bp.magnetic_field, axis=1
    ) / jnp.linalg.norm(bp.magnetic_field, axis=1)
    assert jnp.any(jnp.abs(n_parallel) > 1)
    assert not result.incomplete
    assert len(bp.arc_length) < 200
    np.testing.assert_allclose(result.optical_depth, np.log(1000), atol=1e-5, rtol=0)
    np.testing.assert_allclose(bp.arc_length[-1], 1.14, atol=0.002, rtol=0)
