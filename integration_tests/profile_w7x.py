"""Comprehensive profiling script for raytrax W7-X ECRH performance.

Measures:
- Equilibrium loading and interpolator setup
- Compilation time (first run)
- Runtime performance (steady-state)
- Memory usage and trajectory statistics
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from raytrax.api import trace
from raytrax.types import Beam, RadialProfiles, MagneticConfiguration
from raytrax.data import get_w7x_wout
from raytrax.interpolate import (
    build_magnetic_field_interpolator,
    build_rho_interpolator,
)


def travis_profile(rho, t0, width, exponent, pedestal_value, pedestal_position, shift):
    """Travis profile function (from ecrh_w7x)."""
    return (
        (t0 - pedestal_value) * (1 - ((rho + shift) / (1 + shift)) ** exponent) ** width
        + pedestal_value
    ) * jnp.where(rho < pedestal_position, 1, 0) + pedestal_value * jnp.where(
        rho >= pedestal_position, 1, 0
    )


def cylindrical_to_cartesian(r_cyl, phi_deg, z):
    """Convert cylindrical coordinates to Cartesian."""
    phi_rad = np.deg2rad(phi_deg)
    return (
        r_cyl * np.cos(phi_rad),
        r_cyl * np.sin(phi_rad),
        z,
    )


def w7x_aiming_angles_to_direction(alpha_deg, beta_deg, phi_antenna_deg):
    """Convert W7-X aiming angles to direction vector (from ecrh_w7x.py)."""
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    phi = np.deg2rad(phi_antenna_deg)

    d_r = -np.cos(alpha) * np.cos(beta)
    d_phi = np.cos(alpha) * np.sin(beta)
    d_z = np.sin(alpha)
    d_x = d_r * np.cos(phi) - d_phi * np.sin(phi)
    d_y = d_r * np.sin(phi) + d_phi * np.cos(phi)
    norm = np.sqrt(d_x**2 + d_y**2 + d_z**2)
    return (float(d_x / norm), float(d_y / norm), float(d_z / norm))


def find_b0_on_axis(wout):
    """Find |B| on the magnetic axis at phi=0, z=0."""
    eq_unscaled = MagneticConfiguration.from_vmec_wout(wout)
    B_interp = build_magnetic_field_interpolator(eq_unscaled)
    rho_interp = build_rho_interpolator(eq_unscaled)

    # Scan R at phi=0, z=0 to find the axis (minimum rho)
    best_rho, best_B = 999.0, 0.0
    for R in np.arange(5.0, 6.5, 0.002):
        rho_val = float(rho_interp(R, 0.0, 0.0))
        if rho_val < best_rho:
            best_rho = rho_val
            B_grid = B_interp(R, 0.0, 0.0)
            best_B = float(jnp.linalg.norm(B_grid))
    return best_B


def setup_w7x_scenario():
    """Set up W7-X ECRH scenario (140 GHz O-mode)."""
    # Load equilibrium
    wout = get_w7x_wout()

    # Scale to target B0
    B0_TARGET = 2.52076
    b0_unscaled = find_b0_on_axis(wout)
    b_scale = B0_TARGET / b0_unscaled

    eq = MagneticConfiguration.from_vmec_wout(wout, magnetic_field_scale=b_scale)

    # Build profiles
    n_rho = 501
    rho = jnp.linspace(0, 1, n_rho)
    ne_1e20 = travis_profile(rho, 0.75, 0.05, 20, 2.8, 0, 0.32)
    te_keV = travis_profile(rho, 5.0, 0.03, 4, 2.0, -0.2, 0.5)
    profiles = RadialProfiles(
        rho=rho,
        electron_density=ne_1e20,
        electron_temperature=te_keV,
    )

    # Define beam
    antenna_cyl = (6.50866, -6.56378, -0.38)
    target_w7x_angles = (15.7, 19.7001)
    antenna_cart = cylindrical_to_cartesian(*antenna_cyl)
    direction = w7x_aiming_angles_to_direction(
        target_w7x_angles[0], target_w7x_angles[1], antenna_cyl[1]
    )
    beam = Beam(
        position=jnp.array(antenna_cart),
        direction=jnp.array(direction),
        frequency=jnp.array(140e9),
        mode="O",
    )

    return eq, profiles, beam


def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds*1e6:.1f} μs"
    elif seconds < 1:
        return f"{seconds*1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def main():
    print("=" * 80)
    print("RAYTRAX W7-X ECRH PERFORMANCE PROFILE")
    print("=" * 80)
    print()

    # ========================================================================
    print("PHASE 1: Setup and Interpolator Construction")
    print("=" * 80)

    t0_total = time.perf_counter()

    # Load equilibrium
    t0 = time.perf_counter()
    wout = get_w7x_wout()
    t_load = time.perf_counter() - t0
    print(f"  Load VMEC equilibrium:        {format_time(t_load)}")

    # Find B0 scaling
    t0 = time.perf_counter()
    B0_TARGET = 2.52076
    b0_unscaled = find_b0_on_axis(wout)
    b_scale = B0_TARGET / b0_unscaled
    t_scaling = time.perf_counter() - t0
    print(f"  Compute B0 scaling:           {format_time(t_scaling)}")
    print(f"    B0 unscaled: {b0_unscaled:.5f} T")
    print(f"    B0 target:   {B0_TARGET:.5f} T")
    print(f"    Scale factor: {b_scale:.6f}")

    # Build interpolators
    t0 = time.perf_counter()
    eq = MagneticConfiguration.from_vmec_wout(wout, magnetic_field_scale=b_scale)
    t_interp_build = time.perf_counter() - t0
    print(f"  Build MagneticConfiguration:  {format_time(t_interp_build)}")

    # Setup profiles
    t0 = time.perf_counter()
    n_rho = 501
    rho = jnp.linspace(0, 1, n_rho)
    ne_1e20 = travis_profile(rho, 0.75, 0.05, 20, 2.8, 0, 0.32)
    te_keV = travis_profile(rho, 5.0, 0.03, 4, 2.0, -0.2, 0.5)
    profiles = RadialProfiles(
        rho=rho,
        electron_density=ne_1e20,
        electron_temperature=te_keV,
    )
    t_profiles = time.perf_counter() - t0
    print(f"  Build profiles:               {format_time(t_profiles)}")

    # Define beam
    antenna_cyl = (6.50866, -6.56378, -0.38)
    target_w7x_angles = (15.7, 19.7001)
    antenna_cart = cylindrical_to_cartesian(*antenna_cyl)
    direction = w7x_aiming_angles_to_direction(
        target_w7x_angles[0], target_w7x_angles[1], antenna_cyl[1]
    )
    beam = Beam(
        position=jnp.array(antenna_cart),
        direction=jnp.array(direction),
        frequency=jnp.array(140e9),
        mode="O",
    )

    t_setup_total = time.perf_counter() - t0_total
    print(f"  TOTAL SETUP TIME:             {format_time(t_setup_total)}")
    print()

    # ========================================================================
    print("PHASE 2: Compilation (First Run)")
    print("=" * 80)

    t0 = time.perf_counter()
    result = trace(eq, profiles, beam)
    jax.block_until_ready(result.beam_profile.position)
    t_compile = time.perf_counter() - t0
    print(f"  First trace (with compilation): {format_time(t_compile)}")

    # Get basic trajectory info
    s = np.asarray(result.beam_profile.arc_length)
    tau = np.asarray(result.beam_profile.optical_depth)
    alpha = np.asarray(result.beam_profile.absorption_coefficient)
    print(f"    Trajectory points: {len(s)}")
    print(f"    Arc length: {s[0]:.3f} -> {s[-1]:.3f} m")
    print(f"    Final optical depth: τ = {tau[-1]:.4f}")
    print(f"    Absorption: {100*(1 - np.exp(-tau[-1])):.2f}%")
    print(f"    Max α: {np.nanmax(alpha):.2f} m⁻¹")
    print()

    # ========================================================================
    print("PHASE 3: Runtime Performance (Steady-State)")
    print("=" * 80)

    # Warmup
    print("  Warming up...")
    for _ in range(2):
        result = trace(eq, profiles, beam)
        jax.block_until_ready(result.beam_profile.position)

    # Timed runs
    print("  Timing 20 runs...")
    times = []
    for i in range(20):
        t0 = time.perf_counter()
        result = trace(eq, profiles, beam)
        jax.block_until_ready(result.beam_profile.position)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if (i + 1) % 5 == 0:
            print(f"    Run {i+1:2d}: {format_time(elapsed)}")

    times = np.array(times)
    print()
    print("  STATISTICS:")
    print(f"    Median:  {format_time(np.median(times))}")
    print(f"    Mean:    {format_time(np.mean(times))}")
    print(f"    Std:     {format_time(np.std(times))}")
    print(f"    Min:     {format_time(np.min(times))}")
    print(f"    Max:     {format_time(np.max(times))}")
    print(f"    Range:   {format_time(np.max(times) - np.min(times))}")
    print()

    # ========================================================================
    print("PHASE 4: Detailed Trajectory Analysis")
    print("=" * 80)

    pos = np.asarray(result.beam_profile.position)
    ne = np.asarray(result.beam_profile.electron_density)
    te = np.asarray(result.beam_profile.electron_temperature)
    B = np.asarray(result.beam_profile.magnetic_field)
    rho = np.asarray(result.beam_profile.normalized_effective_radius)

    print(f"  Position range:")
    print(f"    x: [{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}] m")
    print(f"    y: [{pos[:, 1].min():.3f}, {pos[:, 1].max():.3f}] m")
    print(f"    z: [{pos[:, 2].min():.3f}, {pos[:, 2].max():.3f}] m")
    print(
        f"    R: [{np.sqrt(pos[:, 0]**2 + pos[:, 1]**2).min():.3f}, {np.sqrt(pos[:, 0]**2 + pos[:, 1]**2).max():.3f}] m"
    )
    print()

    print(f"  Plasma parameters:")
    print(f"    ρ: [{rho.min():.3f}, {rho.max():.3f}]")
    print(f"    ne: [{ne.min():.2f}, {ne.max():.2f}] × 10²⁰ m⁻³")
    print(f"    Te: [{te.min():.3f}, {te.max():.3f}] keV")
    print(
        f"    |B|: [{np.linalg.norm(B, axis=1).min():.3f}, {np.linalg.norm(B, axis=1).max():.3f}] T"
    )
    print()

    print(f"  Absorption profile:")
    print(f"    α range: [{np.nanmin(alpha):.2e}, {np.nanmax(alpha):.2e}] m⁻¹")
    print(f"    Points with α > 1 m⁻¹: {np.sum(alpha > 1)}/{len(alpha)}")
    print(f"    Points with α > 10 m⁻¹: {np.sum(alpha > 10)}/{len(alpha)}")
    print()

    # ========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Setup time:           {format_time(t_setup_total)}")
    print(f"  Compilation time:     {format_time(t_compile)}")
    print(f"  Runtime (median):     {format_time(np.median(times))}")
    print(f"  Speedup over compile: {t_compile / np.median(times):.1f}x")
    print(f"  Trajectory length:    {len(s)} points, {s[-1]-s[0]:.3f} m")
    print(f"  Final absorption:     {100*(1 - np.exp(-tau[-1])):.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
