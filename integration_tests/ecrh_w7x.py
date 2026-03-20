"""Integration test: W7-X ECRH absorption comparing raytrax with TRAVIS."""

import jax

jax.config.update("jax_enable_x64", True)

import shutil
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from travis_wrapper import TravisECRHInput, run_travis, save_reference_data

from raytrax.api import trace
from raytrax.equilibrium.interpolate import MagneticConfiguration
from raytrax.examples import get_w7x_equilibrium as get_w7x_wout
from raytrax.types import Beam, RadialProfiles

# Find TRAVIS executable in PATH, or None if not available
_travis_exe_path = shutil.which("travis-nc")
TRAVIS_EXE = Path(_travis_exe_path) if _travis_exe_path else None
TRAVIS_OUTPUT_DIR = Path("/tmp/travis_ecrh_w7x")
TRAVIS_REF_FILE = Path(__file__).parent / "data" / "travis_w7x_reference.json"
B0_TARGET = 2.52076  # Target B0 normalization at magnetic axis


def cylindrical_to_cartesian(r_m, phi_deg, z_m):
    """Convert cylindrical to Cartesian coordinates."""
    phi_rad = np.deg2rad(phi_deg)
    x = r_m * np.cos(phi_rad)
    y = r_m * np.sin(phi_rad)
    return (float(x), float(y), float(z_m))


def w7x_aiming_angles_to_direction(theta_pol_deg, theta_tor_deg, antenna_phi_deg):
    """Convert W7-X aiming angles to Cartesian direction vector."""
    alpha = np.deg2rad(theta_pol_deg)
    beta = np.deg2rad(theta_tor_deg)
    phi = np.deg2rad(antenna_phi_deg)
    d_r = -np.cos(alpha) * np.cos(beta)
    d_phi = np.cos(alpha) * np.sin(beta)
    d_z = np.sin(alpha)
    d_x = d_r * np.cos(phi) - d_phi * np.sin(phi)
    d_y = d_r * np.sin(phi) + d_phi * np.cos(phi)
    norm = np.sqrt(d_x**2 + d_y**2 + d_z**2)
    return (float(d_x / norm), float(d_y / norm), float(d_z / norm))


def travis_profile(rho, central_value, a, p, q, h, w):
    """TRAVIS analytic profile formula.

    From TRAVIS source (plasma_profiles.f90):
    y(rho) = a + (1-a) * (1 - rho^p)^q + h*(1-exp(-rho^2/w^2))
    where TRAVIS divides input p by 2 internally and applies to z=rho^2.
    """
    rho = np.asarray(rho)

    # Adjust a if h is significant
    if abs(h) > 1e-6:
        a_adj = a - h
    else:
        a_adj = a
        h = 0

    # Compute hole contribution
    if abs(h) > 1e-6 and abs(w) > 1e-3:
        z_over_w2 = rho**2 / w**2
        z_over_w2 = np.clip(z_over_w2, 0, 40)  # Prevent overflow
        hole = h * (1 - np.exp(-z_over_w2))
    else:
        hole = 0

    # Main profile
    if q < 1e-6 and abs(h) < 1e-6:
        y = np.ones_like(rho)
    else:
        rho_clipped = np.clip(rho, 0, 1)
        y = a_adj + (1 - a_adj) * (1 - rho_clipped**p) ** q + hole

    return central_value * y


def find_b0_on_axis(wout):
    """Find |B| on the magnetic axis at phi=0, z=0."""
    from raytrax.equilibrium.interpolate import (
        build_magnetic_field_interpolator,
        build_rho_interpolator,
    )

    eq_unscaled = MagneticConfiguration.from_vmec_wout(wout)
    B_interp = build_magnetic_field_interpolator(eq_unscaled)
    rho_interp = build_rho_interpolator(eq_unscaled)

    # Scan R at phi=0, z=0 to find the axis (minimum rho)
    # At phi=0, z=0: no stellarator symmetry transformation needed
    best_rho, best_B = 999.0, 0.0
    for R in np.arange(5.0, 6.5, 0.002):
        rho_val = float(rho_interp(R, 0.0, 0.0))
        if rho_val < best_rho:
            best_rho = rho_val
            B_grid = B_interp(R, 0.0, 0.0)
            best_B = float(jnp.linalg.norm(B_grid))
    return best_B


def main():
    # Create data directory
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Load equilibrium from JSON and save to NetCDF for TRAVIS
    wout_nc_path = data_dir / "w7x.nc"

    print("Loading W7-X equilibrium from JSON...")
    wout = get_w7x_wout()

    # Get native B0 and compute scaling factor
    b0_native = find_b0_on_axis(wout)
    b_scale = B0_TARGET / b0_native
    print(f"B0 on axis (native): {b0_native:.4f} T")
    print(f"Target B0: {B0_TARGET:.4f} T")
    print(f"Scaling factor: {b_scale:.6f}")

    # Save unscaled equilibrium for TRAVIS (will use b0_normalization to scale)
    print(f"Saving equilibrium as NetCDF: {wout_nc_path}")
    wout.save(wout_nc_path)

    # For raytrax, apply magnetic_field_scale to match target
    eq_interp = MagneticConfiguration.from_vmec_wout(wout, magnetic_field_scale=b_scale)

    # W7-X realistic profiles matching TRAVIS input
    # Central_Ne_[1e20/m^3] 0.75, Ne-parm 0.05 20 2.8 0 0.32
    # Central_Te_[keV] 5, Te-parm 0.03 4 2 -0.2 0.5
    n_rho = 501  # High resolution for accurate interpolation
    rho = np.linspace(0, 1, n_rho)

    ne_central = 0.75  # 10^20 m^-3
    ne_parm = (0.05, 20, 2.8, 0, 0.32)
    ne_1e20 = travis_profile(rho, ne_central, *ne_parm)

    te_central = 5.0  # keV
    te_parm = (0.03, 4, 2.0, -0.2, 0.5)
    te_keV = travis_profile(rho, te_central, *te_parm)

    print("\nProfile values:")
    print(f"  ne(0) = {ne_1e20[0]:.4f} × 10^20 m^-3, Te(0) = {te_keV[0]:.4f} keV")
    print(
        f"  ne(0.5) = {ne_1e20[250]:.4f} × 10^20 m^-3, Te(0.5) = {te_keV[250]:.4f} keV"
    )
    print(f"  ne(1.0) = {ne_1e20[-1]:.4f} × 10^20 m^-3, Te(1.0) = {te_keV[-1]:.4f} keV")

    profiles = RadialProfiles(
        rho=jnp.array(rho),
        electron_density=jnp.array(ne_1e20),
        electron_temperature=jnp.array(te_keV),
    )

    # W7-X ECRH beam - exact parameters from TRAVIS input
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
        power=1e6,
    )

    # Run TRAVIS
    print("\n" + "=" * 80)
    print("Running TRAVIS...")
    print("=" * 80)

    TRAVIS_OUTPUT_DIR.mkdir(exist_ok=True)

    if TRAVIS_EXE is None or not TRAVIS_EXE.exists():
        print("WARNING: TRAVIS executable 'travis-nc' not found in PATH")
        print("Skipping TRAVIS comparison")
        travis_result = None
    else:
        travis_params = TravisECRHInput(
            antenna_position_cyl=jnp.array(antenna_cyl),
            target_position=jnp.array(target_w7x_angles + (0.0,)),
            frequency_ghz=140.0,
            power_mw=1.0,
            equilibrium_file=str(wout_nc_path),
            target_coords_type="W7X-angles",
            rho_grid=jnp.array(rho),
            electron_density_1e20=jnp.array(ne_1e20),
            electron_temperature_keV=jnp.array(te_keV),
            b0_normalization=B0_TARGET,  # TRAVIS will scale from native B0 to this
            dielectric_tracing="cold",
            hamiltonian="West",
            ne_parm=ne_parm,
            te_parm=te_parm,
        )

        travis_result = run_travis(
            TRAVIS_EXE, travis_params, output_dir=TRAVIS_OUTPUT_DIR
        )

        # Save reference data
        save_reference_data(travis_result, TRAVIS_REF_FILE)
        print(f"TRAVIS reference data saved to {TRAVIS_REF_FILE}")

        print(f"TRAVIS trajectory: {len(travis_result.arc_length_m)} points")
        print(
            f"TRAVIS arc length: {travis_result.arc_length_m[0]:.3f} -> {travis_result.arc_length_m[-1]:.3f} m"
        )
        print(f"TRAVIS final optical depth: {travis_result.optical_depth[-1]:.4f}")
        print(
            f"TRAVIS absorption: {100 * (1 - np.exp(-travis_result.optical_depth[-1])):.2f}%"
        )

    # Run raytrax
    print("\n" + "=" * 80)
    print("Running raytrax...")
    print("=" * 80)
    result = trace(eq_interp, profiles, beam)

    s = np.asarray(result.beam_profile.arc_length)
    tau = np.asarray(result.beam_profile.optical_depth)
    alpha = np.asarray(result.beam_profile.absorption_coefficient)
    ne_arr = np.asarray(result.beam_profile.electron_density)
    te_arr = np.asarray(result.beam_profile.electron_temperature)
    B_arr = np.asarray(result.beam_profile.magnetic_field)
    pos = np.asarray(result.beam_profile.position)
    rho_arr = np.asarray(result.beam_profile.normalized_effective_radius)

    print(
        f"raytrax trajectory: {len(s)} points, arc length {s[0]:.3f} -> {s[-1]:.3f} m"
    )
    print(f"raytrax final optical depth: tau = {tau[-1]:.4f}")
    print(f"raytrax absorption: {100 * (1 - np.exp(-tau[-1])):.2f}%")
    print(f"raytrax max absorption coefficient: {np.nanmax(alpha):.2f} m^-1")

    # Comparison
    if travis_result is not None:
        print("\n" + "=" * 80)
        print("COMPARISON: raytrax vs TRAVIS")
        print("=" * 80)

        # Find where TRAVIS enters plasma (rho < 1)
        travis_in_plasma = travis_result.rho < 1.0
        if np.any(travis_in_plasma):
            travis_lcms_idx = np.where(travis_in_plasma)[0][0]
            s_lcms_travis = travis_result.arc_length_m[travis_lcms_idx]
            print(f"\nTRAVIS LCMS entry: s = {s_lcms_travis:.4f} m")
            print(f"raytrax starts at:  s = {s[0]:.4f} m")

        from scipy.interpolate import interp1d

        # Use absolute arc length comparison (no LCMS alignment shift)
        s_travis = travis_result.arc_length_m
        pos_travis = travis_result.position_m

        # Find overlapping arc length region
        s_min = max(s[0], s_travis[0])
        s_max = min(s[-1], s_travis[-1])

        mask_rx = (s >= s_min) & (s <= s_max)

        # Interpolate TRAVIS data onto raytrax arc length points
        tr_pos_interp = interp1d(
            s_travis,
            pos_travis,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        tr_rho_interp = interp1d(
            s_travis,
            travis_result.rho,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        tr_B_interp = interp1d(
            s_travis,
            travis_result.magnetic_field_magnitude_T,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        pos_travis_at_rx = tr_pos_interp(s[mask_rx])
        rho_travis_at_rx = tr_rho_interp(s[mask_rx])
        B_travis_at_rx = tr_B_interp(s[mask_rx])

        # Evaluate profiles at interpolated rho
        ne_travis_at_rx = travis_profile(rho_travis_at_rx, ne_central, *ne_parm)
        te_travis_at_rx = travis_profile(rho_travis_at_rx, te_central, *te_parm)

        # Compute differences
        pos_diff = pos[mask_rx] - pos_travis_at_rx
        pos_dist = np.linalg.norm(pos_diff, axis=1)

        B_arr_rx = np.array([np.linalg.norm(B_arr[i]) for i in np.where(mask_rx)[0]])
        B_diff = B_arr_rx - B_travis_at_rx

        ne_arr_rx = ne_arr[mask_rx]
        ne_diff = ne_arr_rx - ne_travis_at_rx

        te_arr_rx = te_arr[mask_rx]
        te_diff = te_arr_rx - te_travis_at_rx

        rho_arr_rx = rho_arr[mask_rx]
        rho_diff = rho_arr_rx - rho_travis_at_rx

        print("\nComparison statistics:")
        print(f"  XYZ Position: RMS = {np.sqrt(np.mean(pos_dist**2)) * 1000:.2f} mm")
        print(
            f"  |B|: Mean error = {np.mean(np.abs(B_diff / B_travis_at_rx)) * 100:.2f}%"
        )
        print(
            f"  ne:  Mean error = {np.mean(np.abs(ne_diff / ne_travis_at_rx)) * 100:.2f}%"
        )
        print(
            f"  Te:  Mean error = {np.mean(np.abs(te_diff / te_travis_at_rx)) * 100:.2f}%"
        )
        print(f"  rho: RMS = {np.sqrt(np.mean(rho_diff**2)):.4f}")

        # Detailed tabulation - write to file
        table_file = Path(__file__).parent / "data" / "rho_comparison.txt"
        with open(table_file, "w") as f:
            f.write(f"{'=' * 100}\n")
            f.write("DETAILED ρ vs s COMPARISON\n")
            f.write(f"{'=' * 100}\n")
            f.write(
                f"{'s [m]':>10} {'ρ_raytrax':>12} {'ρ_TRAVIS':>12} {'Δρ':>10} {'ne_rx':>10} {'ne_tr':>10} {'Te_rx':>10} {'Te_tr':>10}\n"
            )
            f.write(f"{'-' * 100}\n")
            for i in range(len(rho_arr_rx)):
                s_val = s[mask_rx][i]
                f.write(
                    f"{s_val:10.4f} {rho_arr_rx[i]:12.4f} {rho_travis_at_rx[i]:12.4f} {rho_diff[i]:10.4f} "
                    f"{ne_arr_rx[i]:10.4f} {ne_travis_at_rx[i]:10.4f} {te_arr_rx[i]:10.3f} {te_travis_at_rx[i]:10.3f}\n"
                )
            f.write(f"{'=' * 100}\n")
        print(f"\nDetailed ρ vs s comparison saved to: {table_file}")

        # Debug: Check vacuum propagation and rho gradient
        print(f"\n{'=' * 80}")
        print("DEBUG: Vacuum propagation and ρ gradient check")
        print(f"{'=' * 80}")

        # Check if rays are actually in vacuum at start
        idx_rx_first = 0
        s_rx_first = s[mask_rx][idx_rx_first]
        xyz_rx_first = pos[mask_rx][idx_rx_first]
        rho_rx_first = rho_arr_rx[idx_rx_first]

        print(f"\nraytrax first point at s={s_rx_first:.4f} m:")
        print(
            f"  XYZ: [{xyz_rx_first[0]:.6f}, {xyz_rx_first[1]:.6f}, {xyz_rx_first[2]:.6f}] m"
        )
        print(f"  ρ_raytrax: {rho_rx_first:.6f}")
        print(f"  Is this in plasma (ρ<1)? {rho_rx_first < 1.0}")

        # Find corresponding TRAVIS point - but check actual TRAVIS data, not interpolated
        print(f"\nTRAVIS points near s={s_rx_first:.4f} m (aligned):")
        s_travis_aligned = travis_result.arc_length_m + (s[0] - s_lcms_travis)
        for i in range(
            max(0, travis_lcms_idx - 2),
            min(len(travis_result.arc_length_m), travis_lcms_idx + 5),
        ):
            print(
                f"  idx={i}: s={s_travis_aligned[i]:.4f} m, XYZ=[{travis_result.position_m[i, 0]:.6f}, {travis_result.position_m[i, 1]:.6f}, {travis_result.position_m[i, 2]:.6f}], ρ={travis_result.rho[i]:.6f}"
            )

        # Check ρ gradient at edge
        print("\nρ gradient test at edge:")
        from raytrax.equilibrium.interpolate import build_rho_interpolator

        rho_fn = build_rho_interpolator(eq_interp)

        # Sample points along a line near the edge
        test_positions = [
            xyz_rx_first,
            xyz_rx_first + np.array([0.001, 0, 0]),
            xyz_rx_first + np.array([0.005, 0, 0]),
            xyz_rx_first + np.array([0.010, 0, 0]),
        ]
        for i, xyz in enumerate(test_positions):
            # Convert to cylindrical coordinates for interpolator
            r = float(np.sqrt(xyz[0] ** 2 + xyz[1] ** 2))
            phi = float(np.arctan2(xyz[1], xyz[0]))
            z = float(xyz[2])
            rho_val = float(rho_fn(r, phi, z))
            dist = np.linalg.norm(xyz - xyz_rx_first) * 1000
            print(
                f"  +{dist:.1f}mm in X: ρ={rho_val:.6f}, Δρ={rho_val - rho_rx_first:.6f}"
            )

        # Check antenna position
        print("\nBeam starting position (antenna):")
        print(
            f"  XYZ: [{antenna_cart[0]:.6f}, {antenna_cart[1]:.6f}, {antenna_cart[2]:.6f}] m"
        )
        print(
            f"  Nominal direction: [{direction[0]:.6f}, {direction[1]:.6f}, {direction[2]:.6f}]"
        )

        # Compute actual directions from antenna to first points
        vec_to_rx_first = xyz_rx_first - np.array(antenna_cart)
        dist_to_rx_first = np.linalg.norm(vec_to_rx_first)
        dir_rx_actual = vec_to_rx_first / dist_to_rx_first

        # Find first TRAVIS point
        xyz_tr_first = travis_result.position_m[travis_lcms_idx]
        vec_to_tr_first = xyz_tr_first - np.array(antenna_cart)
        dist_to_tr_first = np.linalg.norm(vec_to_tr_first)
        dir_tr_actual = vec_to_tr_first / dist_to_tr_first

        print("\nraytrax first point:")
        print(f"  Distance from antenna: {dist_to_rx_first:.6f} m")
        print(
            f"  Actual direction: [{dir_rx_actual[0]:.6f}, {dir_rx_actual[1]:.6f}, {dir_rx_actual[2]:.6f}]"
        )
        print(
            f"  Direction error: {np.linalg.norm(dir_rx_actual - np.array(direction)) * 1e6:.2f} ppm"
        )

        print("\nTRAVIS first point:")
        print(f"  Distance from antenna: {dist_to_tr_first:.6f} m")
        print(
            f"  Actual direction: [{dir_tr_actual[0]:.6f}, {dir_tr_actual[1]:.6f}, {dir_tr_actual[2]:.6f}]"
        )
        print(
            f"  Direction error: {np.linalg.norm(dir_tr_actual - np.array(direction)) * 1e6:.2f} ppm"
        )

        print("\nDirection difference (raytrax - TRAVIS):")
        dir_diff = dir_rx_actual - dir_tr_actual
        print(f"  Δdir: [{dir_diff[0]:.8f}, {dir_diff[1]:.8f}, {dir_diff[2]:.8f}]")
        print(f"  |Δdir|: {np.linalg.norm(dir_diff) * 1e6:.2f} ppm")

        print(f"{'=' * 80}\n")

        print("\nOptical depth:")
        print(f"  raytrax: {tau[-1]:.4f}")
        print(f"  TRAVIS:  {travis_result.optical_depth[-1]:.4f}")
        rel_error = (
            abs(tau[-1] - travis_result.optical_depth[-1])
            / travis_result.optical_depth[-1]
            * 100
        )
        print(f"  Error:   {rel_error:.2f}%")

        print("\nAbsorption fraction:")
        abs_rx = 1 - np.exp(-tau[-1])
        abs_tr = 1 - np.exp(-travis_result.optical_depth[-1])
        print(f"  raytrax: {abs_rx * 100:.2f}%")
        print(f"  TRAVIS:  {abs_tr * 100:.2f}%")
        print(f"  Diff:    {abs(abs_rx - abs_tr) * 100:.2f} percentage points")

        # Check if agreement is good
        if rel_error < 5.0:
            print(f"\n✓ PASS: Optical depth agrees within 5% (error: {rel_error:.2f}%)")
        else:
            print(
                f"\n✗ FAIL: Optical depth error {rel_error:.2f}% exceeds 5% threshold"
            )


if __name__ == "__main__":
    main()
