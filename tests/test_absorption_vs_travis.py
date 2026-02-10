"""Compare raytrax absorption with TRAVIS reference implementation."""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
from vmecpp import VmecWOut

from raytrax.api import get_interpolator_for_equilibrium, trace
from raytrax.types import Beam, RadialProfiles
from travis_wrapper import run_travis, TravisECRHInput, save_reference_data

WOUT_FILE = "/home/david/Code/Forschung/travis_2018u/projects/files_W7X/MagnConfigs/wout-w7x-hm13.nc"
TRAVIS_EXE = Path("/home/david/Code/Forschung/travis_2018u/bin/travis-nc")
TRAVIS_OUTPUT_DIR = Path("/tmp/travis_comparison_output")
TRAVIS_REF_FILE = Path("tests/data/travis_w7x_reference.json")
B0_TARGET = 2.52076  # TRAVIS B0 normalization at angle 0 on magnetic axis


def cylindrical_to_cartesian(r_m, phi_deg, z_m):
    phi_rad = np.deg2rad(phi_deg)
    x = r_m * np.cos(phi_rad)
    y = r_m * np.sin(phi_rad)
    return (float(x), float(y), float(z_m))


def w7x_aiming_angles_to_direction(theta_pol_deg, theta_tor_deg, antenna_phi_deg):
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
    """
    TRAVIS analytic profile formula.
    
    From TRAVIS source (plasma_profiles.f90):
    TRAVIS uses z = rho^2 (normalized flux), then:
    p_internal = p/2  (divided in code)
    y(z) = a + (1-a) * (1 - z^p_internal)^q + h*(1-exp(-z/w^2))
    
    Which simplifies to:
    y(rho) = a + (1-a) * (1 - (rho^2)^(p/2))^q + h*(1-exp(-rho^2/w^2))
           = a + (1-a) * (1 - rho^p)^q + h*(1-exp(-rho^2/w^2))
    
    If h != 0, TRAVIS first subtracts h from a.
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
    
    # Main profile: use rho^p not rho^(p/2)!
    # TRAVIS divides p by 2 internally, but applies it to z=rho^2, so (rho^2)^(p/2) = rho^p
    if q < 1e-6 and abs(h) < 1e-6:
        y = np.ones_like(rho)
    else:
        rho_clipped = np.clip(rho, 0, 1)
        y = a_adj + (1 - a_adj) * (1 - rho_clipped**p)**q + hole
    
    return central_value * y


def find_b0_on_axis(wout):
    """Find |B| on the magnetic axis at phi=0 from an unscaled equilibrium."""
    from raytrax.interpolate import build_magnetic_field_interpolator, build_rho_interpolator
    eq_unscaled = get_interpolator_for_equilibrium(wout)
    B_fn = build_magnetic_field_interpolator(eq_unscaled)
    rho_fn = build_rho_interpolator(eq_unscaled)
    # Scan R at phi=0, z=0 to find the axis (minimum rho)
    best_rho, best_B = 999.0, 0.0
    for R in np.arange(5.0, 6.5, 0.002):
        pos = jnp.array([R, 0.0, 0.0])
        rho_val = float(rho_fn(pos))
        if rho_val < best_rho:
            best_rho = rho_val
            best_B = float(jnp.linalg.norm(B_fn(pos)))
    return best_B


def main():
    print(f"Loading equilibrium from {WOUT_FILE}")
    wout = VmecWOut.from_wout_file(WOUT_FILE)

    # Compute B0 scaling: match TRAVIS's "at angle on magn.axis" normalization
    b0_actual = find_b0_on_axis(wout)
    b_scale = B0_TARGET / b0_actual
    print(f"B0 on axis (unscaled): {b0_actual:.4f} T, target: {B0_TARGET} T, scale: {b_scale:.6f}")

    eq_interp = get_interpolator_for_equilibrium(wout, magnetic_field_scale=b_scale)

    # W7-X realistic profiles matching TRAVIS input
    # Central_Ne_[1e20/m^3] 0.75, Ne-parm 0.05 20 2.8 0 0.32
    # Central_Te_[keV] 5, Te-parm 0.03 4 2 -0.2 0.5
    n_rho = 501  # High resolution for accurate interpolation
    rho = np.linspace(0, 1, n_rho)
    
    ne_central = 0.75  # 10^20 m^-3
    ne_parm = (0.05, 20, 2.8, 0, 0.32)
    ne_1e20 = travis_profile(rho, ne_central, *ne_parm)
    
    te_central = 5.0   # keV
    te_parm = (0.03, 4, 2.0, -0.2, 0.5)
    te_keV = travis_profile(rho, te_central, *te_parm)
    
    print(f"\nProfile values:")
    print(f"  ne(0) = {ne_1e20[0]:.4f} × 10^20 m^-3, Te(0) = {te_keV[0]:.4f} keV")
    print(f"  ne(0.5) = {ne_1e20[25]:.4f} × 10^20 m^-3, Te(0.5) = {te_keV[25]:.4f} keV")
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
    )

    # Run TRAVIS
    print("\n" + "="*80)
    print("Running TRAVIS...")
    print("="*80)
    
    TRAVIS_OUTPUT_DIR.mkdir(exist_ok=True)
    
    if not TRAVIS_EXE.exists():
        print(f"WARNING: TRAVIS executable not found at {TRAVIS_EXE}")
        print("Skipping TRAVIS comparison")
        travis_result = None
    else:
        travis_params = TravisECRHInput(
            antenna_position_cyl=jnp.array(antenna_cyl),
            target_position=jnp.array(target_w7x_angles + (0.0,)),
            frequency_ghz=140.0,
            power_mw=1.0,
            equilibrium_file=WOUT_FILE,
            target_coords_type="W7X-angles",
            rho_grid=jnp.array(rho),
            electron_density_1e20=jnp.array(ne_1e20),
            electron_temperature_keV=jnp.array(te_keV),
            b0_normalization=B0_TARGET,
            dielectric_tracing="cold",
            hamiltonian="West",
            ne_parm=ne_parm,
            te_parm=te_parm,
        )
        
        travis_result = run_travis(TRAVIS_EXE, travis_params, output_dir=TRAVIS_OUTPUT_DIR)
        
        # Save reference data
        save_reference_data(travis_result, TRAVIS_REF_FILE)
        print(f"TRAVIS reference data saved to {TRAVIS_REF_FILE}")
        
        print(f"TRAVIS trajectory: {len(travis_result.arc_length_m)} points")
        print(f"TRAVIS arc length: {travis_result.arc_length_m[0]:.3f} -> {travis_result.arc_length_m[-1]:.3f} m")
        print(f"TRAVIS final optical depth: {travis_result.optical_depth[-1]:.4f}")
        print(f"TRAVIS absorption fraction: {1 - np.exp(-travis_result.optical_depth[-1]):.4f} ({100*(1 - np.exp(-travis_result.optical_depth[-1])):.2f}%)")
        print(f"TRAVIS max absorption: {np.max(travis_result.absorption_m_inv):.2f} m^-1")
        print(f"TRAVIS output saved to: {TRAVIS_OUTPUT_DIR}")

    # Run raytrax
    print("\n" + "="*80)
    print("Running raytrax...")
    print("="*80)
    
    result = trace(eq_interp, profiles, beam)

    s = np.asarray(result.beam_profile.arc_length)
    tau = np.asarray(result.beam_profile.optical_depth)
    alpha = np.asarray(result.beam_profile.absorption_coefficient)
    ne_arr = np.asarray(result.beam_profile.electron_density)
    te_arr = np.asarray(result.beam_profile.electron_temperature)
    B_arr = np.asarray(result.beam_profile.magnetic_field)
    pos = np.asarray(result.beam_profile.position)
    N_arr = np.asarray(result.beam_profile.refractive_index)
    rho_arr = np.asarray(result.beam_profile.normalized_effective_radius)

    print(f"raytrax trajectory: {len(s)} points, arc length {s[0]:.3f} -> {s[-1]:.3f} m")
    print(f"raytrax final optical depth: tau = {tau[-1]:.4f}")
    print(f"raytrax max absorption coefficient: {np.nanmax(alpha):.2f} m^-1")

    # Comparison
    if travis_result is not None:
        print("\n" + "="*80)
        print("COMPARISON: raytrax vs TRAVIS")
        print("="*80)
        
        # Note: raytrax starts at LCMS (s_rx[0] ~ 0.54 m), TRAVIS starts at antenna (s_tr[0] = 0)
        # Find where TRAVIS enters plasma (rho < 1)
        travis_in_plasma = travis_result.rho < 1.0
        if np.any(travis_in_plasma):
            travis_lcms_idx = np.where(travis_in_plasma)[0][0]
            s_lcms_travis = travis_result.arc_length_m[travis_lcms_idx]
            print(f"\nTRAVIS LCMS entry: s = {s_lcms_travis:.4f} m")
            print(f"raytrax starts at:  s = {s[0]:.4f} m")
            print(f"Difference (antenna to LCMS): {abs(s[0] - s_lcms_travis):.4f} m")
        
        # Compare XYZ trajectories - align by offsetting TRAVIS arc length
        from scipy.interpolate import interp1d
        
        # Offset TRAVIS arc length to match raytrax start
        s_travis_aligned = travis_result.arc_length_m + (s[0] - s_lcms_travis)
        
        # Find overlapping region
        s_min = max(s[0], s_travis_aligned[travis_lcms_idx])
        s_max = min(s[-1], s_travis_aligned[-1])
        
        mask_rx = (s >= s_min) & (s <= s_max)
        mask_tr = (s_travis_aligned >= s_min) & (s_travis_aligned <= s_max)
        
        print(f"\nOverlapping arc length region: {s_min:.4f} to {s_max:.4f} m")
        print(f"  raytrax points in region: {np.sum(mask_rx)}")
        print(f"  TRAVIS points in region:  {np.sum(mask_tr)}")
        
        # Interpolate TRAVIS onto raytrax points for comparison
        tr_pos_interp = interp1d(s_travis_aligned[travis_lcms_idx:], 
                                 travis_result.position_m[travis_lcms_idx:], axis=0,
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        tr_tau_interp = interp1d(s_travis_aligned[travis_lcms_idx:], 
                                 travis_result.optical_depth[travis_lcms_idx:],
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        tr_alpha_interp = interp1d(s_travis_aligned[travis_lcms_idx:], 
                                   travis_result.absorption_m_inv[travis_lcms_idx:],
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
        tr_rho_interp = interp1d(s_travis_aligned[travis_lcms_idx:], 
                                 travis_result.rho[travis_lcms_idx:],
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        
        pos_travis_at_rx = tr_pos_interp(s[mask_rx])
        tau_travis_at_rx = tr_tau_interp(s[mask_rx])
        alpha_travis_at_rx = tr_alpha_interp(s[mask_rx])
        rho_travis_at_rx = tr_rho_interp(s[mask_rx])
        
        # Evaluate profiles at the interpolated rho values
        # This is correct since profiles only depend on rho, not arc length!
        ne_travis_at_rx = travis_profile(rho_travis_at_rx, ne_central, *ne_parm)
        te_travis_at_rx = travis_profile(rho_travis_at_rx, te_central, *te_parm)
        
        # |B| needs to be interpolated in arc length (depends on position, not just rho)
        tr_B_interp = interp1d(s_travis_aligned[travis_lcms_idx:], 
                               travis_result.magnetic_field_magnitude_T[travis_lcms_idx:],
                               kind='linear', bounds_error=False, fill_value='extrapolate')
        B_travis_at_rx = tr_B_interp(s[mask_rx])
        
        pos_diff = pos[mask_rx] - pos_travis_at_rx
        pos_dist = np.linalg.norm(pos_diff, axis=1)
        tau_diff = tau[mask_rx] - tau_travis_at_rx
        alpha_diff = alpha[mask_rx] - alpha_travis_at_rx
        
        B_arr_rx = np.array([np.linalg.norm(B_arr[i]) for i in np.where(mask_rx)[0]])
        B_diff = B_arr_rx - B_travis_at_rx
        ne_arr_rx = ne_arr[mask_rx]
        ne_diff = ne_arr_rx - ne_travis_at_rx
        te_arr_rx = te_arr[mask_rx]
        te_diff = te_arr_rx - te_travis_at_rx
        rho_arr_rx = rho_arr[mask_rx]
        rho_diff = rho_arr_rx - rho_travis_at_rx
        
        print(f"\n{'='*80}")
        print("XYZ Position Comparison [m]:")
        print(f"  Mean distance:   {np.mean(pos_dist):.6f}")
        print(f"  Max distance:    {np.max(pos_dist):.6f}")
        print(f"  RMS distance:    {np.sqrt(np.mean(pos_dist**2)):.6f}")
        
        print(f"\n|B| Comparison [T]:")
        print(f"  Mean abs. diff:  {np.mean(np.abs(B_diff)):.6f}")
        print(f"  Max abs. diff:   {np.max(np.abs(B_diff)):.6f}")
        print(f"  RMS diff:        {np.sqrt(np.mean(B_diff**2)):.6f}")
        print(f"  Mean rel. error: {np.mean(np.abs(B_diff/B_travis_at_rx))*100:.4f}%")
        
        print(f"\nne Comparison [1e20 m^-3]:")
        print(f"  Mean abs. diff:  {np.mean(np.abs(ne_diff)):.6f}")
        print(f"  Max abs. diff:   {np.max(np.abs(ne_diff)):.6f}")
        print(f"  RMS diff:        {np.sqrt(np.mean(ne_diff**2)):.6f}")
        print(f"  Mean rel. error: {np.mean(np.abs(ne_diff/ne_travis_at_rx))*100:.4f}%")
        
        print(f"\nTe Comparison [keV]:")
        print(f"  Mean abs. diff:  {np.mean(np.abs(te_diff)):.6f}")
        print(f"  Max abs. diff:   {np.max(np.abs(te_diff)):.6f}")
        print(f"  RMS diff:        {np.sqrt(np.mean(te_diff**2)):.6f}")
        print(f"  Mean rel. error: {np.mean(np.abs(te_diff/te_travis_at_rx))*100:.4f}%")
        
        print(f"\nrho Comparison:")
        print(f"  Mean abs. diff:  {np.mean(np.abs(rho_diff)):.6f}")
        print(f"  Max abs. diff:   {np.max(np.abs(rho_diff)):.6f}")
        print(f"  RMS diff:        {np.sqrt(np.mean(rho_diff**2)):.6f}")
        
        print(f"\nOptical depth (tau):")
        print(f"  raytrax final:  {tau[-1]:.4f}")
        print(f"  TRAVIS final:   {travis_result.optical_depth[-1]:.4f}")
        print(f"  Difference:     {tau[-1] - travis_result.optical_depth[-1]:.4f}")
        if travis_result.optical_depth[-1] > 0:
            print(f"  Rel. error:     {abs(tau[-1] - travis_result.optical_depth[-1])/travis_result.optical_depth[-1]*100:.2f}%")
        
        print(f"\nAbsorption coefficient (alpha) [m^-1]:")
        print(f"  Mean abs. diff:  {np.mean(np.abs(alpha_diff)):.4f}")
        print(f"  Max abs. diff:   {np.max(np.abs(alpha_diff)):.4f}")
        print(f"  RMS diff:        {np.sqrt(np.mean(alpha_diff**2)):.4f}")
        
        # Print side-by-side comparison
        print(f"\n{'='*80}")
        print("Side-by-side comparison at selected points:")
        print(f"{'s [m]':>8} {'rho':>7} | {'|B| [T]':>8} {'ne':>6} {'Te':>6} | XYZ distance [mm]")
        print(f"{' '*16} | raytrax/TRAVIS diff")
        print(f"{'-'*80}")
        step = max(1, np.sum(mask_rx) // 10)
        rx_indices = np.where(mask_rx)[0]
        for i in range(0, len(rx_indices), step):
            idx = rx_indices[i]
            idx_in_mask = i
            dist = pos_dist[idx_in_mask] * 1000  # convert to mm
            B_rx = B_arr_rx[idx_in_mask]
            B_tr = B_travis_at_rx[idx_in_mask]
            ne_rx = ne_arr_rx[idx_in_mask]
            ne_tr = ne_travis_at_rx[idx_in_mask]
            te_rx = te_arr_rx[idx_in_mask]
            te_tr = te_travis_at_rx[idx_in_mask]
            rho_rx = rho_arr_rx[idx_in_mask]
            rho_tr = rho_travis_at_rx[idx_in_mask]
            print(f"{s[idx]:8.4f} {rho_rx:7.3f} | {B_rx:8.4f} {ne_rx:6.3f} {te_rx:6.2f} | {dist:6.2f}")
            print(f"         {rho_tr:7.3f} | {B_tr:8.4f} {ne_tr:6.3f} {te_tr:6.2f} |")
        # Last point
        idx = rx_indices[-1]
        idx_in_mask = -1
        dist = pos_dist[idx_in_mask] * 1000
        B_rx = B_arr_rx[idx_in_mask]
        B_tr = B_travis_at_rx[idx_in_mask]
        ne_rx = ne_arr_rx[idx_in_mask]
        ne_tr = ne_travis_at_rx[idx_in_mask]
        te_rx = te_arr_rx[idx_in_mask]
        te_tr = te_travis_at_rx[idx_in_mask]
        rho_rx = rho_arr_rx[idx_in_mask]
        rho_tr = rho_travis_at_rx[idx_in_mask]
        print(f"{s[idx]:8.4f} {rho_rx:7.3f} | {B_rx:8.4f} {ne_rx:6.3f} {te_rx:6.2f} | {dist:6.2f}")
        print(f"         {rho_tr:7.3f} | {B_tr:8.4f} {ne_tr:6.3f} {te_tr:6.2f} |")

    # Print detailed profile
    print("\n" + "="*80)
    print("DETAILED PROFILE (raytrax)")
    print("="*80)
    print(f"\n{'s [m]':>8}  {'rho':>6}  {'|B| [T]':>8}  {'ne [1e20]':>10}  "
          f"{'Te [keV]':>10}  {'tau':>10}  {'alpha [1/m]':>12}  "
          f"{'x [m]':>8}  {'y [m]':>8}  {'z [m]':>8}")
    step = max(1, len(s) // 30)
    for i in list(range(0, len(s), step)) + [len(s) - 1]:
        B_mag = np.linalg.norm(B_arr[i])
        print(
            f"{s[i]:8.4f}  {rho_arr[i]:6.3f}  {B_mag:8.4f}  {ne_arr[i]:10.4f}  "
            f"{te_arr[i]:10.4f}  {tau[i]:10.4f}  {alpha[i]:12.4f}  "
            f"{pos[i, 0]:8.4f}  {pos[i, 1]:8.4f}  {pos[i, 2]:8.4f}"
        )
    print(f"\nraytrax transmitted power: exp(-tau) = {np.exp(-tau[-1]):.4f} ({100*np.exp(-tau[-1]):.2f}%)")
    print(f"raytrax absorption fraction: {1 - np.exp(-tau[-1]):.4f} ({100*(1 - np.exp(-tau[-1])):.2f}%)")
    if travis_result is not None:
        print(f"TRAVIS transmitted power:  exp(-tau) = {np.exp(-travis_result.optical_depth[-1]):.4f} ({100*np.exp(-travis_result.optical_depth[-1]):.2f}%)")
        print(f"TRAVIS absorption fraction: {1 - np.exp(-travis_result.optical_depth[-1]):.4f} ({100*(1 - np.exp(-travis_result.optical_depth[-1])):.2f}%)")
        
        # Detailed rho comparison table
        print("\n" + "="*80)
        print("RHO COMPARISON ALONG TRAJECTORY")
        print("="*80)
        print(f"\n{'s [m]':>8} | {'rho_rx':>8} {'rho_tr':>8} {'diff':>8} | {'|B|_rx [T]':>10} {'|B|_tr [T]':>10} | {'XYZ dist [mm]':>14}")
        print("-" * 80)
        
        rx_indices = np.where(mask_rx)[0]
        step = max(1, len(rx_indices) // 25)  # Show ~25 points
        for i in range(0, len(rx_indices), step):
            idx = rx_indices[i]
            idx_in_mask = i
            rho_rx = rho_arr_rx[idx_in_mask]
            rho_tr = rho_travis_at_rx[idx_in_mask]
            rho_diff = rho_rx - rho_tr
            B_rx = B_arr_rx[idx_in_mask]
            B_tr = B_travis_at_rx[idx_in_mask]
            xyz_dist = pos_dist[idx_in_mask] * 1000
            print(f"{s[idx]:8.4f} | {rho_rx:8.4f} {rho_tr:8.4f} {rho_diff:+8.4f} | {B_rx:10.4f} {B_tr:10.4f} | {xyz_dist:14.2f}")
        
        # Last point
        idx = rx_indices[-1]
        idx_in_mask = -1
        rho_rx = rho_arr_rx[idx_in_mask]
        rho_tr = rho_travis_at_rx[idx_in_mask]
        rho_diff = rho_rx - rho_tr
        B_rx = B_arr_rx[idx_in_mask]
        B_tr = B_travis_at_rx[idx_in_mask]
        xyz_dist = pos_dist[idx_in_mask] * 1000
        print(f"{s[idx]:8.4f} | {rho_rx:8.4f} {rho_tr:8.4f} {rho_diff:+8.4f} | {B_rx:10.4f} {B_tr:10.4f} | {xyz_dist:14.2f}")


if __name__ == "__main__":
    main()
