"""Integration tests comparing raytrax with TRAVIS ECRH simulations."""

from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pytest

from raytrax.api import get_interpolator_for_equilibrium, trace
from raytrax.types import Beam, RadialProfiles
from tests.fixtures import w7x_wout, w7x_travis_wout
from tests.travis_wrapper import load_reference_data

REFERENCE_DATA_PATH = Path(__file__).parent / "data" / "travis_reference_w7x.json"


def cylindrical_to_cartesian(r_m: float, phi_deg: float, z_m: float) -> tuple[float, float, float]:
    """Convert cylindrical to Cartesian coordinates."""
    phi_rad = np.deg2rad(phi_deg)
    x = r_m * np.cos(phi_rad)
    y = r_m * np.sin(phi_rad)
    return (float(x), float(y), float(z_m))


def w7x_aiming_angles_to_direction(
    theta_pol_deg: float, 
    theta_tor_deg: float,
    antenna_phi_deg: float
) -> tuple[float, float, float]:
    """Convert W7X aiming angles to Cartesian direction vector.
    
    Args:
        theta_pol_deg: Poloidal aiming angle in degrees
        theta_tor_deg: Toroidal aiming angle in degrees
        antenna_phi_deg: Toroidal angle of antenna position in degrees
    
    Returns:
        Normalized direction vector in Cartesian coordinates (x, y, z)
    
    This implements the same formula as TRAVIS:
        alpha = theta_pol * rad
        beta = theta_tor * rad
        d_cyl = [-cos(alpha)*cos(beta), cos(alpha)*sin(beta), sin(alpha)]
        d_cart = cyl2cart('v', d_cyl, phi)
    """
    alpha = np.deg2rad(theta_pol_deg)
    beta = np.deg2rad(theta_tor_deg)
    phi = np.deg2rad(antenna_phi_deg)
    
    # Direction in cylindrical coordinates (Nr, Nphi, Nz)
    d_r = -np.cos(alpha) * np.cos(beta)
    d_phi = np.cos(alpha) * np.sin(beta)
    d_z = np.sin(alpha)
    
    # Convert vector from cylindrical to Cartesian
    # XYZ(1) = RpZ(1)*cos(phi) - RpZ(2)*sin(phi)
    # XYZ(2) = RpZ(1)*sin(phi) + RpZ(2)*cos(phi)
    # XYZ(3) = RpZ(3)
    d_x = d_r * np.cos(phi) - d_phi * np.sin(phi)
    d_y = d_r * np.sin(phi) + d_phi * np.cos(phi)
    d_z_cart = d_z
    
    # Normalize
    norm = np.sqrt(d_x**2 + d_y**2 + d_z_cart**2)
    return (float(d_x/norm), float(d_y/norm), float(d_z_cart/norm))


def create_matching_plasma_profiles() -> RadialProfiles:
    """Create plasma profiles matching TRAVIS reference run."""
    n_rho = 51
    rho = np.linspace(0, 1, n_rho)
    ne_1e20 = 1.0 * (1 - rho**2)
    te_keV = 3.0 * (1 - rho**2)
    
    return RadialProfiles(
        rho=jnp.array(rho),
        electron_density=jnp.array(ne_1e20),
        electron_temperature=jnp.array(te_keV),
    )


def interpolate_to_common_grid(
    arc_length_m_1: np.ndarray,
    values_1: np.ndarray,
    arc_length_m_2: np.ndarray,
    values_2: np.ndarray,
    n_points: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate two trajectories to a common arc length grid."""
    s_min = max(arc_length_m_1[0], arc_length_m_2[0])
    s_max = min(arc_length_m_1[-1], arc_length_m_2[-1])
    common_s = np.linspace(s_min, s_max, n_points)
    interp_1 = np.interp(common_s, arc_length_m_1, values_1)
    interp_2 = np.interp(common_s, arc_length_m_2, values_2)
    return common_s, interp_1, interp_2


@pytest.mark.skipif(
    not REFERENCE_DATA_PATH.exists(),
    reason="TRAVIS reference data not available. Run generate_travis_reference.py first."
)
def test_compare_trajectory_with_travis(w7x_travis_wout):
    """Compare raytrax ray trajectory with TRAVIS reference data.

    Uses the same equilibrium file as TRAVIS (set via TRAVIS_W7X_WOUT env var).
    """
    travis_data = load_reference_data(REFERENCE_DATA_PATH)

    equilibrium_interpolator = get_interpolator_for_equilibrium(w7x_travis_wout)
    plasma_profiles = create_matching_plasma_profiles()
    
    # Use same parameters as TRAVIS reference generation
    antenna_cyl = (6.509, -6.564, 0.38)  # (R_m, phi_deg, Z_m)
    target_w7x_angles = (15.7, 19.7)  # (theta_pol_deg, theta_tor_deg)
    
    antenna_cart = cylindrical_to_cartesian(*antenna_cyl)
    direction = w7x_aiming_angles_to_direction(target_w7x_angles[0], target_w7x_angles[1], antenna_cyl[1])
    
    beam = Beam(
        position=jnp.array(antenna_cart),
        direction=jnp.array(direction),
        frequency=jnp.array(140e9),
        mode="O",
    )
    
    result = trace(equilibrium_interpolator, plasma_profiles, beam)
    
    raytrax_pos = result.beam_profile.position
    raytrax_s_m = result.beam_profile.arc_length
    
    travis_pos_m = travis_data.position_m
    travis_s_m = travis_data.arc_length_m
    
    n_compare = 50
    s_common, raytrax_x_interp, travis_x_interp = interpolate_to_common_grid(
        np.asarray(raytrax_s_m), np.asarray(raytrax_pos[:, 0]),
        np.asarray(travis_s_m), np.asarray(travis_pos_m[:, 0]),
        n_compare
    )
    
    position_error_m = np.abs(raytrax_x_interp - travis_x_interp)
    max_position_error_cm = np.max(position_error_m) * 100
    mean_position_error_cm = np.mean(position_error_m) * 100
    
    print(f"\nTrajectory comparison:")
    print(f"  Max position error: {max_position_error_cm:.2f} cm")
    print(f"  Mean position error: {mean_position_error_cm:.2f} cm")
    
    assert max_position_error_cm < 1.0


@pytest.mark.skipif(
    not REFERENCE_DATA_PATH.exists(),
    reason="TRAVIS reference data not available. Run generate_travis_reference.py first."
)
def test_compare_arc_length_with_travis(w7x_travis_wout):
    """Compare arc length evolution between raytrax and TRAVIS.

    Uses the same equilibrium file as TRAVIS (set via TRAVIS_W7X_WOUT env var).
    """
    travis_data = load_reference_data(REFERENCE_DATA_PATH)

    equilibrium_interpolator = get_interpolator_for_equilibrium(w7x_travis_wout)
    plasma_profiles = create_matching_plasma_profiles()
    
    # Use same parameters as TRAVIS reference generation
    antenna_cyl = (6.509, -6.564, 0.38)  # (R_m, phi_deg, Z_m)
    target_w7x_angles = (15.7, 19.7)  # (theta_pol_deg, theta_tor_deg)
    
    antenna_cart = cylindrical_to_cartesian(*antenna_cyl)
    direction = w7x_aiming_angles_to_direction(target_w7x_angles[0], target_w7x_angles[1], antenna_cyl[1])
    
    beam = Beam(
        position=jnp.array(antenna_cart),
        direction=jnp.array(direction),
        frequency=jnp.array(140e9),
        mode="O",
    )
    
    result = trace(equilibrium_interpolator, plasma_profiles, beam)
    
    # Compare arc lengths
    raytrax_s = np.asarray(result.beam_profile.arc_length)
    travis_s = np.asarray(travis_data.arc_length_m)
    
    # Arc lengths should match at each point with very strict tolerance
    # Interpolate to common points count
    n_common = min(len(raytrax_s), len(travis_s))
    raytrax_s_interp = np.interp(np.linspace(0, 1, n_common), np.linspace(0, 1, len(raytrax_s)), raytrax_s)
    travis_s_interp = np.interp(np.linspace(0, 1, n_common), np.linspace(0, 1, len(travis_s)), travis_s)
    
    abs_error_cm = np.abs(raytrax_s_interp - travis_s_interp) * 100
    rel_error = np.abs((raytrax_s_interp - travis_s_interp) / travis_s_interp)
    
    max_abs_error_cm = np.max(abs_error_cm)
    max_rel_error_pct = np.max(rel_error) * 100
    
    print(f"\nArc length comparison:")
    print(f"  Max absolute error: {max_abs_error_cm:.2f} cm")
    print(f"  Max relative error: {max_rel_error_pct:.3f}%")
    print(f"  Final arc length - raytrax: {raytrax_s[-1]:.4f} m, TRAVIS: {travis_s[-1]:.4f} m")
    
    # Strict tolerance: < 0.1 cm or < 0.01% relative error
    assert max_abs_error_cm < 0.1, f"Arc length differs by {max_abs_error_cm:.2f} cm"
    assert max_rel_error_pct < 0.01, f"Arc length differs by {max_rel_error_pct:.3f}%"


@pytest.mark.skipif(
    not REFERENCE_DATA_PATH.exists(),
    reason="TRAVIS reference data not available. Run generate_travis_reference.py first."
)
def test_compare_optical_depth_with_travis(w7x_travis_wout):
    """Compare optical depth evolution between raytrax and TRAVIS.

    Uses the same equilibrium file as TRAVIS (set via TRAVIS_W7X_WOUT env var).
    """
    travis_data = load_reference_data(REFERENCE_DATA_PATH)

    equilibrium_interpolator = get_interpolator_for_equilibrium(w7x_travis_wout)
    plasma_profiles = create_matching_plasma_profiles()
    
    # Use same parameters as TRAVIS reference generation
    antenna_cyl = (6.509, -6.564, 0.38)  # (R_m, phi_deg, Z_m)
    target_w7x_angles = (15.7, 19.7)  # (theta_pol_deg, theta_tor_deg)
    
    antenna_cart = cylindrical_to_cartesian(*antenna_cyl)
    direction = w7x_aiming_angles_to_direction(target_w7x_angles[0], target_w7x_angles[1], antenna_cyl[1])
    
    beam = Beam(
        position=jnp.array(antenna_cart),
        direction=jnp.array(direction),
        frequency=jnp.array(140e9),
        mode="O",
    )
    
    result = trace(equilibrium_interpolator, plasma_profiles, beam)
    
    raytrax_s_m = result.beam_profile.arc_length
    raytrax_tau = result.beam_profile.optical_depth
    
    travis_s_m = travis_data.arc_length_m
    travis_tau = travis_data.optical_depth
    
    s_common, raytrax_tau_interp, travis_tau_interp = interpolate_to_common_grid(
        np.asarray(raytrax_s_m), np.asarray(raytrax_tau),
        np.asarray(travis_s_m), np.asarray(travis_tau),
        n_points=50
    )
    
    mask = travis_tau_interp > 0.1
    relative_error = np.abs(
        (raytrax_tau_interp[mask] - travis_tau_interp[mask]) / travis_tau_interp[mask]
    )
    
    if len(relative_error) > 0:
        max_rel_error = np.max(relative_error)
        print(f"\nOptical depth comparison:")
        print(f"  Max relative error: {max_rel_error*100:.1f}%")
        print(f"  Mean relative error: {np.mean(relative_error)*100:.1f}%")
        
        assert max_rel_error < 0.05


@pytest.mark.skipif(
    not REFERENCE_DATA_PATH.exists(),
    reason="TRAVIS reference data not available. Run generate_travis_reference.py first."
)
def test_compare_radial_profile_with_travis(w7x_travis_wout):
    """Compare radial deposition profiles between raytrax and TRAVIS.

    Uses the same equilibrium file as TRAVIS (set via TRAVIS_W7X_WOUT env var).
    """
    travis_data = load_reference_data(REFERENCE_DATA_PATH)

    equilibrium_interpolator = get_interpolator_for_equilibrium(w7x_travis_wout)
    plasma_profiles = create_matching_plasma_profiles()
    
    # Use same parameters as TRAVIS reference generation
    antenna_cyl = (6.509, -6.564, 0.38)  # (R_m, phi_deg, Z_m)
    target_w7x_angles = (15.7, 19.7)  # (theta_pol_deg, theta_tor_deg)
    
    antenna_cart = cylindrical_to_cartesian(*antenna_cyl)
    direction = w7x_aiming_angles_to_direction(target_w7x_angles[0], target_w7x_angles[1], antenna_cyl[1])
    
    beam = Beam(
        position=jnp.array(antenna_cart),
        direction=jnp.array(direction),
        frequency=jnp.array(140e9),
        mode="O",
    )
    
    result = trace(equilibrium_interpolator, plasma_profiles, beam)
    
    raytrax_rho = np.asarray(result.radial_profile.rho)
    raytrax_power = np.asarray(result.radial_profile.volumetric_power_density)
    
    travis_rho = np.asarray(travis_data.rho_profile)
    travis_power = np.asarray(travis_data.power_density_w_per_m3)
    
    rho_common = np.linspace(0, 1, 50)
    raytrax_power_interp = np.interp(rho_common, raytrax_rho, raytrax_power)
    travis_power_interp = np.interp(rho_common, travis_rho, travis_power)
    
    raytrax_peak_idx = np.argmax(raytrax_power_interp)
    travis_peak_idx = np.argmax(travis_power_interp)
    
    raytrax_peak_rho = rho_common[raytrax_peak_idx]
    travis_peak_rho = rho_common[travis_peak_idx]
    
    peak_rho_diff = abs(raytrax_peak_rho - travis_peak_rho)
    
    print(f"\nRadial profile comparison:")
    print(f"  Raytrax peak location: rho = {raytrax_peak_rho:.3f}")
    print(f"  TRAVIS peak location: rho = {travis_peak_rho:.3f}")
    print(f"  Peak location difference: {peak_rho_diff:.3f}")
    
    assert peak_rho_diff < 0.05
    
    raytrax_total = float(np.trapz(raytrax_power_interp, rho_common))
    travis_total = float(np.trapz(travis_power_interp, rho_common))
    
    power_rel_error = abs(raytrax_total - travis_total) / travis_total
    
    print(f"  Raytrax integrated power: {raytrax_total:.3e}")
    print(f"  TRAVIS integrated power: {travis_total:.3e}")
    print(f"  Relative error: {power_rel_error*100:.1f}%")
    
    assert power_rel_error < 0.10


def test_w7x_beam_runs_without_reference():
    """Basic test that W7-X beam tracing runs (doesn't require TRAVIS reference)."""
    from raytrax.data import get_w7x_wout
    
    wout = get_w7x_wout()
    equilibrium_interpolator = get_interpolator_for_equilibrium(wout)
    plasma_profiles = create_matching_plasma_profiles()
    
    antenna_cyl = (6.509, -6.564, 0.38)
    target_w7x_angles = (15.7, 19.7)  # theta_pol, theta_tor in degrees
    antenna_cart = cylindrical_to_cartesian(*antenna_cyl)
    antenna_phi_deg = np.rad2deg(antenna_cyl[1])  # phi in degrees
    direction = w7x_aiming_angles_to_direction(
        target_w7x_angles[0], target_w7x_angles[1], antenna_phi_deg
    )
    
    beam = Beam(
        position=jnp.array(antenna_cart),
        direction=jnp.array(direction),
        frequency=jnp.array(140e9),
        mode="O",
    )
    
    result = trace(equilibrium_interpolator, plasma_profiles, beam)
    
    assert len(result.beam_profile.position) > 0
    assert result.beam_profile.arc_length[-1] > 0
    assert len(result.radial_profile.rho) > 0
