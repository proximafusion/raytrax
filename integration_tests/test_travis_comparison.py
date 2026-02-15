"""Integration test: compare raytrax against TRAVIS reference data.

The TRAVIS reference JSON is committed to the repo so this test runs in CI
without needing the TRAVIS binary installed.
"""

import numpy as np
import pytest
import jax.numpy as jnp
from pathlib import Path
from scipy.interpolate import interp1d

from raytrax.api import trace
from raytrax.interpolate import MagneticConfiguration
from raytrax.types import Beam, RadialProfiles
from raytrax.data import get_w7x_wout
from travis_wrapper import load_reference_data

TRAVIS_REF_FILE = Path(__file__).parent / "data" / "travis_w7x_reference.json"
B0_TARGET = 2.52076


def cylindrical_to_cartesian(r_m, phi_deg, z_m):
    phi_rad = np.deg2rad(phi_deg)
    return (float(r_m * np.cos(phi_rad)), float(r_m * np.sin(phi_rad)), float(z_m))


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
    """TRAVIS analytic profile formula (from plasma_profiles.f90)."""
    rho = np.asarray(rho)
    if abs(h) > 1e-6:
        a_adj = a - h
    else:
        a_adj = a
        h = 0
    if abs(h) > 1e-6 and abs(w) > 1e-3:
        z_over_w2 = np.clip(rho**2 / w**2, 0, 40)
        hole = h * (1 - np.exp(-z_over_w2))
    else:
        hole = 0
    if q < 1e-6 and abs(h) < 1e-6:
        y = np.ones_like(rho)
    else:
        y = a_adj + (1 - a_adj) * (1 - np.clip(rho, 0, 1) ** p) ** q + hole
    return central_value * y


def find_b0_on_axis(wout):
    from raytrax.interpolate import (
        build_magnetic_field_interpolator,
        build_rho_interpolator,
    )

    eq_unscaled = MagneticConfiguration.from_vmec_wout(wout)
    B_interp = build_magnetic_field_interpolator(eq_unscaled)
    rho_interp = build_rho_interpolator(eq_unscaled)
    best_rho, best_B = 999.0, 0.0
    for R in np.arange(5.0, 6.5, 0.002):
        rho_val = float(rho_interp(R, 0.0, 0.0))
        if rho_val < best_rho:
            best_rho = rho_val
            B_grid = B_interp(R, 0.0, 0.0)
            best_B = float(jnp.linalg.norm(B_grid))
    return best_B


@pytest.fixture(scope="module")
def travis_reference():
    return load_reference_data(TRAVIS_REF_FILE)


@pytest.fixture(scope="module")
def raytrax_result():
    wout = get_w7x_wout()
    b0_native = find_b0_on_axis(wout)
    b_scale = B0_TARGET / b0_native
    eq_interp = MagneticConfiguration.from_vmec_wout(wout, magnetic_field_scale=b_scale)

    n_rho = 501
    rho = np.linspace(0, 1, n_rho)
    ne_1e20 = travis_profile(rho, 0.75, 0.05, 20, 2.8, 0, 0.32)
    te_keV = travis_profile(rho, 5.0, 0.03, 4, 2.0, -0.2, 0.5)

    profiles = RadialProfiles(
        rho=jnp.array(rho),
        electron_density=jnp.array(ne_1e20),
        electron_temperature=jnp.array(te_keV),
    )

    antenna_cyl = (6.50866, -6.56378, -0.38)
    target_angles = (15.7, 19.7001)
    antenna_cart = cylindrical_to_cartesian(*antenna_cyl)
    direction = w7x_aiming_angles_to_direction(
        target_angles[0], target_angles[1], antenna_cyl[1]
    )

    beam = Beam(
        position=jnp.array(antenna_cart),
        direction=jnp.array(direction),
        frequency=jnp.array(140e9),
        mode="O",
    )

    return trace(eq_interp, profiles, beam)


@pytest.fixture(scope="module")
def comparison(raytrax_result, travis_reference):
    """Interpolate TRAVIS data onto raytrax arc lengths and compute errors."""
    tr = travis_reference
    rx = raytrax_result.beam_profile

    s_rx = np.asarray(rx.arc_length)
    s_tr = np.asarray(tr.arc_length_m)

    # Overlapping region
    s_min = max(s_rx[0], s_tr[0])
    s_max = min(s_rx[-1], s_tr[-1])
    mask = (s_rx >= s_min) & (s_rx <= s_max)
    s_overlap = s_rx[mask]

    # Interpolate TRAVIS onto raytrax arc lengths
    def _interp(travis_data, **kw):
        return interp1d(
            s_tr,
            travis_data,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            **kw,
        )(s_overlap)

    pos_tr = _interp(np.asarray(tr.position_m), axis=0)
    B_tr = _interp(np.asarray(tr.magnetic_field_magnitude_T))
    rho_tr = _interp(np.asarray(tr.rho))

    # ne/Te from profile formula evaluated at TRAVIS rho
    ne_parm = (0.05, 20, 2.8, 0, 0.32)
    te_parm = (0.03, 4, 2.0, -0.2, 0.5)
    ne_tr = travis_profile(rho_tr, 0.75, *ne_parm)
    te_tr = travis_profile(rho_tr, 5.0, *te_parm)

    # raytrax values in overlap
    pos_rx = np.asarray(rx.position)[mask]
    B_rx = np.linalg.norm(np.asarray(rx.magnetic_field)[mask], axis=1)
    rho_rx = np.asarray(rx.normalized_effective_radius)[mask]
    ne_rx = np.asarray(rx.electron_density)[mask]
    te_rx = np.asarray(rx.electron_temperature)[mask]
    tau_rx = np.asarray(rx.optical_depth)[-1]
    tau_tr = float(np.asarray(tr.optical_depth)[-1])

    # Compute metrics
    pos_rms_mm = float(np.sqrt(np.mean(np.sum((pos_rx - pos_tr) ** 2, axis=1)))) * 1000
    B_mean_pct = float(np.mean(np.abs((B_rx - B_tr) / B_tr))) * 100
    ne_mean_pct = float(np.mean(np.abs((ne_rx - ne_tr) / ne_tr))) * 100
    te_mean_pct = float(np.mean(np.abs((te_rx - te_tr) / te_tr))) * 100
    rho_rms = float(np.sqrt(np.mean((rho_rx - rho_tr) ** 2)))
    tau_pct = float(abs(tau_rx - tau_tr) / tau_tr) * 100

    return {
        "pos_rms_mm": pos_rms_mm,
        "B_mean_pct": B_mean_pct,
        "ne_mean_pct": ne_mean_pct,
        "te_mean_pct": te_mean_pct,
        "rho_rms": rho_rms,
        "tau_pct": tau_pct,
        "tau_rx": float(tau_rx),
        "tau_tr": tau_tr,
    }


def _print_summary(comparison):
    print("\n--- raytrax vs TRAVIS comparison ---")
    print(f"  Position:      RMS = {comparison['pos_rms_mm']:.2f} mm")
    print(f"  |B|:           mean error = {comparison['B_mean_pct']:.2f}%")
    print(f"  ne:            mean error = {comparison['ne_mean_pct']:.2f}%")
    print(f"  Te:            mean error = {comparison['te_mean_pct']:.2f}%")
    print(f"  rho:           RMS = {comparison['rho_rms']:.4f}")
    print(
        f"  Optical depth: raytrax={comparison['tau_rx']:.4f}  "
        f"TRAVIS={comparison['tau_tr']:.4f}  "
        f"error={comparison['tau_pct']:.1f}%"
    )


@pytest.mark.integration
class TestTravisComparison:
    """Compare raytrax ray tracing against TRAVIS reference output."""

    def test_position(self, comparison):
        _print_summary(comparison)
        assert (
            comparison["pos_rms_mm"] < 5.0
        ), f"Position RMS {comparison['pos_rms_mm']:.2f} mm exceeds 5 mm"

    def test_magnetic_field(self, comparison):
        assert (
            comparison["B_mean_pct"] < 2.0
        ), f"|B| mean error {comparison['B_mean_pct']:.2f}% exceeds 2%"

    def test_electron_density(self, comparison):
        assert (
            comparison["ne_mean_pct"] < 3.0
        ), f"ne mean error {comparison['ne_mean_pct']:.2f}% exceeds 3%"

    def test_electron_temperature(self, comparison):
        assert (
            comparison["te_mean_pct"] < 3.0
        ), f"Te mean error {comparison['te_mean_pct']:.2f}% exceeds 3%"

    def test_rho(self, comparison):
        assert (
            comparison["rho_rms"] < 0.02
        ), f"rho RMS {comparison['rho_rms']:.4f} exceeds 0.02"

    def test_optical_depth_report(self, comparison):
        """Report optical depth deviation. Known ~25% off (absorption model WIP)."""
        print(
            f"\n  [INFO] Optical depth error: {comparison['tau_pct']:.1f}% "
            "(no assertion — absorption model is work in progress)"
        )
