#!/usr/bin/env python3
"""Generate interactive 3D HTML scenes for the documentation.

Produces two self-contained HTML files in docs/assets/:
  - 3d_scene_lcfs.html  : W7-X last closed flux surface
  - 3d_scene_beam.html  : LCFS (semi-transparent) + ECRH beam tube

Run from the repository root:
    python scripts/generate_3d_html.py
"""

from pathlib import Path

import jax.numpy as jnp
import pyvista as pv

from raytrax import Beam, RadialProfiles, trace
from raytrax.examples.w7x import (
    PortA,
    get_w7x_magnetic_configuration,
    w7x_aiming_angles_to_direction,
)
from raytrax.plot.plot3d import plot_beam_profile_3d, plot_flux_surface_3d

OUT_DIR = Path("docs/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pv.OFF_SCREEN = True

print("Loading magnetic configuration...", flush=True)
mag_conf = get_w7x_magnetic_configuration()

rho = jnp.linspace(0, 1, 200)
profiles = RadialProfiles(
    rho=rho,
    electron_density=0.5 * (1.0 - rho**2),
    electron_temperature=3.0 * (1.0 - rho**2),
)
beam = Beam(
    position=jnp.array(PortA.D1.cartesian),
    direction=jnp.array(
        w7x_aiming_angles_to_direction(
            theta_pol_deg=-10.0,
            theta_tor_deg=0.0,
            antenna_phi_deg=PortA.D1.phi_deg,
        )
    ),
    frequency=jnp.array(140e9),
    mode="O",
    power=1e6,
)

print("Tracing beam...", flush=True)
result = trace(mag_conf, profiles, beam)

# ── LCFS only ──────────────────────────────────────────────────────────────────

print("Generating LCFS scene...", flush=True)
plotter = pv.Plotter()
plotter.add_axes()
plot_flux_surface_3d(mag_conf, rho_value=1.0, plotter=plotter)
plotter.export_html(str(OUT_DIR / "3d_scene_lcfs.html"))
plotter.close()

# ── LCFS + beam ────────────────────────────────────────────────────────────────

print("Generating LCFS + beam scene...", flush=True)
plotter = pv.Plotter()
plotter.add_axes()
plot_flux_surface_3d(mag_conf, rho_value=1.0, plotter=plotter, opacity=0.25)
plot_beam_profile_3d(result.beam_profile, plotter=plotter, tube_radius=0.02)
plotter.export_html(str(OUT_DIR / "3d_scene_beam.html"))
plotter.close()

print(f"Written to {OUT_DIR.resolve()}/")
