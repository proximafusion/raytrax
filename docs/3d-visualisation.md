---
icon: lucide/box
---

# 3D Visualisation

The `raytrax.plot.plot3d` module uses [PyVista](https://pyvista.org) to render
flux surfaces and beam trajectories in three dimensions.

## Usage

```python
import pyvista as pv
import jax.numpy as jnp

from raytrax import Beam, RadialProfiles, trace
from raytrax.examples.w7x import (
    PortA,
    get_w7x_magnetic_configuration,
    w7x_aiming_angles_to_direction,
)
from raytrax.plot.plot3d import plot_flux_surface_3d, plot_beam_profile_3d

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
        w7x_aiming_angles_to_direction(-10.0, 0.0, PortA.D1.phi_deg)
    ),
    frequency=jnp.array(140e9),
    mode="O",
    power=1e6,
)
result = trace(mag_conf, profiles, beam)

plotter = pv.Plotter()
plot_flux_surface_3d(mag_conf, rho_value=1.0, plotter=plotter, opacity=0.25)
plot_beam_profile_3d(result.beam_profile, plotter=plotter, tube_radius=0.02)

plotter.export_html("scene.html")  # save interactive HTML
```

!!! tip "Jupyter"
    Replace `plotter.export_html(...)` with `plotter.show()` to get an
    interactive inline widget directly in the notebook.

## Last closed flux surface

The LCFS ($\rho = 1$) of the W7-X equilibrium, showing the characteristic
bean-shaped and triangular cross-sections around the torus.

<iframe src="../assets/3d_scene_lcfs.html" width="100%" height="500" style="border:none; border-radius:4px;"></iframe>

[Open in new tab](../assets/3d_scene_lcfs.html){target="_blank"}

## LCFS + ECRH beam

The LCFS rendered semi-transparent with the ECRH beam tube overlaid,
coloured by linear power density along the ray path.

<iframe src="../assets/3d_scene_beam.html" width="100%" height="500" style="border:none; border-radius:4px;"></iframe>

[Open in new tab](../assets/3d_scene_beam.html){target="_blank"}
