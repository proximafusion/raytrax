"""This example demonstrates how to generate the interpolated minor radius
and magnetic field on a cylindrical grid, given a VMEC equilibrium, and how
to visualize it in ParaView.

The example requires the vmecpp and pyvista packages to be installed.

Run as:

```
python visualize_equilibrium.py
```

To open the generated ParaView state file:

- Open ParaView (requires version 6.0)
- Click File > load state
- Select "Python state file (*.py)" as file type
- Select paraview_state_w7x.py and click OK
"""

from pathlib import Path

import numpy as np

try:
    import pyvista as pv
except ImportError as exc:
    raise ImportError("You must install pyvista to run this example!") from exc

from raytrax.data import get_w7x_wout
from raytrax.interpolate import cylindrical_grid_for_equilibrium

if __name__ == "__main__":
    data_dir = Path(__file__).parent

    equilibrium = get_w7x_wout()

    # interpolate quantities on cylindrical grid
    cylindrical_grid = cylindrical_grid_for_equilibrium(
        equilibrium=equilibrium, n_rho=40, n_theta=45, n_phi=50, n_r=45, n_z=55
    )
    cylindrical_grid_np = np.array(cylindrical_grid)

    # cylindrical to cartesian
    r = cylindrical_grid_np[..., 0]
    phi = cylindrical_grid_np[..., 1]
    z = cylindrical_grid_np[..., 2]
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # create pyvista structured grid
    grid = pv.StructuredGrid(x, y, z)
    grid.point_data["B"] = cylindrical_grid_np[..., -3:].reshape(-1, 3, order="F")
    grid.point_data["absB"] = np.linalg.norm(
        cylindrical_grid_np[..., -3:], axis=-1
    ).reshape(-1, order="F")
    grid.point_data["rho"] = cylindrical_grid_np[..., 3].reshape(-1, order="F")

    # save VTS
    grid.save(data_dir / "w7x.vts")
