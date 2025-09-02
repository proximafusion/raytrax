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

import urllib.request
from pathlib import Path

import numpy as np

try:
    import pyvista as pv
except ImportError as exc:
    raise ImportError("You must install pyvista to run this example!") from exc

try:
    import vmecpp
except ImportError as exc:
    raise ImportError("You must install vmecpp to run this example!") from exc

from raytrax.interpolate import cylindrical_grid_for_equilibrium

W7X_JSON = "https://github.com/proximafusion/vmecpp/raw/main/examples/data/w7x.json"


def download_file(url: str, dest_path: Path):
    if not dest_path.exists():
        print(f"Downloading {url} to {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    else:
        print(f"File already exists at {dest_path}.")


def equilibrium_from_vmec_input(input_path: Path) -> vmecpp.VmecWOut:
    """Run VMEC++ from a given input file and return the equilibrium."""
    vmec_input = vmecpp.VmecInput.from_file(input_path)
    print("Running VMEC++...")
    vmec_output = vmecpp.run(vmec_input)
    print("VMEC++ run complete.")
    return vmec_output.wout


if __name__ == "__main__":
    # create data directory
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)

    # download VMEC++ JSON file for W7-X
    download_file(W7X_JSON, data_dir / "w7x.json")

    # run VMEC++
    equilibrium = equilibrium_from_vmec_input(data_dir / "w7x.json")

    # interpolate quantities on cylindrical grid
    cylindrical_grid = cylindrical_grid_for_equilibrium(
        equilibrium=equilibrium, n_rho=40, n_theta=45, n_phi=50, n_r=45, n_z=55
    )

    # cylindrical to cartesian
    r = cylindrical_grid[..., 0]
    phi = cylindrical_grid[..., 1]
    z = cylindrical_grid[..., 2]
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # create pyvista structured grid
    grid = pv.StructuredGrid(np.array(x), np.array(y), np.array(z))
    grid.point_data["B"] = cylindrical_grid[..., -3:].reshape(-1, 3, order="F")
    grid.point_data["absB"] = np.linalg.norm(
        cylindrical_grid[..., -3:], axis=-1
    ).reshape(-1, order="F")
    grid.point_data["rho"] = cylindrical_grid[..., 3].reshape(-1, order="F")

    # save VTS
    grid.save(data_dir / "w7x.vts")
