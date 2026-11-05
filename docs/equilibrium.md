---
icon: lucide/magnet
---

# Loading a MHD Equilibrium

One of the inputs to Raytrax is a [`MagneticConfiguration`][raytrax.equilibrium.interpolate.MagneticConfiguration], which is a grid in cylindrical coordinates holding the values of the magnetic field $\vec{B}$, the effective minor radius $\rho$, and some other geometric quantities. Raytrax can compute this configuration from a magnetohydrodynamic (MHD) equilibrium.

At present, Raytrax supports loading equilibria from the [VMEC++](https://proximafusion.github.io/vmecpp/) code, which is a modern implementation of the widely used VMEC code for computing stellarator equilibria.

!!! info
    Support for Tokamak equilibria is planned for a future release.

### Using VMEC++

To instantiate a magnetic configuration from a VMEC++ equilibrium, you need a `vmecpp.VmecWOut` object, which can be either obtained by running VMEC++ on an input file, or by loading an existing equilibrium file in NetCDF "wout" format (created with VMEC++ or the original VMEC).

Example for loading an existing wout file:

```python
import vmecpp

vmec_wout = vmecpp.VmecWOut.from_wout_file("w7x.nc")
```

Alternatively, you can run VMEC++ on an input file yourself:

```python
import vmecpp

vmec_input = vmecpp.VmecInput.from_file("input.w7x")
vmec_output = vmecpp.run(vmec_input)
vmec_wout = vmec_output.wout
```

For more options and details, see the [VMEC++ documentation](https://proximafusion.github.io/vmecpp/).


Once you have the `VmecWOut` object, you can create a [`MagneticConfiguration`][raytrax.equilibrium.interpolate.MagneticConfiguration] from it:

```python
mag_conf = raytrax.MagneticConfiguration.from_vmec_wout(vmec_wout)
```

#### Tuning grid resolution

The ray tracer operates on a cylindrical grid $(R, \phi, Z)$. At each grid node, the import solves for the VMEC flux coordinates $(\rho, \theta)$ of the node in its $\phi$ plane with Newton's method and evaluates $\rho$ and $\boldsymbol{B}$ from the Fourier series there, so the values at the nodes are exact. A coarser sampling of the flux surfaces up to $\rho = 1.2$ fixes the $R$ and $Z$ extent of the grid. Both are controlled via [`VmecGridResolution`][raytrax.equilibrium.interpolate.VmecGridResolution]:

```python
grid = raytrax.VmecGridResolution(
    cylindrical=raytrax.CylindricalGridResolution(
        n_r=60,  # R points on the output cylindrical grid
        n_z=70,  # Z points on the output cylindrical grid
        n_phi=64,  # toroidal planes on the output cylindrical grid
        n_rho_profile=200,  # points for the 1-D dV/dρ profile
    ),
    n_rho=50,  # flux surfaces sampled for the grid extent
    n_theta=60,  # poloidal points sampled for the grid extent
)
mag_conf = raytrax.MagneticConfiguration.from_vmec_wout(vmec_wout, grid=grid)
```

The defaults (`n_r=95`, `n_z=105`, `n_phi=50`, `n_rho=40`, `n_theta=45`) are adequate for most stellarator geometries. The accuracy of the ray tracer is set by the cylindrical grid. Between its nodes, $\boldsymbol{B}$ and $\rho^2$ are interpolated with Catmull-Rom splines, which have continuous first derivatives, so gradients through the ray tracer are correct. Refining it costs memory and import time, which scale roughly as $n_r \cdot n_\phi \cdot n_z$. `n_rho` and `n_theta` only set the extent of the grid and have little effect on accuracy.