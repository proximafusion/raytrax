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

The import involves two grids: an intermediate curvilinear grid in VMEC flux coordinates $(\rho, \theta, \phi)$ on which the Fourier series are evaluated, and the final cylindrical grid $(R, \phi, Z)$ on which the ray tracer operates. Both are controlled via [`VmecGridResolution`][raytrax.equilibrium.interpolate.VmecGridResolution]:

```python
grid = raytrax.VmecGridResolution(
    cylindrical=raytrax.CylindricalGridResolution(
        n_r=60,    # R points on the output cylindrical grid
        n_z=70,    # Z points on the output cylindrical grid
        n_phi=64,  # toroidal planes on the output cylindrical grid
        n_rho_profile=200,  # points for the 1-D dV/dρ profile
    ),
    n_rho=50,    # radial flux surfaces on the intermediate VMEC grid
    n_theta=60,  # poloidal points on the intermediate VMEC grid
)
mag_conf = raytrax.MagneticConfiguration.from_vmec_wout(vmec_wout, grid=grid)
```

The defaults (`n_r=45`, `n_z=55`, `n_phi=50`, `n_rho=40`, `n_theta=45`) are adequate for most stellarator geometries. Increasing them improves accuracy at the cost of memory and import time, which scales roughly as $n_r \cdot n_\phi \cdot n_z$.