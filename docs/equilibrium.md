---
icon: lucide/magnet
---

# Loading a MHD Equilibrium

One of the inputs to Raytrax is a [`MagneticConfiguration`][raytrax.equilibrium.interpolate.MagneticConfiguration], which is a grid in cylindrical coordinates holding the values of the magnetic field $\vec{B}$, the effective minor radius $\rho$, and some other geometric quantities. Raytrax can compute this configuration from a magnetohydrodynamic (MHD) equilibrium.

At present, Raytrax supports loading equilibria from two codes:

- [VMEC++](https://proximafusion.github.io/vmecpp/), a modern implementation of the widely used VMEC code for computing stellarator equilibria, and
- [GVEC](https://gvec.readthedocs.io), which is not restricted to cylindrical cross-sections and can describe strongly shaped stellarators in a frame adapted to their geometry.

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
        n_r=60,  # R points on the output cylindrical grid
        n_z=70,  # Z points on the output cylindrical grid
        n_phi=64,  # toroidal planes on the output cylindrical grid
        n_rho_profile=200,  # points for the 1-D dV/dρ profile
    ),
    n_rho=50,  # radial flux surfaces on the intermediate VMEC grid
    n_theta=60,  # poloidal points on the intermediate VMEC grid
)
mag_conf = raytrax.MagneticConfiguration.from_vmec_wout(vmec_wout, grid=grid)
```

The defaults (`n_r=45`, `n_z=55`, `n_phi=50`, `n_rho=40`, `n_theta=45`) are adequate for most stellarator geometries. Increasing them improves accuracy at the cost of memory and import time, which scales roughly as $n_r \cdot n_\phi \cdot n_z$.

### Using GVEC

GVEC support is an optional extra. GVEC has no prebuilt wheels, so installing it compiles GVEC from source, which needs a Fortran compiler and the netCDF-Fortran library:

```bash
python -m pip install "raytrax[gvec]"
```

To instantiate a magnetic configuration from a GVEC equilibrium, you need a `gvec.State` object. You can load an existing GVEC run from its output directory:

```python
import gvec

state = gvec.find_state("path/to/gvec/run")
```

or compute a new equilibrium with `gvec.run` (see the [GVEC documentation](https://gvec.readthedocs.io)). Once you have the state, you can create a [`MagneticConfiguration`][raytrax.equilibrium.interpolate.MagneticConfiguration] from it:

```python
mag_conf = raytrax.MagneticConfiguration.from_gvec(state)
```

The result is the same cylindrical-grid representation as for VMEC++, so tracing works identically.

#### Supported equilibria

GVEC describes the geometry through a map from its coordinates $(X^1, X^2, \zeta)$ to space that is chosen per equilibrium (the "hmap"). In general, a cross-section $\zeta = \text{const}$ is then a plane that is tilted against the planes $\phi = \text{const}$ of the cylindrical grid. Raytrax handles this for any hmap by finding where each GVEC coordinate line crosses each grid plane. The import requires that the equilibrium

- is periodic under a rotation by $2\pi/n_\text{fp}$ about the $Z$-axis,
- is stellarator symmetric, and
- is cut by every half-plane $\phi = \text{const}$ in a single closed curve per flux surface.

These conditions are checked during the import, which raises `NotImplementedError` if one is violated. The last condition excludes figure-8 and knotted geometries.

#### Tuning grid resolution

As for VMEC++, the import samples the equilibrium on an intermediate grid in GVEC coordinates $(\rho, \theta, \zeta)$ before interpolating it onto the cylindrical grid. Both are controlled via [`GvecGridResolution`][raytrax.equilibrium.gvec.GvecGridResolution]:

```python
grid = raytrax.GvecGridResolution(
    cylindrical=raytrax.CylindricalGridResolution(n_r=95, n_z=105, n_phi=50),
    n_rho=40,  # radial flux surfaces on the intermediate GVEC grid
    n_theta=45,  # poloidal points on the intermediate GVEC grid
    n_zeta=120,  # toroidal points per field period on the intermediate GVEC grid
)
mag_conf = raytrax.MagneticConfiguration.from_gvec(state, grid=grid)
```

`n_zeta` sets how finely each coordinate line is sampled to locate its crossings with the grid planes; the defaults locate the flux surfaces of W7-X to better than a millimetre.
