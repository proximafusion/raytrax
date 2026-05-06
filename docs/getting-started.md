---
icon: lucide/rocket
---

# Getting Started

## Installation

To install Raytrax, simply use `pip`:

```bash
python -m pip install raytrax
```

We recommend using a virtual environment to manage your dependencies.

## Prerequisites: Enable 64-bit Precision

Raytrax requires 64-bit floating-point arithmetic. JAX defaults to 32-bit, so you must opt in **before importing JAX or raytrax**:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

If you forget this, raytrax will emit a warning at import time.

## Your First Trace

Before starting to simulate fusion heating, it's important to understand some of the technical characteristics of Raytrax. Since it's based on [JAX](https://docs.jax.dev), it profits from just-in-time compilation and automatic differentiability. This comes with a cost: the first invocation of [`trace`][raytrax.api.trace] will be fairly slow since the computation needs to be compiled first. That's why it wouldn't make much sense to use Raytrax as a command line script. Instead, it's meant to be used inside a Python program — for example in an optimization loop — where the compiled function is reused across many calls.

### Prepare Inputs

Raytrax requires three inputs:

1. A magnetic configuration
2. Radial plasma profiles
3. Beam settings.

The **[`MagneticConfiguration`][raytrax.equilibrium.interpolate.MagneticConfiguration]** is a grid in cylindrical coordinates holding the values of the magnetic field $\vec{B}$, the effective minor radius $\rho$, and some other geometric quantities. Raytrax can compute this configuration from a [VMEC++](https://proximafusion.github.io/vmecpp/) MHD equilibrium.

An example where an equilibrium is loaded from a NetCDF file:

```python
import jax
jax.config.update("jax_enable_x64", True)

import raytrax, vmecpp

vmec_wout = vmecpp.VmecWOut.from_wout_file("w7x.nc")
mag_conf = raytrax.MagneticConfiguration.from_vmec_wout(vmec_wout)
```

You can save the configuration to a file and load it back with the object's `.save` and `.load` methods.

The **[`RadialProfiles`][raytrax.types.RadialProfiles]** are gridded one-dimensional profiles for the electron density $n_e$ (in units of 10<sup>20</sup>/m³) and temperature $T_e$ (in units of keV) as a function of the effective minor radius $\rho$, which must extend from 0 to 1. Example:

```python
import raytrax, jax.numpy as jnp

rho = jnp.linspace(0, 1, 40)
n_e = 1.0 * (1 - rho**2)
T_e = 2.0 * (1 - rho**1.5)
profiles = raytrax.RadialProfiles(
    rho=rho,
    electron_density=n_e,
    electron_temperature=T_e
)
```

!!! tip "Profiles with non-zero density at the boundary"
    If $n_e(\rho{=}1) > 0$, call [`with_zero_density_at_boundary`][raytrax.types.RadialProfiles.with_zero_density_at_boundary] on the profiles object before tracing.  This smoothly tapers the density to zero over the outermost 10% of the minor radius and avoids a spurious discontinuity at the plasma-vacuum interface.

    ```python
    profiles_tapered = profiles.with_zero_density_at_boundary(boundary_layer_width=0.1)
    result = raytrax.trace(mag_conf, profiles_tapered, beam)
    ```

The **[`Beam`][raytrax.types.Beam]** defines the properties of the microwave beam to be traced: its starting position (a vector in Cartesian coordinates), initial direction (a unit 3-vector), frequency (in Hz, not GHz!), wave mode (ordinary or extraordinary mode), and initial power (in W). The optional `max_harmonic` parameter (default: `2`) sets the highest cyclotron harmonic included in the absorption calculation — increase it to `3` for third-harmonic scenarios. Example:

```python
import raytrax, jax.numpy as jnp

beam = raytrax.Beam(
    position=jnp.array([1.0, 2.0, 3.0]),
    direction=jnp.array([0.0, -1.0, 0.0]),
    frequency=140e9,  # Hz!
    mode="O",
    power=1e6,  # W
)
```

### Trace

Once the inputs are ready, you can run the ray tracer:

```python
import raytrax

result = raytrax.trace(
    magnetic_configuration=mag_conf,
    radial_profiles=profiles,
    beam=beam,
)
```

The returned [`TraceResult`][raytrax.types.TraceResult] contains:

- `beam_profile` — a [`BeamProfile`][raytrax.types.BeamProfile] with quantities along the beam trajectory in Cartesian space: position, arc length, refractive index, optical depth, absorption coefficient, electron density, electron temperature, magnetic field, effective minor radius $\rho$, and linear power density.
- `radial_profile` — a [`RadialProfile`][raytrax.types.RadialProfile] with the volumetric power deposition density as a function of $\rho$.
- `absorbed_power` — total power absorbed by the plasma in W.
- `absorbed_power_fraction` — fraction of the input beam power absorbed, i.e. $1 - e^{-\tau}$.
- `optical_depth` — total optical depth $\tau$ accumulated along the ray.
- `deposition_rho_mean` / `deposition_rho_std` — volume-weighted mean and standard deviation of the deposition location in $\rho$.

## Jupyter Notebooks

The [`notebooks/`](https://github.com/proximafusion/raytrax/tree/main/notebooks) directory contains ready-to-run notebooks ideal for exploring Raytrax interactively. You can launch them directly in your browser — no local installation required:

| Notebook | Description | |
|----------|-------------|---|
| [`01_first_trace.ipynb`](https://github.com/proximafusion/raytrax/blob/main/notebooks/01_first_trace.ipynb) | Build a synthetic tokamak, trace a beam, and visualise the R–Z trajectory — no external files needed. | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/proximafusion/raytrax/main?labpath=notebooks%2F01_first_trace.ipynb) |
| [`02_w7x_trace_and_visualization.ipynb`](https://github.com/proximafusion/raytrax/blob/main/notebooks/02_w7x_trace_and_visualization.ipynb) | Load the bundled W7-X equilibrium, set a realistic antenna position, and produce polished cross-section and profile plots. | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/proximafusion/raytrax/main?labpath=notebooks%2F02_w7x_trace_and_visualization.ipynb) |
| [`03_gradient_optimization.ipynb`](https://github.com/proximafusion/raytrax/blob/main/notebooks/03_gradient_optimization.ipynb) | Differentiate through the ray tracer with `jax.grad` and steer a beam via gradient ascent. | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/proximafusion/raytrax/main?labpath=notebooks%2F03_gradient_optimization.ipynb) |
