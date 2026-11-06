"""JAX pytree types for the ODE solver: Interpolators (field/profile inputs) and TraceBuffers (padded output arrays)."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field

import interpax
import jax
import jaxtyping as jt

from raytrax.equilibrium.interpolate import RegularGridInterpolator


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Interpolators:
    """Bundle of interpolation functions for ray tracing.

    Groups the four interpolators needed by the ODE solver into a single
    pytree-compatible object passed through JIT as one argument.
    """

    magnetic_field: RegularGridInterpolator
    """Interpolator for B field. 3D (R, phi, Z) for stellarators, 2D (R, Z) for tokamaks."""

    rho: RegularGridInterpolator
    """Interpolator for rho. 3D (R, phi, Z) for stellarators, 2D (R, Z) for tokamaks."""

    electron_density: interpax.Interpolator1D
    """Interpolator for ne(rho)."""

    electron_temperature: interpax.Interpolator1D
    """Interpolator for Te(rho)."""

    is_axisymmetric: bool = dataclass_field(default=False, metadata={"static": True})
    """Whether the equilibrium is axisymmetric (tokamak). Stored in pytree aux_data."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TraceBuffers:
    """Raw output from the JIT-compiled trace, before trimming padded buffers.

    All arrays are padded to `TracerSettings.max_steps + 1` entries. Invalid
    entries beyond the last integration step are inf/NaN and must be trimmed by
    the caller.
    """

    arc_length: jt.Float[jax.Array, " nsteps"]
    ode_state: jt.Float[jax.Array, "nsteps 8"]
    """(position, refractive index, optical depth, arc length) at each step."""
    magnetic_field: jt.Float[jax.Array, "nsteps 3"]
    normalized_effective_radius: jt.Float[jax.Array, " nsteps"]
    electron_density: jt.Float[jax.Array, " nsteps"]
    electron_temperature: jt.Float[jax.Array, " nsteps"]
    absorption_coefficient: jt.Float[jax.Array, " nsteps"]
    linear_power_density: jt.Float[jax.Array, " nsteps"]
    incomplete: jt.Bool[jax.Array, ""]
    """Whether the solve ended without a termination event (ray left the plasma,
    was absorbed or left the domain)."""
