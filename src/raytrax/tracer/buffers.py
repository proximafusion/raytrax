"""JAX pytree types for the ODE solver: Interpolators (field/profile inputs) and TraceBuffers (padded output arrays)."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field

import interpax
import jax
import jaxtyping as jt


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Interpolators:
    """Bundle of interpolation functions for ray tracing.

    Groups the four interpolators needed by the ODE solver into a single
    pytree-compatible object passed through JIT as one argument.
    """

    magnetic_field: interpax.Interpolator3D | interpax.Interpolator2D
    """Interpolator for B field. 3D (R, phi, Z) for stellarators, 2D (R, Z) for tokamaks."""

    rho: interpax.Interpolator3D | interpax.Interpolator2D
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

    All arrays are padded to max_steps (4096). Invalid entries beyond the
    last integration step are inf/NaN and must be trimmed by the caller.
    """

    arc_length: jt.Float[jax.Array, " nsteps"]
    ode_state: jt.Float[jax.Array, "nsteps 7"]
    magnetic_field: jt.Float[jax.Array, "nsteps 3"]
    normalized_effective_radius: jt.Float[jax.Array, " nsteps"]
    electron_density: jt.Float[jax.Array, " nsteps"]
    electron_temperature: jt.Float[jax.Array, " nsteps"]
    absorption_coefficient: jt.Float[jax.Array, " nsteps"]
    linear_power_density: jt.Float[jax.Array, " nsteps"]
    volumetric_power_density: jt.Float[jax.Array, " nsteps"]
