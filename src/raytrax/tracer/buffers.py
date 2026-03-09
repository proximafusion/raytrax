"""JAX pytree types for the ODE solver: Interpolators (field/profile inputs) and TraceBuffers (padded output arrays)."""

from dataclasses import dataclass

import interpax
import jax
import jaxtyping as jt


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

    is_axisymmetric: bool = False
    """Whether the equilibrium is axisymmetric (tokamak). Stored in pytree aux_data."""


jax.tree_util.register_pytree_node(
    Interpolators,
    lambda i: (
        (i.magnetic_field, i.rho, i.electron_density, i.electron_temperature),
        (i.is_axisymmetric,),
    ),
    lambda aux, children: Interpolators(
        magnetic_field=children[0],
        rho=children[1],
        electron_density=children[2],
        electron_temperature=children[3],
        is_axisymmetric=aux[0],
    ),
)


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


jax.tree_util.register_pytree_node(
    TraceBuffers,
    lambda r: (
        (
            r.arc_length,
            r.ode_state,
            r.magnetic_field,
            r.normalized_effective_radius,
            r.electron_density,
            r.electron_temperature,
            r.absorption_coefficient,
            r.linear_power_density,
            r.volumetric_power_density,
        ),
        None,
    ),
    lambda _, children: TraceBuffers(
        arc_length=children[0],
        ode_state=children[1],
        magnetic_field=children[2],
        normalized_effective_radius=children[3],
        electron_density=children[4],
        electron_temperature=children[5],
        absorption_coefficient=children[6],
        linear_power_density=children[7],
        volumetric_power_density=children[8],
    ),
)
