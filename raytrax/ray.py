from dataclasses import dataclass
from typing import Literal

import jax
import jaxtyping as jt


@dataclass(frozen=True)
class RaySetting:
    frequency: jt.Float[jax.Array, ""]
    mode: Literal["X", "O"]


@dataclass(frozen=True)
class RayState:
    position: jt.Float[jax.Array, "3"]
    refractive_index: jt.Float[jax.Array, "3"]
    optical_depth: jt.Float[jax.Array, ""]
    arc_length: jt.Float[jax.Array, ""]


@dataclass(frozen=True)
class RayQuantities:
    magnetic_field: jt.Float[jax.Array, "3"]
    absorption_coefficient: jt.Float[jax.Array, ""]
    electron_density: jt.Float[jax.Array, ""]
    electron_temperature: jt.Float[jax.Array, ""]
    linear_power_density: jt.Float[jax.Array, ""]
    normalized_effective_radius: jt.Float[jax.Array, ""]


jax.tree_util.register_pytree_node(
    RayState,
    lambda rs: (
        (
            rs.position,
            rs.refractive_index,
            rs.optical_depth,
            rs.arc_length,
        ),
        None,
    ),
    lambda _, children: RayState(
        position=children[0],
        refractive_index=children[1],
        optical_depth=children[2],
        arc_length=children[3],
    ),
)

jax.tree_util.register_pytree_node(
    RaySetting,
    lambda rs: ((rs.frequency,), (rs.mode,)),
    lambda aux_data, children: RaySetting(
        frequency=children[0],
        mode=aux_data[0],  # type: ignore
    ),
)
