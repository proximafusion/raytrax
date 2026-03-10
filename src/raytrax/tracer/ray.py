"""ODE state types: RaySetting (frequency, mode) and RayState (position, refractive index, optical depth)."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Literal

import jax
import jaxtyping as jt


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RaySetting:
    frequency: jt.Float[jax.Array, ""]
    mode: Literal["X", "O"] = dataclass_field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RayState:
    position: jt.Float[jax.Array, "3"]
    refractive_index: jt.Float[jax.Array, "3"]
    optical_depth: jt.Float[jax.Array, ""]
    arc_length: jt.Float[jax.Array, ""]
