"""Public data types for the raytrax API: beam inputs, trace outputs, and shared utilities."""

from dataclasses import dataclass, fields
from dataclasses import field as dataclass_field
from typing import Any, Literal, TypeVar

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

ScalarFloat = float | jt.Float[jax.Array, " "]
T = TypeVar("T", bound="SafetensorsMixin")


class SafetensorsMixin:
    """Mixin for dataclasses to add safetensors save/load functionality."""

    def save(self, path: str) -> None:
        """Save dataclass to a safetensors file.

        Args:
            path: Path to save to (should end in .safetensors)
        """

        # Separate arrays from scalars
        tensors = {}
        metadata = {}

        for field in fields(self):  # type: ignore[arg-type]
            value = getattr(self, field.name)
            if isinstance(value, jax.Array):
                # Convert JAX arrays to numpy (zero-copy on CPU)
                tensors[field.name] = np.asarray(value)
            else:
                # Store scalars as metadata strings
                metadata[field.name] = str(value)

        save_file(tensors, path, metadata=metadata)

    @classmethod
    def load(cls: type[T], path: str) -> T:
        """Load dataclass from a safetensors file.

        Args:
            path: Path to the safetensors file to load

        Returns:
            Loaded instance of the dataclass
        """

        # Load tensors and metadata
        tensors = load_file(path)

        with safe_open(path, framework="numpy") as f:
            metadata = f.metadata() or {}

        # Reconstruct all fields
        field_values: dict[str, Any] = {}
        for field in fields(cls):  # type: ignore[arg-type]
            if field.name in tensors:
                # It's an array - convert back to JAX
                field_values[field.name] = jnp.array(tensors[field.name])
            elif field.name in metadata:
                # It's a scalar - parse back from string
                value_str = metadata[field.name]
                if field.type is int:
                    field_values[field.name] = int(value_str)
                elif field.type is bool:
                    field_values[field.name] = value_str == "True"
                else:
                    field_values[field.name] = value_str

        return cls(**field_values)


@jax.tree_util.register_dataclass
@dataclass
class Beam:
    """Beam parameter inputs for tracing."""

    position: jt.Float[jax.Array, "3"]
    """The starting position of the beam in cartesian coordinates."""

    direction: jt.Float[jax.Array, "3"]
    """The starting direction of the beam in cartesian coordinates. Must be a unit vector."""

    frequency: jt.Float[jax.Array, ""]
    """The frequency of the beam in Hz (not GHz!)."""

    mode: Literal["X", "O"] = dataclass_field(metadata={"static": True})
    """The polarization mode of the beam, either `"X"` for extraordinary or `"O"` for ordinary mode."""

    power: float
    """The initial power of the beam in W (not MW!)."""

    max_harmonic: int = dataclass_field(default=2, metadata={"static": True})
    """Highest cyclotron harmonic included in the absorption calculation.
    Controls both the KO-tensor FLR order and the number of resonance integrals
    evaluated. Increase to 3 for 3rd-harmonic scenarios, etc. Defaults to 2."""


@jax.tree_util.register_dataclass
@dataclass
class BeamProfile:
    """Beam profile in real space resulting from tracing."""

    position: jt.Float[jax.Array, "npoints 3"]
    """The position of the beam in cartesian coordinates."""

    arc_length: jt.Float[jax.Array, " npoints"]
    """The arc length along the beam."""

    refractive_index: jt.Float[jax.Array, "npoints 3"]
    """The refractive index vector at each point along the beam."""

    optical_depth: jt.Float[jax.Array, " npoints"]
    """The optical depth along the beam."""

    absorption_coefficient: jt.Float[jax.Array, " npoints"]
    """The absorption coefficient along the beam."""

    electron_density: jt.Float[jax.Array, " npoints"]
    """The electron density along the beam in units of $10^{20}$ m$^{-3}."""

    electron_temperature: jt.Float[jax.Array, " npoints"]
    """The electron temperature along the beam in keV."""

    magnetic_field: jt.Float[jax.Array, "npoints 3"]
    """The magnetic field vector along the beam in T."""

    normalized_effective_radius: jt.Float[jax.Array, " npoints"]
    r"""The normalized effective minor radius $\rho$ along the beam."""

    linear_power_density: jt.Float[jax.Array, " npoints"]
    """The linear power density along the beam."""


@jax.tree_util.register_dataclass
@dataclass
class RadialProfile:
    """Beam profile in radial coordinates."""

    rho: jt.Float[jax.Array, " npoints"]
    """The normalized effective minor radius."""

    volumetric_power_density: jt.Float[jax.Array, " npoints"]
    """The volumetric power density in W/m³."""


@jax.tree_util.register_dataclass
@dataclass
class TraceResult:
    """The result of a ray tracing calculation."""

    beam_profile: BeamProfile
    """The traced beam profile in real space."""

    radial_profile: RadialProfile
    """The radial deposition profile."""

    absorbed_power: jt.Float[jax.Array, ""]
    """Total power absorbed by the plasma in W."""

    absorbed_power_fraction: jt.Float[jax.Array, ""]
    """Fraction of input beam power absorbed by the plasma, i.e. $1 - e^{-\\tau}$."""

    optical_depth: jt.Float[jax.Array, ""]
    r"""Total optical depth $\tau$ accumulated along the ray."""

    deposition_rho_mean: jt.Float[jax.Array, ""]
    r"""Flux-surface-volume-weighted mean normalised radius $\langle\rho\rangle$ of power deposition."""

    deposition_rho_std: jt.Float[jax.Array, ""]
    r"""Flux-surface-volume-weighted standard deviation of $\rho$ for power deposition."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TracerSettings:
    """Settings for the ray tracing ODE solver.

    All float fields are JAX-traceable (changing them does **not** trigger
    recompilation). Only ``max_steps`` in the solver — which controls static
    buffer sizes — would require recompilation, and is intentionally omitted
    here for that reason.
    """

    relative_tolerance: float = 1e-4
    """Relative tolerance of the adaptive step-size controller."""

    absolute_tolerance: float = 1e-6
    """Absolute tolerance of the adaptive step-size controller."""

    max_step_size: float = 0.05
    """Maximum ODE step size in metres (arc length). Decreasing this improves
    accuracy at the cost of more steps."""

    max_arc_length: float = 20.0
    """Maximum arc length to integrate in metres before the solver gives up."""


@jax.tree_util.register_dataclass
@dataclass
class RadialProfiles:
    r"""Radial profiles of electron density and temperature.

    Defines the plasma parameters (electron density, temperature) as functions of the
    normalized effective radius $\rho$.
    """

    rho: jt.Float[jax.Array, " nrho"]
    """The normalized effective minor radius grid."""

    electron_density: jt.Float[jax.Array, " nrho"]
    """The electron density profile in units of $10^{20}$ m$^{-3}$."""

    electron_temperature: jt.Float[jax.Array, " nrho"]
    """The electron temperature profile in keV."""

    def with_zero_density_at_boundary(
        self, boundary_layer_width: float
    ) -> "RadialProfiles":
        r"""Return a copy with the electron density tapered to zero near the LCFS.

        Applies a cosine taper to `electron_density` over the outermost
        ``boundary_layer_width`` fraction of the minor radius.  The taper multiplies
        the profile by

        $$w(\rho) = \frac{1}{2}\left[1 + \cos\!\left(
            \pi\,\frac{\rho^2 - \rho_1^2}{\rho_{\max}^2 - \rho_1^2}
        \right)\right], \quad \rho_1 \le \rho \le \rho_{\max},$$

        where $\rho_1 = \rho_{\max} - \text{boundary\_layer\_width}$
        (in $s = \rho^2$ space).

        Use this whenever $n_e(\rho{=}1) > 0$ to avoid a hard
        discontinuity at the plasma–vacuum interface that would cause spurious
        ray behaviour.

        Args:
            boundary_layer_width: Width of the taper in $\rho$ units
                (fraction of the minor radius).  Must satisfy
                ``0 < boundary_layer_width <= rho_max``.

        Returns:
            A new `RadialProfiles` with tapered `electron_density`
            and the same `rho` and `electron_temperature`.
        """
        rho_max = float(jnp.max(self.rho))
        if rho_max <= 0.0:
            raise ValueError(f"rho_max must be positive, got {rho_max!r}.")
        if not (0.0 < boundary_layer_width <= rho_max):
            raise ValueError(
                f"boundary_layer_width must be in (0, rho_max={rho_max!r}], "
                f"got {boundary_layer_width!r}."
            )
        s2 = rho_max**2
        s1 = (rho_max - boundary_layer_width) ** 2
        s = self.rho**2
        weight = jnp.where(
            s <= s1,
            1.0,
            jnp.where(
                s >= s2,
                0.0,
                0.5 * (1.0 + jnp.cos(jnp.pi * (s - s1) / (s2 - s1))),
            ),
        )
        return RadialProfiles(
            rho=self.rho,
            electron_density=self.electron_density * weight,
            electron_temperature=self.electron_temperature,
        )
