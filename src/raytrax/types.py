from dataclasses import dataclass, fields
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import interpax
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


@runtime_checkable
class WoutLike(Protocol):
    """Protocol for objects that can be used as VmecWOut."""

    rmnc: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    zmns: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    xm: jt.Int[jax.Array, " n_fourier_coefficients"]
    xn: jt.Int[jax.Array, " n_fourier_coefficients"]
    gmnc: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    gmns: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    bsupumnc: jt.Float[jax.Array, "n_fourier_coefficients_nyquist n_surfaces"]
    bsupvmnc: jt.Float[jax.Array, "n_fourier_coefficients_nyquist n_surfaces"]
    xm_nyq: jt.Int[jax.Array, " n_fourier_coefficients_nyquist"]
    xn_nyq: jt.Int[jax.Array, " n_fourier_coefficients_nyquist"]
    ns: int
    nfp: int
    lasym: bool


@dataclass
class Beam:
    """Beam parameter inputs for tracing."""

    position: jt.Float[jax.Array, "3"]
    """The starting position of the beam in cartesian coordinates."""

    direction: jt.Float[jax.Array, "3"]
    """The starting direction of the beam in cartesian coordinates. Must be a unit vector."""

    frequency: jt.Float[jax.Array, ""]
    """The frequency of the beam in Hz (not GHz!)."""

    mode: Literal["X", "O"]
    """The polarization mode of the beam, either `"X"` for extraordinary or `"O"` for ordinary mode."""

    power: float
    """The initial power of the beam in W (not MW!)."""


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


@dataclass
class RadialProfile:
    """Beam profile in radial coordinates."""

    rho: jt.Float[jax.Array, " npoints"]
    """The normalized effective minor radius."""

    volumetric_power_density: jt.Float[jax.Array, " npoints"]
    """The volumetric power density in W/m³."""


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
