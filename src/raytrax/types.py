from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable, Callable

import interpax
import jax
import jaxtyping as jt


class SafetensorsMixin:
    """Mixin for dataclasses to add safetensors save/load functionality."""

    def save(self, path: str) -> None:
        """Save dataclass to a safetensors file.

        Args:
            path: Path to save to (should end in .safetensors)
        """
        from dataclasses import fields
        from safetensors.numpy import save_file
        import numpy as np

        # Separate arrays from scalars
        tensors = {}
        metadata = {}

        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, jax.Array):
                # Convert JAX arrays to numpy (zero-copy on CPU)
                tensors[field.name] = np.asarray(value)
            else:
                # Store scalars as metadata strings
                metadata[field.name] = str(value)

        save_file(tensors, path, metadata=metadata)

    @classmethod
    def load(cls, path: str):
        """Load dataclass from a safetensors file.

        Args:
            path: Path to the safetensors file to load

        Returns:
            Loaded instance of the dataclass
        """
        from dataclasses import fields
        from safetensors.numpy import load_file
        from safetensors import safe_open
        import jax.numpy as jnp

        # Load tensors and metadata
        tensors = load_file(path)

        with safe_open(path, framework="numpy") as f:
            metadata = f.metadata() or {}

        # Reconstruct all fields
        field_values = {}
        for field in fields(cls):
            if field.name in tensors:
                # It's an array - convert back to JAX
                field_values[field.name] = jnp.array(tensors[field.name])
            elif field.name in metadata:
                # It's a scalar - parse back from string
                value_str = metadata[field.name]
                if field.type == int:
                    field_values[field.name] = int(value_str)
                elif field.type == bool:
                    field_values[field.name] = value_str == "True"
                else:
                    field_values[field.name] = value_str

        return cls(**field_values)


@runtime_checkable
class WoutLike(Protocol):
    """Protocol for objects that can be used as VmecWOut."""

    rmnc: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    zmns: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    xm: jt.Int[jax.Array, "n_fourier_coefficients"]
    xn: jt.Int[jax.Array, "n_fourier_coefficients"]
    gmnc: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    gmns: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    bsupumnc: jt.Float[jax.Array, "n_fourier_coefficients_nyquist n_surfaces"]
    bsupvmnc: jt.Float[jax.Array, "n_fourier_coefficients_nyquist n_surfaces"]
    xm_nyq: jt.Int[jax.Array, "n_fourier_coefficients_nyquist"]
    xn_nyq: jt.Int[jax.Array, "n_fourier_coefficients_nyquist"]
    ns: int
    nfp: int
    lasym: bool


@dataclass
class Beam:
    """Dataclass for a beam to be traced."""

    position: jt.Float[jax.Array, "3"]
    """The starting position of the beam in cartesian coordinates."""

    direction: jt.Float[jax.Array, "3"]
    """The starting direction of the beam in cartesian coordinates."""

    frequency: jt.Float[jax.Array, ""]
    """The frequency of the beam in Hz."""

    mode: Literal["X", "O"]
    """The polarization mode of the beam, either 'X' or 'O'."""


@dataclass
class BeamProfile:
    """Dataclass for a traced beam profile."""

    position: jt.Float[jax.Array, "npoints 3"]
    """The position of the beam in cartesian coordinates."""

    arc_length: jt.Float[jax.Array, "npoints"]
    """The arc length along the beam."""

    refractive_index: jt.Float[jax.Array, "npoints 3"]
    """The refractive index at each point along the beam."""

    optical_depth: jt.Float[jax.Array, "npoints"]
    """The optical depth along the beam."""

    absorption_coefficient: jt.Float[jax.Array, "npoints"]
    """The absorption coefficient along the beam."""

    electron_density: jt.Float[jax.Array, "npoints"]
    """The electron density along the beam in units of 10^20 m^-3."""

    electron_temperature: jt.Float[jax.Array, "npoints"]
    """The electron temperature along the beam in keV."""

    magnetic_field: jt.Float[jax.Array, "npoints 3"]
    """The magnetic field vector along the beam in T."""

    normalized_effective_radius: jt.Float[jax.Array, "npoints"]
    """The normalized effective minor radius (rho) along the beam."""

    linear_power_density: jt.Float[jax.Array, "npoints"]
    """The linear power density along the beam."""


@dataclass
class RadialProfile:
    """The deposition profile projected onto the radial coordinate."""

    rho: jt.Float[jax.Array, "npoints"]
    """The normalized effective minor radius."""

    volumetric_power_density: jt.Float[jax.Array, "npoints"]
    """The volumetric power density in W/m³."""


@dataclass
class TracingResult:
    """The result of a ray tracing calculation."""

    beam_profile: BeamProfile
    """The traced beam profile."""

    radial_profile: RadialProfile
    """The radial deposition profile."""


@dataclass
class MagneticConfiguration(SafetensorsMixin):
    """Magnetic configuration and geometry on a cylindrical grid.

    Contains the magnetic field B and normalized effective radius rho on a
    3D cylindrical grid (r, phi, z), along with volume information for
    computing deposition profiles.
    """

    rphiz: jt.Float[jax.Array, "npoints 3"]
    """The (r, phi, z) coordinates of the points on the interpolation grid."""

    magnetic_field: jt.Float[jax.Array, "npoints 3"]
    """The magnetic field at each point on the interpolation grid."""

    rho: jt.Float[jax.Array, "npoints"]
    """The normalized effective minor radius at each point on the interpolation grid."""

    nfp: int
    """Number of field periods (toroidal periodicity)."""

    stellarator_symmetric: bool
    """Whether the configuration has stellarator symmetry."""

    rho_1d: jt.Float[jax.Array, "nrho_1d"]
    """1D radial grid for volume derivative."""

    dvolume_drho: jt.Float[jax.Array, "nrho_1d"]
    """Volume derivative dV/drho on the 1D radial grid."""

    @classmethod
    def from_vmec_wout(
        cls,
        equilibrium: WoutLike,
        magnetic_field_scale: float = 1.0,
    ) -> "MagneticConfiguration":
        """Generate interpolators for the given MHD equilibrium.

        Args:
            equilibrium: an MHD equilibrium compatible with `vmecpp.VmecWOut`
            magnetic_field_scale: Factor to multiply all magnetic field values by.

        Returns:
            A MagneticConfiguration object containing interpolation data.
        """
        from .fourier import dvolume_drho as compute_dvolume_drho
        from .interpolate import cylindrical_grid_for_equilibrium
        import jax.numpy as jnp

        # TODO add settings for grid resolution
        interpolated_array = cylindrical_grid_for_equilibrium(
            equilibrium=equilibrium, n_rho=40, n_theta=45, n_phi=50, n_r=45, n_z=55
        )
        rphiz = interpolated_array[..., :3]
        rho = interpolated_array[..., 3]
        magnetic_field = interpolated_array[..., 4:] * magnetic_field_scale

        # Compute volume derivative on 1D radial grid
        rho_1d = jnp.linspace(0, 1, 200)
        dv_drho = compute_dvolume_drho(equilibrium, rho_1d)

        return cls(
            rphiz=rphiz,
            magnetic_field=magnetic_field,
            rho=rho,
            nfp=equilibrium.nfp,
            stellarator_symmetric=not equilibrium.lasym,
            rho_1d=rho_1d,
            dvolume_drho=dv_drho,
        )


@dataclass
class RadialProfiles:
    """Radial profiles of electron density and temperature.

    Defines the plasma parameters (density, temperature) as functions of the
    normalized effective radius rho.
    """

    rho: jt.Float[jax.Array, "nrho"]
    """The normalized effective minor radius grid."""

    electron_density: jt.Float[jax.Array, "nrho"]
    """The electron density profile in units of 10^20 m^-3."""

    electron_temperature: jt.Float[jax.Array, "nrho"]
    """The electron temperature profile in keV."""


@dataclass(frozen=True)
class Interpolators:
    """Bundle of interpolation functions for ray tracing.

    Groups the four interpolators needed by the ODE solver into a single
    pytree-compatible object passed through JIT as one argument.
    """

    magnetic_field: interpax.Interpolator3D
    """Interpolator for B(r, phi, z) on the fundamental domain."""

    rho: interpax.Interpolator3D
    """Interpolator for rho(r, phi, z) on the fundamental domain."""

    electron_density: interpax.Interpolator1D
    """Interpolator for ne(rho)."""

    electron_temperature: interpax.Interpolator1D
    """Interpolator for Te(rho)."""


jax.tree_util.register_pytree_node(
    Interpolators,
    lambda i: (
        (i.magnetic_field, i.rho, i.electron_density, i.electron_temperature),
        None,
    ),
    lambda _, children: Interpolators(
        magnetic_field=children[0],
        rho=children[1],
        electron_density=children[2],
        electron_temperature=children[3],
    ),
)
