from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable, Callable

import interpax
import jax
import jaxtyping as jt


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
class MagneticConfiguration:
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
