from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import jax
import jaxtyping as jt


@runtime_checkable
class WoutLike(Protocol):
    """Protocol for objects that can be used as VmecWOut."""

    rmnc: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    zmns: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    xm: jt.Int[jax.Array, "n_fourier_coefficients"]
    xn: jt.Int[jax.Array, "n_fourier_coefficients"]
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


@dataclass
class RadialProfile:
    """The deposition profile projected onto the radial coordinate."""

    rho: jt.Float[jax.Array, "npoints"]
    """The normalized effective minor radius."""


@dataclass
class TracingResult:
    """The result of a ray tracing calculation."""

    beam_profile: BeamProfile
    """The traced beam profile."""

    radial_profile: RadialProfile
    """The radial deposition profile."""


@dataclass
class EquilibriumInterpolator:
    """Dataclass representing interpolation data for an MHD equilibrium."""

    rphiz: jt.Float[jax.Array, "npoints 3"]
    """The (r, phi, z) coordinates of the points on the interpolation grid."""

    magnetic_field: jt.Float[jax.Array, "npoints 3"]
    """The magnetic field at each point on the interpolation grid."""

    rho: jt.Float[jax.Array, "npoints"]
    """The normalized effective minor radius at each point on the interpolation grid."""

@dataclass
class RadialProfiles:
    """Dataclass for holding the electron radial profiles."""

    rho: jt.Float[jax.Array, "nrho"]
    """The normalized effective minor radius grid."""

    electron_density: jt.Float[jax.Array, "nrho"]
    """The electron density profile in units of 10^20 m^-3."""

    electron_temperature: jt.Float[jax.Array, "nrho"]
    """The electron temperature profile in keV."""
