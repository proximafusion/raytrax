r"""Utilities for interpolating flux surface quantities on a cylindrical grid.

Since ray tracing is computed in cartesian space (which can easily be transformed
to/from cylindrical coordinates), the effective minor radius $\rho$ and the magnetic
field $B$ need to be extrapolated from toroidal coordinates ($\rho$, $\theta$, $\phi$)
to cylindrical coordinates ($r$, $\phi$, $z$).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv

import interpax
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from beartype import beartype as typechecker
from scipy.interpolate import griddata

from raytrax.equilibrium.protocol import WoutLike
from raytrax.types import RadialProfiles, SafetensorsMixin

from .fourier import dvolume_drho as compute_dvolume_drho
from .fourier import (
    evaluate_magnetic_field_on_toroidal_grid,
    evaluate_rphiz_on_toroidal_grid,
)


@jax.tree_util.register_dataclass
@dataclass
class MagneticConfiguration(SafetensorsMixin):
    r"""Magnetic configuration and geometry on a cylindrical grid.

    Contains the magnetic field $\boldsymbol{B}$ and normalized effective radius $\rho$ on a
    3D cylindrical grid ($R$, $\phi$, $Z$), along with volume information for
    computing deposition profiles.
    """

    rphiz: jt.Float[jax.Array, "npoints 3"]
    r"""The ($R$, $\phi$, $Z$) coordinates of the points on the interpolation grid."""

    magnetic_field: jt.Float[jax.Array, "npoints 3"]
    r"""The magnetic field $(B_R, B_\phi, B_Z)$ in cylindrical components at each grid point."""

    rho: jt.Float[jax.Array, " npoints"]
    """The normalized effective minor radius at each point on the interpolation grid."""

    nfp: int = dataclass_field(metadata={"static": True})
    """Number of field periods (toroidal periodicity)."""

    is_stellarator_symmetric: bool = dataclass_field(metadata={"static": True})
    """Whether the configuration has stellarator symmetry."""

    rho_1d: jt.Float[jax.Array, " nrho_1d"]
    """1D radial grid for volume derivative."""

    dvolume_drho: jt.Float[jax.Array, " nrho_1d"]
    r"""Volume derivative $dV/d\rho$ on the 1D radial grid."""

    is_axisymmetric: bool = dataclass_field(default=False, metadata={"static": True})
    """Whether the configuration is axisymmetric (tokamak) or 3D (stellarator)."""

    @classmethod
    def from_vmec_wout(
        cls,
        equilibrium: WoutLike,
        magnetic_field_scale: float = 1.0,
    ) -> MagneticConfiguration:
        """Create a MagneticConfiguration from a VMEC++ equilibrium.

        Optionally, apply a uniform scale factor to the magnetic field magnitude
        with respect to the equilibrium data.

        Args:
            equilibrium: an MHD equilibrium compatible with `vmecpp.VmecWOut`
            magnetic_field_scale: Factor to multiply all magnetic field values by.

        Returns:
            A MagneticConfiguration object containing interpolation data.
        """

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
            is_stellarator_symmetric=not equilibrium.lasym,
            rho_1d=rho_1d,
            dvolume_drho=dv_drho,
        )

    def to_pyvista_grid(self) -> pv.StructuredGrid:
        """Convert to a PyVista StructuredGrid in Cartesian coordinates.

        Point arrays written:

        - ``rho``: normalised effective minor radius (NaN outside LCFS)
        - ``B``: magnetic field vector :math:`(B_R, B_\\phi, B_Z)` in T
        - ``absB``: magnetic field magnitude :math:`|\\boldsymbol{B}|` in T

        Requires the ``pyvista`` package.

        Returns:
            A :class:`pyvista.StructuredGrid` that can be visualised or saved
            directly via ``.save("output.vts")``.
        """
        import pyvista as pv

        rphiz = np.array(self.rphiz)
        R, phi, Z = rphiz[..., 0], rphiz[..., 1], rphiz[..., 2]

        B = np.array(self.magnetic_field)
        rho = np.array(self.rho)

        grid = pv.StructuredGrid(R * np.cos(phi), R * np.sin(phi), Z)
        grid["rho"] = rho.flatten(order="F")  # NaN outside LCFS
        grid["B"] = B.reshape(-1, 3, order="F")
        grid["absB"] = np.linalg.norm(B, axis=-1).flatten(order="F")

        return grid


@jt.jaxtyped(typechecker=typechecker)
def interpolate_toroidal_to_cylindrical_grid(
    rphiz_toroidal: jt.Float[jax.Array, "n_rho n_theta n_phi rphiz=3"],
    rz_cylindrical: jt.Float[jax.Array, "n_r n_z rz=2"],
    value_toroidal: jt.Float[jax.Array, "n_rho n_theta n_phi n_values"],
) -> jt.Float[jax.Array, "n_r n_phi n_z n_values"]:
    """Interpolate toroidal coordinates to a cylindrical grid.

    Args:
    - rphiz_toroidal: An array of cylindrical coordinates (the last dimension
        are the three coordinates R, phi, Z) on the toroidal grid.
    - rz_cylindrical: The cylindrical coordinates (R, Z) on the cylindrical grid.
        Note that phi does not need to be provided because we use the same phi
        grid as for the toroidal grid, so no interpolation in phi is needed.
    - value_toroidal: The values of the quantities to be interpolated on the
        toroidal grid nodes.

    Returns:
    The values of the quantities interpolated to the cylindrical grid as a JAX
    array.
    """
    n_r, n_z, _ = rz_cylindrical.shape
    n_values = value_toroidal.shape[-1]
    values = []
    phis = rphiz_toroidal[0, 0, :, 1]
    for i_phi, _ in enumerate(phis):
        # this uses plain numpy, because griddata is anyway not JAX compatible
        value = griddata(
            np.array(rphiz_toroidal[:, :, i_phi, ::2]).reshape(-1, 2),
            np.array(value_toroidal[:, :, i_phi]).reshape(-1, n_values),
            np.array(rz_cylindrical).reshape(-1, 2),
            method="linear",
        ).reshape(n_r, n_z, n_values)
        values.append(value)
    return jnp.stack(values, axis=1)


@jt.jaxtyped(typechecker=typechecker)
def cylindrical_grid_for_equilibrium(
    equilibrium: WoutLike, n_rho: int, n_theta: int, n_phi: int, n_r: int, n_z: int
) -> jt.Float[jax.Array, "n_r n_phi n_z rphizrhoBxyz=7"]:
    """Compute the effective minor radius and the magnetic field on a cylindrical grid.

    Args:
        equilibrium: The MHD equilibrium.
        n_rho: Number of radial points.
        n_theta: Number of poloidal points.
        n_phi: Number of toroidal points (equal in both grids!).
        n_r: Number of radial points in the cylindrical grid.
        n_z: Number of vertical points in the cylindrical grid.

    Returns:
        A JAX array of values on the cylindrical grid with shape (n_r, n_phi, n_z, 7).
        The last dimension is a vector with the following components:
        (R, phi, Z, rho, B_R, B_phi, B_Z) where the magnetic field components
        are in cylindrical coordinates.
    """
    if equilibrium.lasym:
        raise NotImplementedError(
            "Non stellarator symmetric equilibria are not supported yet."
        )
    phi_max = 2 * jnp.pi / equilibrium.nfp / 2
    # Extend grid to rho_max = 1.2 to allow extrapolation beyond LCMS
    rho_max = 1.2
    rho_theta_phi = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, rho_max, n_rho),
            jnp.linspace(0, 2 * jnp.pi, n_theta),
            jnp.linspace(0, phi_max, n_phi),
            indexing="ij",
        ),
        axis=-1,
    )
    rphiz = evaluate_rphiz_on_toroidal_grid(
        equilibrium,
        rho_theta_phi,
    )
    Bxyz = evaluate_magnetic_field_on_toroidal_grid(
        equilibrium,
        rho_theta_phi,
    )
    # Convert Cartesian B to cylindrical (B_R, B_phi, B_Z) for storage
    phi = rphiz[..., 1]
    cp, sp = jnp.cos(phi), jnp.sin(phi)
    Bcyl = jnp.stack(
        [
            Bxyz[..., 0] * cp + Bxyz[..., 1] * sp,
            -Bxyz[..., 0] * sp + Bxyz[..., 1] * cp,
            Bxyz[..., 2],
        ],
        axis=-1,
    )
    rmin = jnp.min(rphiz[..., 0])
    rmax = jnp.max(rphiz[..., 0])
    zmin = jnp.min(rphiz[..., 2])
    zmax = jnp.max(rphiz[..., 2])
    rz_cylindrical = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(rmin, rmax, n_r),
            jnp.linspace(zmin, zmax, n_z),
            indexing="ij",
        ),
        axis=-1,
    )
    value_toroidal = jnp.concatenate([rho_theta_phi[..., :1], Bcyl], axis=-1)
    rhoBxyz_cylindrical = interpolate_toroidal_to_cylindrical_grid(
        rphiz_toroidal=rphiz,
        rz_cylindrical=rz_cylindrical,
        value_toroidal=value_toroidal,
    )
    rphiz_cylindrical = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(rmin, rmax, n_r),
            jnp.linspace(0, phi_max, n_phi),
            jnp.linspace(zmin, zmax, n_z),
            indexing="ij",
        ),
        axis=-1,
    )
    result = jnp.concatenate([rphiz_cylindrical, rhoBxyz_cylindrical], axis=-1)
    return result


def build_magnetic_field_interpolator(
    magnetic_configuration: MagneticConfiguration,
) -> interpax.Interpolator3D | interpax.Interpolator2D:
    """Build a magnetic field interpolator from the magnetic configuration.

    For stellarators, returns an interpax.Interpolator3D that interpolates
    (B_R, B_phi, B_Z) in cylindrical coordinates (R, phi, Z) on the
    fundamental domain [0, pi/nfp].

    For axisymmetric equilibria, returns an interpax.Interpolator2D that
    interpolates (B_R, B_phi, B_Z) as a function of (R, Z).
    """
    if magnetic_configuration.is_axisymmetric:
        B_cyl = magnetic_configuration.magnetic_field
        rphiz = magnetic_configuration.rphiz
        return interpax.Interpolator2D(
            x=rphiz[:, 0, 0, 0],
            y=rphiz[0, 0, :, 2],
            f=jnp.nan_to_num(B_cyl[:, 0, :, :], nan=0.0),
            method="linear",
            extrap=0.0,
        )

    if not magnetic_configuration.is_stellarator_symmetric:
        raise NotImplementedError(
            "Non stellarator-symmetric equilibria not yet supported"
        )

    B_cyl = magnetic_configuration.magnetic_field
    rphiz = magnetic_configuration.rphiz

    return interpax.Interpolator3D(
        x=rphiz[:, 0, 0, 0],
        y=rphiz[0, :, 0, 1],
        z=rphiz[0, 0, :, 2],
        f=jnp.nan_to_num(B_cyl, nan=0.0),
        method="linear",
        extrap=0.0,
    )


def build_rho_interpolator(
    magnetic_configuration: MagneticConfiguration,
) -> interpax.Interpolator3D | interpax.Interpolator2D:
    """Build rho interpolator for the given magnetic configuration.

    For stellarators, returns an interpax.Interpolator3D that interpolates rho
    in cylindrical coordinates (R, phi, Z) on the fundamental domain [0, pi/nfp].

    For axisymmetric equilibria, returns an interpax.Interpolator2D that
    interpolates rho as a function of (R, Z).
    """
    if magnetic_configuration.is_axisymmetric:
        rho = magnetic_configuration.rho
        rphiz = magnetic_configuration.rphiz
        return interpax.Interpolator2D(
            x=rphiz[:, 0, 0, 0],
            y=rphiz[0, 0, :, 2],
            f=jnp.nan_to_num(rho[:, 0, :], nan=1.1),
            method="linear",
            extrap=1.1,
        )

    if not magnetic_configuration.is_stellarator_symmetric:
        raise NotImplementedError(
            "Non stellarator-symmetric equilibria not yet supported"
        )

    rho = magnetic_configuration.rho
    rphiz = magnetic_configuration.rphiz

    return interpax.Interpolator3D(
        x=rphiz[:, 0, 0, 0],
        y=rphiz[0, :, 0, 1],
        z=rphiz[0, 0, :, 2],
        f=jnp.nan_to_num(rho, nan=1.1),
        method="linear",
        extrap=1.1,
    )


def build_electron_density_profile_interpolator(
    radial_profiles: RadialProfiles,
) -> interpax.Interpolator1D:
    """Build electron density profile interpolator.

    Args:
        radial_profiles: The radial profiles.

    Returns:
        An interpax.Interpolator1D that maps rho to electron density.
    """
    return interpax.Interpolator1D(
        x=radial_profiles.rho,
        f=radial_profiles.electron_density,
        method="linear",
        extrap=0.0,
    )


def build_electron_temperature_profile_interpolator(
    radial_profiles: RadialProfiles,
) -> interpax.Interpolator1D:
    """Build electron temperature profile interpolator.

    Args:
        radial_profiles: The radial profiles.

    Returns:
        An interpax.Interpolator1D that maps rho to electron temperature.
    """
    return interpax.Interpolator1D(
        x=radial_profiles.rho,
        f=radial_profiles.electron_temperature,
        method="linear",
        extrap=0.0,
    )


def build_radial_interpolators(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
) -> tuple[
    interpax.Interpolator3D | interpax.Interpolator2D,
    interpax.Interpolator1D,
    interpax.Interpolator1D,
]:
    """Build radial interpolators for the given magnetic configuration and radial profiles.

    This is a convenience function that combines the individual interpolator builders.
    For cleaner code, consider using the individual functions directly.

    Args:
        magnetic_configuration: The magnetic configuration.
        radial_profiles: The radial profiles.

    Returns:
        A tuple of three functions:
        - rho_interpolator: maps position to radial coordinate (rho)
        - electron_density_profile_interpolator: maps rho to electron density
        - electron_temperature_profile_interpolator: maps rho to electron temperature
    """
    rho_interpolator = build_rho_interpolator(magnetic_configuration)
    ne_profile_interpolator = build_electron_density_profile_interpolator(
        radial_profiles
    )
    Te_profile_interpolator = build_electron_temperature_profile_interpolator(
        radial_profiles
    )

    return rho_interpolator, ne_profile_interpolator, Te_profile_interpolator
