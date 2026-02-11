r"""Utilities for interpolating flux surface quantities on a cylindrical grid.

Since ray tracing is computed in cartesian space (which can easily be transformed
to/from cylindrical coordinates), the effective minor radius $\rho$ and the magnetic
field $B$ need to be extrapolated from toroidal coordinates ($\rho$, $\theta$, $\phi$)
to cylindrical coordinates ($r$, $\phi$, $z$).
"""

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import interpax
from typing import Callable
from beartype import beartype as typechecker
from scipy.interpolate import griddata

from .fourier import (
    evaluate_magnetic_field_on_toroidal_grid,
    evaluate_rphiz_on_toroidal_grid,
)
from .types import WoutLike, MagneticConfiguration, RadialProfiles


def _map_to_fundamental_domain(
    phi: jt.Float[jax.Array, ""],
    z: jt.Float[jax.Array, ""],
    nfp: int,
) -> tuple[
    jt.Float[jax.Array, ""],
    jt.Float[jax.Array, ""],
    jt.Bool[jax.Array, ""],
]:
    """Map toroidal angle and z to the fundamental domain [0, π/nfp] using stellarator symmetry.

    For points in the second half of a field period (phi_mod > π/nfp), the stellarator
    symmetry maps (R, phi, Z) to (R, phi_mapped, -Z), so z must also be reflected.

    Args:
        phi: Toroidal angle in radians (can be any value)
        z: Cylindrical z coordinate
        nfp: Number of field periods

    Returns:
        A tuple of (phi_mapped, z_query, in_second_half) where phi_mapped is in [0, π/nfp],
        z_query is the z to use for the grid lookup, and in_second_half indicates whether
        the original phi was in the second half of a field period.
    """
    period = 2.0 * jnp.pi / nfp
    half_period = jnp.pi / nfp
    phi_mod = phi % period
    in_second_half = phi_mod > half_period
    phi_mapped = jnp.where(in_second_half, period - phi_mod, phi_mod)
    z_query = jnp.where(in_second_half, -z, z)
    return phi_mapped, z_query, in_second_half


def _apply_B_stellarator_symmetry(
    B_grid: jt.Float[jax.Array, "3"],
    phi_mapped: jt.Float[jax.Array, ""],
    phi: jt.Float[jax.Array, ""],
    in_second_half: jt.Bool[jax.Array, ""],
) -> jt.Float[jax.Array, "3"]:
    """Apply stellarator symmetry transformation to a Cartesian B field vector.

    When phi is in the second half of a field period, the grid was queried at the
    mirror point (phi_mapped, -z). This function applies the correct physical
    transformation to recover B at the actual query point (phi, z).

    Under stellarator symmetry, B_R is odd (changes sign) while B_phi and B_Z
    are even (unchanged) when reflecting across the period boundary.

    Args:
        B_grid: Cartesian B vector from the grid at the mirror point
        phi_mapped: The mapped phi used for the grid lookup
        phi: The actual toroidal angle of the query point
        in_second_half: Whether the query phi is in the second half of a field period

    Returns:
        Cartesian B vector at the actual query position
    """
    cp_m = jnp.cos(phi_mapped)
    sp_m = jnp.sin(phi_mapped)
    BR_m = B_grid[0] * cp_m + B_grid[1] * sp_m
    Bphi_m = -B_grid[0] * sp_m + B_grid[1] * cp_m
    BZ_m = B_grid[2]
    # B_R is odd under the symmetry (changes sign), B_phi and B_Z are even
    sign = jnp.where(in_second_half, -1.0, 1.0)
    BR_q = sign * BR_m
    cp_q = jnp.cos(phi)
    sp_q = jnp.sin(phi)
    return jnp.stack([BR_q * cp_q - Bphi_m * sp_q, BR_q * sp_q + Bphi_m * cp_q, BZ_m])


@jt.jaxtyped(typechecker=typechecker)
def interpolate_toroidal_to_cylindrical_grid(
    rphiz_toroidal: jt.Float[jax.Array, "n_rho n_theta n_phi rphiz=3"],
    rz_cylindrical: jt.Float[jax.Array, "n_r n_z rz=2"],
    value_toroidal: jt.Float[jax.Array, "n_rho n_theta n_phi n_values"],
) -> jt.Float[jax.Array, "n_r n_phi n_z n_values"]:
    """Interpolate toroidal coordinates to a cylindrical grid.

    Args:
    - rphiz_toroidal: An array of cylindrical coordinates (the last dimension
        are the three coordinates r, phi, z) on the toroidal grid.
    - rz_cylindrical: The cylindrical coordinates (r, z) on the cylindrical grid.
        Note that phi does not need to be provided because we use the same phi
        grid as for the toroidal grid, so no interpolation in phi is needed.
    - value_toroidal: The values of the quantities to be interpolated on the
        toroidal grid nodes.

    Returns:
    The values of the quantities interpolated to the cylindrical grid as a JAX
    array.
    """
    (n_r, n_z, _) = rz_cylindrical.shape
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
        (r, phi, z, rho, B_x, B_y, B_z).
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
    value_toroidal = jnp.concatenate([rho_theta_phi[..., :1], Bxyz[..., :]], axis=-1)
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
    equilibrium_interpolator: MagneticConfiguration,
) -> Callable[[jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]]:
    """Build a magnetic field interpolator from the equilibrium interpolator."""
    if not equilibrium_interpolator.stellarator_symmetric:
        raise NotImplementedError(
            "Non stellarator-symmetric equilibria not yet supported"
        )

    Bxyz = equilibrium_interpolator.magnetic_field
    rphiz = equilibrium_interpolator.rphiz
    nfp = equilibrium_interpolator.nfp

    interpolator = interpax.Interpolator3D(
        x=rphiz[:, 0, 0, 0],
        y=rphiz[0, :, 0, 1],
        z=rphiz[0, 0, :, 2],
        f=jnp.nan_to_num(Bxyz, nan=0.0),
        method="linear",
        extrap=0.0,
    )

    def interpolator_cartesian(
        position: jt.Float[jax.Array, "3"],
    ) -> jt.Float[jax.Array, "3"]:
        r = jnp.sqrt(position[0] ** 2 + position[1] ** 2)
        phi = jnp.arctan2(position[1], position[0])
        z = position[2]
        phi_mapped, z_query, in_second_half = _map_to_fundamental_domain(phi, z, nfp)
        B_grid = interpolator(r, phi_mapped, z_query)
        return _apply_B_stellarator_symmetry(B_grid, phi_mapped, phi, in_second_half)

    return interpolator_cartesian


def build_rho_interpolator(
    equilibrium_interpolator: MagneticConfiguration,
) -> Callable[[jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]]:
    """Build rho interpolator for the given equilibrium.

    Args:
        equilibrium_interpolator: The equilibrium interpolator.

    Returns:
        A function that maps position to radial coordinate (rho).
    """
    if not equilibrium_interpolator.stellarator_symmetric:
        raise NotImplementedError(
            "Non stellarator-symmetric equilibria not yet supported"
        )

    rho = equilibrium_interpolator.rho
    rphiz = equilibrium_interpolator.rphiz
    nfp = equilibrium_interpolator.nfp

    rho_interpolator = interpax.Interpolator3D(
        x=rphiz[:, 0, 0, 0],
        y=rphiz[0, :, 0, 1],
        z=rphiz[0, 0, :, 2],
        f=jnp.nan_to_num(rho, nan=1.1),
        method="linear",
        extrap=1.1,
    )

    def rho_interpolator_cartesian(
        position: jt.Float[jax.Array, "3"],
    ) -> jt.Float[jax.Array, ""]:
        r = jnp.sqrt(position[0] ** 2 + position[1] ** 2)
        phi = jnp.arctan2(position[1], position[0])
        z = position[2]
        phi_mapped, z_query, _ = _map_to_fundamental_domain(phi, z, nfp)
        return rho_interpolator(r, phi_mapped, z_query)

    return rho_interpolator_cartesian


def build_electron_density_profile_interpolator(
    radial_profiles: RadialProfiles,
) -> Callable[[jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]]:
    """Build electron density profile interpolator.

    Args:
        radial_profiles: The radial profiles.

    Returns:
        A function that maps rho to electron density.
    """
    ne_interpolator = interpax.Interpolator1D(
        x=radial_profiles.rho,
        f=radial_profiles.electron_density,
        method="linear",
        extrap=0.0,
    )

    def ne_profile_interpolator(
        rho: jt.Float[jax.Array, ""],
    ) -> jt.Float[jax.Array, ""]:
        return ne_interpolator(rho)

    return ne_profile_interpolator


def build_electron_temperature_profile_interpolator(
    radial_profiles: RadialProfiles,
) -> Callable[[jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]]:
    """Build electron temperature profile interpolator.

    Args:
        radial_profiles: The radial profiles.

    Returns:
        A function that maps rho to electron temperature.
    """
    Te_interpolator = interpax.Interpolator1D(
        x=radial_profiles.rho,
        f=radial_profiles.electron_temperature,
        method="linear",
        extrap=0.0,
    )

    def Te_profile_interpolator(
        rho: jt.Float[jax.Array, ""],
    ) -> jt.Float[jax.Array, ""]:
        return Te_interpolator(rho)

    return Te_profile_interpolator


def build_radial_interpolators(
    equilibrium_interpolator: MagneticConfiguration,
    radial_profiles: RadialProfiles,
) -> tuple[
    Callable[[jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]],
    Callable[[jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]],
    Callable[[jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]],
]:
    """Build radial interpolators for the given equilibrium and radial profiles.

    This is a convenience function that combines the individual interpolator builders.
    For cleaner code, consider using the individual functions directly.

    Args:
        equilibrium_interpolator: The configuration grid.
        radial_profiles: The radial profiles.

    Returns:
        A tuple of three functions:
        - rho_interpolator: maps position to radial coordinate (rho)
        - electron_density_profile_interpolator: maps rho to electron density
        - electron_temperature_profile_interpolator: maps rho to electron temperature
    """
    rho_interpolator = build_rho_interpolator(equilibrium_interpolator)
    ne_profile_interpolator = build_electron_density_profile_interpolator(
        radial_profiles
    )
    Te_profile_interpolator = build_electron_temperature_profile_interpolator(
        radial_profiles
    )

    return rho_interpolator, ne_profile_interpolator, Te_profile_interpolator
