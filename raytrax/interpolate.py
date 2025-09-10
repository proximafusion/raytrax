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
from .types import WoutLike, EquilibriumInterpolator, RadialProfiles


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
    rho_theta_phi = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, 1, n_rho),
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
    equilibrium_interpolator: EquilibriumInterpolator,
) -> Callable[[jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]]:
    """Build a magnetic field interpolator from the equilibrium interpolator."""
    Bxyz = equilibrium_interpolator.magnetic_field
    rphiz = equilibrium_interpolator.rphiz
    interpolator = interpax.Interpolator3D(
        x=rphiz[:, 0, 0, 0],
        y=rphiz[0, :, 0, 1],
        z=rphiz[0, 0, :, 2],
        f=Bxyz,
        method="linear",
        # FIXME not a reasonable value
        extrap=0.0,
    )

    def interpolator_cartesian(
        position: jt.Float[jax.Array, "3"],
    ) -> jt.Float[jax.Array, "3"]:
        # TODO handle symmetries
        # TODO handle out of bounds
        return interpolator(
            jnp.sqrt(position[0] ** 2 + position[1] ** 2),
            jnp.arctan2(position[1], position[0]),
            position[2],
        )

    return interpolator_cartesian


def build_radial_interpolators(
    equilibrium_interpolator: EquilibriumInterpolator,
    radial_profiles: RadialProfiles,
) -> tuple[
    Callable[[jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]],
    Callable[[jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]],
]:
    """Build radial interpolators for the given equilibrium and radial profiles.

    Args:
        equilibrium_interpolator: The equilibrium interpolator.
        radial_profiles: The radial profiles.

    Returns:
        A tuple of two functions that interpolate the electron density and
        temperature at a given position.
    """
    rho = equilibrium_interpolator.rho
    rphiz = equilibrium_interpolator.rphiz

    rho_interpolator = interpax.Interpolator3D(
        x=rphiz[:, 0, 0, 0],
        y=rphiz[0, :, 0, 1],
        z=rphiz[0, 0, :, 2],
        f=rho,
        method="linear",
        # When outside of the grid, return a value greater than 1
        extrap=1.1,
    )
    Te_interpolator = interpax.Interpolator1D(
        x=radial_profiles.rho,
        f=radial_profiles.electron_temperature,
        method="linear",
        extrap=0.0,
    )
    ne_interpolator = interpax.Interpolator1D(
        x=radial_profiles.rho,
        f=radial_profiles.electron_density,
        method="linear",
        extrap=0.0,
    )

    def rho_interpolator_cartesian(
        position: jt.Float[jax.Array, "3"],
    ) -> jt.Float[jax.Array, ""]:
        # TODO handle symmetries
        return rho_interpolator(
            jnp.sqrt(position[0] ** 2 + position[1] ** 2),
            jnp.arctan2(position[1], position[0]),
            position[2],
        )

    def ne_interpolator_cartesian(
        position: jt.Float[jax.Array, "3"],
    ) -> jt.Float[jax.Array, ""]:
        rho_at_position = rho_interpolator_cartesian(position)
        return ne_interpolator(rho_at_position)

    def Te_interpolator_cartesian(
        position: jt.Float[jax.Array, "3"],
    ) -> jt.Float[jax.Array, ""]:
        rho_at_position = rho_interpolator_cartesian(position)
        return Te_interpolator(rho_at_position)

    return ne_interpolator_cartesian, Te_interpolator_cartesian
