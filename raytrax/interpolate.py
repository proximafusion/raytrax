import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from beartype import beartype as typechecker
from scipy.interpolate import griddata

from .fourier import (
    evaluate_magnetic_field_on_toroidal_grid,
    evaluate_rphiz_on_toroidal_grid,
)
from .types import WoutLike


@jt.jaxtyped(typechecker=typechecker)
def interpolate_toroidal_to_cylindrical_grid(
    rphiz_toroidal: jt.Float[jax.Array, "n_rho n_theta n_phi rphiz=3"],
    rz_cylindrical: jt.Float[jax.Array, "n_r n_z rz=2"],
    value_toroidal: jt.Float[jax.Array, "n_rho n_theta n_phi n_values"],
) -> jt.Float[jax.Array, "n_r n_phi n_z *dims"]:
    """Interpolate toroidal coordinates to a cylindrical grid."""
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
) -> jt.Float[jax.Array, "n_r n_phi n_z rhoBxyz=4"]:
    """Create a cylindrical grid for the given equilibrium."""
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
    value_toroidal = jnp.concatenate([rphiz[..., :1], Bxyz[..., :]], axis=-1)
    return interpolate_toroidal_to_cylindrical_grid(
        rphiz_toroidal=rphiz,
        rz_cylindrical=rz_cylindrical,
        value_toroidal=value_toroidal,
    )
