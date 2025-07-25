from enum import Enum
from functools import partial
from typing import Protocol, runtime_checkable

import interpax
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from beartype import beartype as typechecker
from scipy.interpolate import griddata


@runtime_checkable
class WoutLike(Protocol):
    """Protocol for objects that can be used as VmecWOut."""

    rmnc: jt.Float[jax.Array, "n_surfaces n_fourier_coefficients"]
    zmns: jt.Float[jax.Array, "n_surfaces n_fourier_coefficients"]
    xm: jt.Int[jax.Array, "n_fourier_coefficients"]
    xn: jt.Int[jax.Array, "n_fourier_coefficients"]
    bsupumnc: jt.Float[jax.Array, "n_surfaces n_fourier_coefficients_nyquist"]
    bsupvmnc: jt.Float[jax.Array, "n_surfaces n_fourier_coefficients_nyquist"]
    xm_nyq: jt.Int[jax.Array, "n_fourier_coefficients_nyquist"]
    xn_nyq: jt.Int[jax.Array, "n_fourier_coefficients_nyquist"]
    ns: int
    lasym: bool


class FourierBasis(Enum):
    COS = "cos"
    SIN = "sin"


class FourierDerivative(Enum):
    NO = "no"
    TOROIDAL = "toroidal"
    POLOIDAL = "poloidal"


@partial(jax.jit, static_argnames=["basis", "derivative"])
@jt.jaxtyped(typechecker=typechecker)
def inverse_fourier_transform(
    fourier_coefficients: jt.Float[jax.Array, "n_rho n_fourier_coefficients"],
    poloidal_mode_numbers: jt.Int[jax.Array, " n_fourier_coefficients"],
    toroidal_mode_numbers: jt.Int[jax.Array, " n_fourier_coefficients"],
    rho_theta_phi: jt.Float[jax.Array, "n_rho n_theta n_phi rho_theta_phi=3"],
    basis: FourierBasis,
    derivative: FourierDerivative = FourierDerivative.NO,
) -> jt.Float[jax.Array, "n_rho n_theta n_phi"]:
    """Transform an array of Fourier coefficients into values on a toroidal grid."""
    m = poloidal_mode_numbers[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    n = toroidal_mode_numbers[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    theta = rho_theta_phi[:, jnp.newaxis, :, :, 1]
    phi = rho_theta_phi[:, jnp.newaxis, :, :, 2]
    angle = m * theta - n * phi
    coefficients = fourier_coefficients[:, :, jnp.newaxis, jnp.newaxis]
    if basis == FourierBasis.COS:
        if derivative == FourierDerivative.POLOIDAL:
            return jnp.sum(-m * coefficients * jnp.sin(angle), axis=1)
        elif derivative == FourierDerivative.TOROIDAL:
            return jnp.sum(n * coefficients * jnp.sin(angle), axis=1)
        else:
            return jnp.sum(coefficients * jnp.cos(angle), axis=1)
    elif basis == FourierBasis.SIN:
        if derivative == FourierDerivative.POLOIDAL:
            return jnp.sum(m * coefficients * jnp.cos(angle), axis=1)
        elif derivative == FourierDerivative.TOROIDAL:
            return jnp.sum(-n * coefficients * jnp.cos(angle), axis=1)
        else:
            return jnp.sum(coefficients * jnp.sin(angle), axis=1)


@jt.jaxtyped(typechecker=typechecker)
def interpolate_coefficients_radially(
    fourier_coefficients: jt.Float[jax.Array, "n_s n_fourier_coefficients"],
    normalized_toroidal_flux_in: jt.Float[jax.Array, " n_s "],
    normalized_effective_radius_out: jt.Float[jax.Array, " n_rho "],
) -> jt.Float[jax.Array, "n_rho n_fourier_coefficients"]:
    """Interpolate Fourier radially at the effective radius values provided."""
    return interpax.interp1d(
        normalized_effective_radius_out,
        jnp.sqrt(normalized_toroidal_flux_in),
        fourier_coefficients,
        extrap=True,
    )


@jt.jaxtyped(typechecker=typechecker)
def evaluate_rphiz_on_toroidal_grid(
    equilibrium: WoutLike,
    rho_theta_phi: jt.Float[jax.Array, "n_rho n_theta n_phi sthetaphi=3"],
) -> jt.Float[jax.Array, "n_rho n_theta n_phi rphiz=3"]:
    """Evaluate the cylindrical coordinates (r, phi, z) on a toroidal grid."""
    if equilibrium.lasym:
        raise NotImplementedError(
            "Non stellarator symmetric equilibria are not supported yet."
        )
    s = jnp.linspace(0, 1, equilibrium.ns)
    rho = rho_theta_phi[:, 0, 0, 0]
    r = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=equilibrium.rmnc,
            normalized_toroidal_flux_in=s,
            normalized_effective_radius_out=rho,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
    )
    z = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=equilibrium.zmns,
            normalized_toroidal_flux_in=s,
            normalized_effective_radius_out=rho,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.SIN,
    )
    phi = rho_theta_phi[:, :, :, 2]
    return jnp.stack([r, phi, z], axis=-1)


@jt.jaxtyped(typechecker=typechecker)
def evaluate_magnetic_field_on_toroidal_grid(
    equilibrium: WoutLike,
    rho_theta_phi: jt.Float[jax.Array, "n_rho n_theta n_phi sthetaphi=3"],
) -> jt.Float[jax.Array, "n_rho n_theta n_phi bxyz=3"]:
    """Evaluate the cartesian components of the magnetic field on a toroidal grid."""
    if equilibrium.lasym:
        raise NotImplementedError(
            "Non stellarator symmetric equilibria are not supported yet."
        )
    ds = 1 / (equilibrium.ns - 1)
    s_half = jnp.linspace(ds / 2, 1 - ds / 2, equilibrium.ns - 1)
    s_full = jnp.linspace(0, 1, equilibrium.ns)
    rho = rho_theta_phi[:, 0, 0, 0]
    b_theta = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=equilibrium.bsupumnc,
            normalized_toroidal_flux_in=s_half,
            normalized_effective_radius_out=rho,
        ),
        poloidal_mode_numbers=equilibrium.xm_nyq,
        toroidal_mode_numbers=equilibrium.xn_nyq,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
    )
    b_phi = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=equilibrium.bsupvmnc,
            normalized_toroidal_flux_in=s_half,
            normalized_effective_radius_out=rho,
        ),
        poloidal_mode_numbers=equilibrium.xm_nyq,
        toroidal_mode_numbers=equilibrium.xn_nyq,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
    )
    r = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=equilibrium.rmnc,
            normalized_toroidal_flux_in=s_full,
            normalized_effective_radius_out=rho,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
    )
    dr_dtheta = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=equilibrium.rmnc,
            normalized_toroidal_flux_in=s_full,
            normalized_effective_radius_out=rho,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
        derivative=FourierDerivative.POLOIDAL,
    )
    dz_dtheta = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=equilibrium.zmns,
            normalized_toroidal_flux_in=s_full,
            normalized_effective_radius_out=rho,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.SIN,
        derivative=FourierDerivative.POLOIDAL,
    )
    dr_dphi = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=equilibrium.rmnc,
            normalized_toroidal_flux_in=s_full,
            normalized_effective_radius_out=rho,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
        derivative=FourierDerivative.TOROIDAL,
    )
    dz_dphi = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=equilibrium.zmns,
            normalized_toroidal_flux_in=s_full,
            normalized_effective_radius_out=rho,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.SIN,
        derivative=FourierDerivative.TOROIDAL,
    )
    phi = rho_theta_phi[:, :, :, 2]
    # See eq. (3.68) of https://arxiv.org/abs/2502.04374
    dxyz_dtheta = jnp.stack(
        [dr_dtheta * jnp.cos(phi), dr_dtheta * jnp.sin(phi), dz_dtheta], axis=-1
    )
    # See eq. (3.69) of https://arxiv.org/abs/2502.04374
    dxyz_dphi = jnp.stack(
        [
            dr_dphi * jnp.cos(phi) - r * jnp.sin(phi),
            dr_dphi * jnp.sin(phi) + r * jnp.cos(phi),
            dz_dphi,
        ],
        axis=-1,
    )
    return b_theta[..., jnp.newaxis] * dxyz_dtheta + b_phi[..., jnp.newaxis] * dxyz_dphi


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
        ).reshape(n_r, n_z, 3)
        values.append(value)
    return jnp.stack(values, axis=1)
