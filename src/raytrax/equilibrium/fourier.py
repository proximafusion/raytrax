"""VMEC Fourier-series evaluation: magnetic field and geometry on toroidal grids, and dV/dρ computation."""

from enum import Enum
from functools import partial

import interpax
import jax
import jax.numpy as jnp
import jaxtyping as jt
from beartype import beartype as typechecker

from raytrax.equilibrium.protocol import WoutLike


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
    fourier_coefficients: jt.Float[jax.Array, "n_fourier_coefficients n_rho"],
    poloidal_mode_numbers: jt.Int[jax.Array, " n_fourier_coefficients"],
    toroidal_mode_numbers: jt.Int[jax.Array, " n_fourier_coefficients"],
    rho_theta_phi: jt.Float[jax.Array, "n_rho n_theta n_phi rho_theta_phi=3"],
    basis: FourierBasis,
    derivative: FourierDerivative = FourierDerivative.NO,
) -> jt.Float[jax.Array, "n_rho n_theta n_phi"]:
    """Transform an array of Fourier coefficients into values on a toroidal grid."""
    m = poloidal_mode_numbers[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    n = toroidal_mode_numbers[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    theta = rho_theta_phi[jnp.newaxis, :, :, :, 1]
    phi = rho_theta_phi[jnp.newaxis, :, :, :, 2]
    angle = m * theta - n * phi
    coefficients = fourier_coefficients[:, :, jnp.newaxis, jnp.newaxis]
    if basis == FourierBasis.COS:
        if derivative == FourierDerivative.POLOIDAL:
            return jnp.sum(-m * coefficients * jnp.sin(angle), axis=0)
        elif derivative == FourierDerivative.TOROIDAL:
            return jnp.sum(n * coefficients * jnp.sin(angle), axis=0)
        else:
            return jnp.sum(coefficients * jnp.cos(angle), axis=0)
    elif basis == FourierBasis.SIN:
        if derivative == FourierDerivative.POLOIDAL:
            return jnp.sum(m * coefficients * jnp.cos(angle), axis=0)
        elif derivative == FourierDerivative.TOROIDAL:
            return jnp.sum(-n * coefficients * jnp.cos(angle), axis=0)
        else:
            return jnp.sum(coefficients * jnp.sin(angle), axis=0)


@jt.jaxtyped(typechecker=typechecker)
def interpolate_coefficients_radially(
    fourier_coefficients: jt.Float[jax.Array, "n_fourier_coefficients n_s"],
    normalized_toroidal_flux_in: jt.Float[jax.Array, " n_s "],
    normalized_effective_radius_out: jt.Float[jax.Array, " n_rho "],
) -> jt.Float[jax.Array, "n_fourier_coefficients n_rho"]:
    """Interpolate Fourier radially at the effective radius values provided."""
    return interpax.interp1d(
        normalized_effective_radius_out,
        jnp.sqrt(normalized_toroidal_flux_in),
        fourier_coefficients.T,
        extrap=True,
    ).T


@jt.jaxtyped(typechecker=typechecker)
def evaluate_rphiz_on_toroidal_grid(
    equilibrium: WoutLike,
    rho_theta_phi: jt.Float[jax.Array, "n_rho n_theta n_phi sthetaphi=3"],
) -> jt.Float[jax.Array, "n_rho n_theta n_phi rphiz=3"]:
    """Evaluate the cylindrical coordinates (r, phi, z) on a toroidal grid.

    For rho > 1, the flux surface shape is linearly extrapolated outward to extend
    the physical domain beyond the LCMS.
    """
    if equilibrium.lasym:
        raise NotImplementedError(
            "Non stellarator symmetric equilibria are not supported yet."
        )
    s = jnp.linspace(0, 1, equilibrium.ns)
    rho = rho_theta_phi[:, 0, 0, 0]

    # For rho <= 1: evaluate normally
    # For rho > 1: evaluate at rho=1 and extrapolate linearly outward
    rho_eval = jnp.clip(rho, 0.0, 1.0)

    r_at_boundary = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=jnp.asarray(equilibrium.rmnc),
            normalized_toroidal_flux_in=s,
            normalized_effective_radius_out=rho_eval,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
    )
    z_at_boundary = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=jnp.asarray(equilibrium.zmns),
            normalized_toroidal_flux_in=s,
            normalized_effective_radius_out=rho_eval,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.SIN,
    )

    # For rho > 1, extrapolate by expanding radially from magnetic axis
    # Get magnetic axis position (R0, Z0) - evaluate at rho=0 for first radial point
    rho_axis = jnp.zeros_like(rho_eval)  # All zeros to evaluate at axis
    r_axis = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=jnp.asarray(equilibrium.rmnc),
            normalized_toroidal_flux_in=s,
            normalized_effective_radius_out=rho_axis,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
    )
    z_axis = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=jnp.asarray(equilibrium.zmns),
            normalized_toroidal_flux_in=s,
            normalized_effective_radius_out=rho_axis,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.SIN,
    )

    # Linear extrapolation: for rho > 1, expand the LCMS surface radially from axis
    # r(rho) = r_axis + rho * (r_boundary - r_axis) where r_boundary is at rho_eval=1
    extrapolation_factor = rho[:, jnp.newaxis, jnp.newaxis] / jnp.clip(
        rho_eval[:, jnp.newaxis, jnp.newaxis], 1e-10, None
    )
    r = r_axis + extrapolation_factor * (r_at_boundary - r_axis)
    z = z_axis + extrapolation_factor * (z_at_boundary - z_axis)

    phi = rho_theta_phi[:, :, :, 2]
    return jnp.stack([r, phi, z], axis=-1)


@jt.jaxtyped(typechecker=typechecker)
def evaluate_magnetic_field_on_toroidal_grid(
    equilibrium: WoutLike,
    rho_theta_phi: jt.Float[jax.Array, "n_rho n_theta n_phi sthetaphi=3"],
) -> jt.Float[jax.Array, "n_rho n_theta n_phi bxyz=3"]:
    """Evaluate the cartesian components of the magnetic field on a toroidal grid.

    For rho > 1, the magnetic field is extrapolated by holding it constant at the
    rho=1 value to provide smooth behavior at the plasma boundary.
    """
    if equilibrium.lasym:
        raise NotImplementedError(
            "Non stellarator symmetric equilibria are not supported yet."
        )
    ds = 1 / (equilibrium.ns - 1)
    s_half = jnp.linspace(ds / 2, 1 - ds / 2, equilibrium.ns - 1)
    s_full = jnp.linspace(0, 1, equilibrium.ns)
    rho = rho_theta_phi[:, 0, 0, 0]

    # Clamp rho to [0, 1] for Fourier evaluation (will extrapolate for rho > 1 later)
    rho_clamped = jnp.clip(rho, 0.0, 1.0)

    b_theta = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=jnp.asarray(equilibrium.bsupumnc[:, 1:]),
            normalized_toroidal_flux_in=s_half,
            normalized_effective_radius_out=rho_clamped,
        ),
        poloidal_mode_numbers=equilibrium.xm_nyq,
        toroidal_mode_numbers=equilibrium.xn_nyq,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
    )
    b_phi = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=jnp.asarray(equilibrium.bsupvmnc[:, 1:]),
            normalized_toroidal_flux_in=s_half,
            normalized_effective_radius_out=rho_clamped,
        ),
        poloidal_mode_numbers=equilibrium.xm_nyq,
        toroidal_mode_numbers=equilibrium.xn_nyq,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
    )
    r = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=jnp.asarray(equilibrium.rmnc),
            normalized_toroidal_flux_in=s_full,
            normalized_effective_radius_out=rho_clamped,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
    )
    dr_dtheta = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=jnp.asarray(equilibrium.rmnc),
            normalized_toroidal_flux_in=s_full,
            normalized_effective_radius_out=rho_clamped,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
        derivative=FourierDerivative.POLOIDAL,
    )
    dz_dtheta = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=jnp.asarray(equilibrium.zmns),
            normalized_toroidal_flux_in=s_full,
            normalized_effective_radius_out=rho_clamped,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.SIN,
        derivative=FourierDerivative.POLOIDAL,
    )
    dr_dphi = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=jnp.asarray(equilibrium.rmnc),
            normalized_toroidal_flux_in=s_full,
            normalized_effective_radius_out=rho_clamped,
        ),
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        rho_theta_phi=rho_theta_phi,
        basis=FourierBasis.COS,
        derivative=FourierDerivative.TOROIDAL,
    )
    dz_dphi = inverse_fourier_transform(
        fourier_coefficients=interpolate_coefficients_radially(
            fourier_coefficients=jnp.asarray(equilibrium.zmns),
            normalized_toroidal_flux_in=s_full,
            normalized_effective_radius_out=rho_clamped,
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
def dvolume_drho(
    equilibrium: WoutLike, rho: jt.Float[jax.Array, " n_rho"]
) -> jt.Float[jax.Array, " n_rho"]:
    """Compute the derivative of the volume with respect to the effective radius."""
    s_full = jnp.linspace(0, 1, equilibrium.ns)
    g00_coefficients = jnp.asarray(equilibrium.gmnc)[0:1, :]  # Shape: (1, ns)
    g00_interpolated = interpolate_coefficients_radially(
        fourier_coefficients=g00_coefficients,
        normalized_toroidal_flux_in=s_full,
        normalized_effective_radius_out=rho,
    )
    g00 = g00_interpolated[0, :]
    # dV/ds = (2π)² g00 where s = ρ² (normalised toroidal flux).
    # By the chain rule dV/dρ = dV/ds × ds/dρ = (2π)² g00 × 2ρ.
    return (2 * jnp.pi) ** 2 * jnp.abs(g00) * 2 * rho
