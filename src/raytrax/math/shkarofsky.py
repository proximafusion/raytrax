"""Shkarofsky integral functions F_{q+1/2}(s) for the warm plasma dielectric tensor."""

import jax
import jax.numpy as jnp

from raytrax.math.faddeeva import plasma_dispersion_function as Z
from raytrax.math.faddeeva import (
    plasma_dispersion_function_derivative as Z_prime,
)
from raytrax.types import ScalarFloat

_PSI_TOLERANCE = 1e-6  # Tolerance for checking if psi is zero


def shkarofsky(
    z: ScalarFloat,
    mu: ScalarFloat,
    n_par: ScalarFloat,
    w: ScalarFloat,
    w_c: ScalarFloat,
    q: ScalarFloat,
):
    r"""Compute the Shkarofsky function :math:`\mathcal{F}_{q+1/2}(s)`

    The function is defined as

    .. math::

        \mathcal{F}_{q+\tfrac{1}{2}}(s) = \left( \frac{\mu}{2\pi} \right)^{1/2}
        \int_{-\infty}^{+\infty} d\rho \, \exp\left( -\tfrac{1}{2} \mu \rho^2 \right)
        F_q(\mu \beta_s),

    (see section 3 of Krivensky and Orefice, J. Plasma Physics (1983), vol. 30, part 1,
    pp. 125-131).
    """
    psi = jnp.asarray(n_par * jnp.sqrt(mu / 2), jnp.complex128)
    alpha_s = jnp.asarray(n_par**2 / 2 - (1 - z * w_c / w), jnp.complex128)
    phi = jnp.sqrt(mu * alpha_s)
    phi = jnp.where(alpha_s < 0, -1j * phi, phi)
    return _shkarofsky_impl(psi, phi, q)


def _shkarofsky_impl(psi: ScalarFloat, phi: ScalarFloat, q: ScalarFloat) -> ScalarFloat:
    """Implementation of the Shkarofsky function with direct parameter inputs.

    Args:
        psi: Calculated parameter from parent function
        phi: Calculated parameter from parent function
        q: Order of the Shkarofsky function

    Returns:
        Value of the Shkarofsky function
    """
    return jax.lax.cond(
        # special handling required for psi=0
        psi < _PSI_TOLERANCE,
        lambda: _shkarofsky_impl_psi_zero(phi, q),
        lambda: _shkarofsky_impl_psi_nonzero(psi, phi, q),
    )


def _shkarofsky_impl_psi_nonzero(
    psi: ScalarFloat, phi: ScalarFloat, q: ScalarFloat
) -> ScalarFloat:
    """Implementation of the Shkarofsky function with direct parameter inputs, assuming
    psi is non-zero.

    Args:
        psi: Calculated parameter from parent function
        phi: Calculated parameter from parent function
        q: Order of the Shkarofsky function

    Returns:
        Value of the Shkarofsky function
    """
    if q < 0:
        return jnp.ones_like(psi) * jnp.nan  # Return NaN for invalid q values
    if q == 0:
        # F_1/2, eq. 29 of Krivensky and Orefice
        return -(Z(psi - phi) + Z(-psi - phi)) / 2 / phi
    if q == 1:  # F_3/2
        # F_3/2, eq. 30 of Krivensky and Orefice
        return -(Z(psi - phi) - Z(-psi - phi)) / 2 / psi
    else:
        # eq. 26 of Krivensky and Orefice
        # F_{q + 1/2}=F_{p + 5/2} for p = q - 2
        p = q - 2
        f_0 = _shkarofsky_impl_psi_nonzero(psi, phi, p)  # F_{p + 1/2}
        f_1 = _shkarofsky_impl_psi_nonzero(psi, phi, p + 1)  # F_{p + 3/2}
        return (1 + phi**2 * f_0 - (p + 0.5) * f_1) / psi**2


def _shkarofsky_impl_psi_zero(phi: ScalarFloat, q: ScalarFloat) -> ScalarFloat:
    """Implementation of the Shkarofsky function for the limit psi=0.

    Args:
        phi: Calculated parameter from parent function
        q: Order of the Shkarofsky function

    Returns:
        Value of the Shkarofsky function
    """
    if q < 0:
        return jnp.ones_like(phi) * jnp.nan  # Return NaN for invalid q values
    if q == 0:
        # F_1/2, eq. 31 of Krivensky and Orefice
        return -Z(-phi) / phi
    if q == 1:  # F_3/2
        # F_3/2, eq. 30 of Krivensky and Orefice
        # UP TO A SIGN - there seems to be a sign error in the original paper
        return -Z_prime(-phi)
    else:
        # For psi->0, we can rewrite eq. 26, setting the term in brackets to zero
        # F_{p + 3/2} = (phi^2 * F_{p + 1/2} + 1 ) / (p + 0.5)
        f_0 = _shkarofsky_impl_psi_zero(phi, q - 1)
        return (phi**2 * f_0 + 1.0) / (q - 0.5)
