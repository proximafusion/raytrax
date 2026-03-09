"""Shkarofsky integral functions F_{q+1/2}(s) for the warm plasma dielectric tensor."""

import jax.numpy as jnp

from raytrax.math.faddeeva import plasma_dispersion_function as Z
from raytrax.math.faddeeva import (
    plasma_dispersion_function_derivative as Z_prime,
)
from raytrax.types import ScalarFloat

_PSI_TOLERANCE = 1e-6  # Tolerance for checking if psi is zero


def _shkarofsky_sequence(
    psi: ScalarFloat,
    phi: ScalarFloat,
    q_max: int,
    *,
    force_psi_zero: bool | None = None,
) -> list:
    """Compute F_{q+1/2} for q=0..q_max with a single iterative recurrence pass."""
    if q_max < 0:
        return []

    if force_psi_zero is None:
        is_psi_zero = jnp.abs(psi) < _PSI_TOLERANCE

        # F_{1/2} and F_{3/2}, eqs. 29-30 of Krivensky and Orefice (psi != 0).
        z_plus = Z(psi - phi)
        z_minus = Z(-psi - phi)
        f0_nonzero = -(z_plus + z_minus) / (2 * phi)
        f1_nonzero = -(z_plus - z_minus) / (2 * psi)

        # F_{1/2}, eq. 31. For F_{3/2} in the psi -> 0 limit we use -Z'(-phi);
        # the original paper appears to have a sign typo for this term.
        f0_zero = -Z(-phi) / phi
        f1_zero = -Z_prime(-phi)
        f0 = jnp.where(is_psi_zero, f0_zero, f0_nonzero)
        f1 = jnp.where(is_psi_zero, f1_zero, f1_nonzero)
    elif force_psi_zero:
        is_psi_zero = True
        f0 = -Z(-phi) / phi
        f1 = -Z_prime(-phi)
    else:
        is_psi_zero = False
        z_plus = Z(psi - phi)
        z_minus = Z(-psi - phi)
        f0 = -(z_plus + z_minus) / (2 * phi)
        f1 = -(z_plus - z_minus) / (2 * psi)

    if q_max == 0:
        return [f0]
    if q_max == 1:
        return [f0, f1]

    results = [f0, f1]
    f_prev2 = f0
    f_prev1 = f1

    for q in range(2, q_max + 1):
        # Upward recurrence, eq. 26, psi != 0 branch.
        p = q - 2
        f_nonzero = (1 + phi**2 * f_prev2 - (p + 0.5) * f_prev1) / (psi**2)
        # psi -> 0 limit of eq. 26:
        # F_{q+1/2} = (phi^2 * F_{q-1/2} + 1) / (q - 1/2)
        f_zero = (phi**2 * f_prev1 + 1.0) / (q - 0.5)
        if force_psi_zero is None:
            f_q = jnp.where(is_psi_zero, f_zero, f_nonzero)
        elif is_psi_zero:
            f_q = f_zero
        else:
            f_q = f_nonzero

        results.append(f_q)
        f_prev2 = f_prev1
        f_prev1 = f_q

    return results


def shkarofsky(
    z: ScalarFloat,
    mu: ScalarFloat,
    n_par: ScalarFloat,
    w: ScalarFloat,
    w_c: ScalarFloat,
    q_max: int,
) -> list:
    r"""Compute Shkarofsky functions :math:`\mathcal{F}_{q+1/2}(s)` for q = 0, 1, ..., q_max.

    Uses iterative upward recurrence from the two base cases (q=0, q=1),
    requiring only 2 plasma dispersion function evaluations (for psi != 0)
    or 1 Z + 1 Z' (for psi = 0), regardless of q_max.

    The function is defined as

    .. math::

        \mathcal{F}_{q+\tfrac{1}{2}}(s) = \left( \frac{\mu}{2\pi} \right)^{1/2}
        \int_{-\infty}^{+\infty} d\rho \, \exp\left( -\tfrac{1}{2} \mu \rho^2 \right)
        F_q(\mu \beta_s),

    (see section 3 of Krivensky and Orefice, J. Plasma Physics (1983), vol. 30, part 1,
    pp. 125-131).

    Args:
        z: Harmonic index parameter (integer, can be negative).
        mu: Inverse normalized temperature, mu = 2 / v_th^2.
        n_par: Parallel refractive index.
        w: Angular wave frequency (2*pi*f).
        w_c: Angular cyclotron frequency.
        q_max: Maximum q value (returns q = 0, 1, ..., q_max).

    Returns:
        List of length q_max + 1, where element i is F_{i+1/2}(z).
    """
    psi = jnp.asarray(n_par * jnp.sqrt(mu / 2), jnp.complex128)
    alpha_s = jnp.asarray(n_par**2 / 2 - (1 - z * w_c / w), jnp.complex128)
    phi = jnp.sqrt(mu * alpha_s)
    phi = jnp.where(alpha_s < 0, -1j * phi, phi)

    return _shkarofsky_sequence(psi, phi, q_max)
