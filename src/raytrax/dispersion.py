from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt

ScalarFloat = float | jt.Float[jax.Array, " "]


def dispersion_cold(
    refractive_index_perp: ScalarFloat,
    refractive_index_para: ScalarFloat,
    frequency: ScalarFloat,
    cyclotron_frequency: ScalarFloat,
    plasma_frequency: ScalarFloat,
    mode: Literal["X", "O"],
) -> ScalarFloat:
    """Computes the refractive index squared (N^2) from the dispersion relation of a
    cold magnetized plasma.

    Args:
        refractive_index_perp: Refractive index perpendicular to the magnetic field.
        refractive_index_para: Refractive index parallel to the magnetic field.
        frequency: Frequency of the wave (in Hertz!).
        cyclotron_frequency: Cyclotron frequency of electrons (in Hertz!).
        plasma_frequency: Plasma frequency of electrons (in Hertz!).
        mode: 'X' or 'O' for extraordinary or ordinary mode.
    """
    X = plasma_frequency**2 / frequency**2
    Y = cyclotron_frequency / frequency

    # Handle the case where refractive_index_para is 0 (perpendicular propagation)
    # When refractive_index_para is 0, the angle is 90 degrees, so sin2theta = 1
    safe_para = jnp.where(refractive_index_para == 0, 1.0, refractive_index_para)
    tantheta = refractive_index_perp / safe_para
    sin2theta = jnp.where(
        refractive_index_para == 0,
        1.0,  # sin²(90°) = 1
        tantheta**2 / (1 + tantheta**2),
    )
    return _dispersion_appleton_hartee(X=X, Y=Y, sin2theta=sin2theta, mode=mode)


def _dispersion_appleton_hartee(
    X: ScalarFloat,
    Y: ScalarFloat,
    sin2theta: ScalarFloat,
    mode: Literal["X", "O"],
) -> ScalarFloat:
    """Computes the refractive index squared (N^2) using the Appleton-Hartree formula
    valid for electromagnetic wave propagation in a cold magnetized plasma.

    See e.g. https://en.wikipedia.org/wiki/Appleton%E2%80%93Hartree_equation

    Args:
        X: (ω_pe / ω)^2, normalized plasma frequency squared.
        Y: ω_ce / ω, normalized cyclotron frequency.
        sin2theta: sine squared of the angle between wave vector and magnetic field.
        mode: 'X' or 'O' for extraordinary or ordinary mode.

    Returns:
        The refractive index squared N^2.
    """
    cos2theta = 1 - sin2theta
    X1 = 1 - X
    D = Y**4 * sin2theta**2 + 4 * X1**2 * Y**2 * cos2theta
    a = 2 * X * X1
    b = 2 * X1 - Y**2 * sin2theta

    G = jnp.sqrt(D)
    sign = -1 if mode == "X" else +1
    N_squared = 1 - a / (b + sign * G)
    return N_squared


def dispersion_tensor_stix(
    refractive_index_perp: ScalarFloat,
    refractive_index_para: ScalarFloat,
    dielectric_tensor: jt.Complex[jax.Array, "3 3"],
) -> jt.Complex[jax.Array, "3 3"]:
    r"""Computes the dispersion tensor in Stix coordinates.

    The dispersion tensor is defined as (Stix convention):

    .. math::
        D_{ij} = \epsilon_{ij} - N^2 \delta_{ij} + N_i N_j

    The refractive index has the following form in Stix coordinates:

    .. math::
        N = (N_\perp, 0, N_\parallel)

    where the magnetic field is aligned with the z-axis.

    Args:
        refractive_index_perp: Refractive index perpendicular to the magnetic field.
        refractive_index_para: Refractive index parallel to the magnetic field.
        dielectric_tensor: Dielectric tensor as a 3x3 complex array.

    Returns:
        Dispersion tensor as a 3x3 complex array.
    """
    n1 = refractive_index_perp
    n3 = refractive_index_para

    # N^2 delta_ij - Ni Nj  (note: we compute dielectric_tensor - nn = eps - N^2 I + NN)
    nn = jnp.array(
        [[n3**2, 0, -n1 * n3], [0, n1**2 + n3**2, 0], [-n1 * n3, 0, n1**2]],
        dtype=jnp.complex128,
    )
    return dielectric_tensor - nn
