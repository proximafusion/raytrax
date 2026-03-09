"""Wave polarization vector for O- and X-mode in Stix coordinates."""

from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt

from raytrax.physics import dispersion
from raytrax.types import ScalarFloat

ComplexFloat = complex | jt.Complex[jax.Array, " "]


def polarization(
    dielectric_tensor: jt.Complex[jax.Array, "3 3"],
    refractive_index_perp: ScalarFloat,
    refractive_index_para: ScalarFloat,
    frequency: ScalarFloat,
    cyclotron_frequency: ScalarFloat,
    mode: Literal["X", "O"],
) -> jt.Complex[jax.Array, "3"]:
    """Computes the normalized, dimensionless polarization vector in Stix coordinates
    given a dielectric tensor.

    Args:
        dielectric_tensor: Dielectric tensor as a 3x3 complex array.
        refractive_index_perp: Refractive index perpendicular to the magnetic field.
        refractive_index_para: Refractive index parallel to the magnetic field.
        frequency: Frequency of the wave.
        cyclotron_frequency: Electron cyclotron frequency.
        mode: Polarization mode, either "X" for extraordinary or "O" for ordinary.

    Returns:
        Polarization vector as a 3-element complex array.
    """
    D = dispersion.dispersion_tensor_stix(
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        dielectric_tensor=dielectric_tensor,
    )

    if mode == "O":
        p1 = D[1, 1] * D[0, 2] - D[0, 1] * D[1, 2]
        p2 = -(D[0, 0] * D[1, 2] + D[0, 2] * D[0, 1])
        p3 = -(D[0, 0] * D[1, 1] + D[0, 1] * D[0, 1])
    elif mode == "X":
        p1 = D[2, 2] * D[0, 1] + D[1, 2] * D[0, 2]
        p2 = D[2, 2] * D[0, 0] - D[0, 2] * D[0, 2]
        p3 = -(D[0, 1] * D[0, 2] + D[1, 2] * D[0, 0])
    else:
        raise ValueError(f"Mode must be either 'X' or 'O', got {mode}.")

    p = jnp.array([p1, p2, p3], dtype=complex)
    p = p / jnp.linalg.norm(p)

    p = jax.lax.cond(frequency < cyclotron_frequency, lambda p: -p, lambda p: p, p)

    return p


def _polarization_low_density(
    refractive_index_para: ScalarFloat,
    frequency: ScalarFloat,
    plasma_frequency: ScalarFloat,
    cyclotron_frequency: ScalarFloat,
    mode: Literal["X", "O"],
) -> jt.Complex[jnp.ndarray, "3"]:
    """Computes the normalized polarization vector in the cold, low-density plasma
    limit.

    This reproduces the behavior of the Fortran routine WavePolar_LowDensLimit.

    Args:
        refractive_index_perp: N_perp component (N1).
        refractive_index_para: N_parallel component (n3).
        frequency: Wave frequency ω.
        plasma_frequency: Electron plasma frequency ω_pe.
        cyclotron_frequency: Electron cyclotron frequency ω_ce.
        mode: Wave mode: 'X' (extraordinary) or 'O' (ordinary).

    Returns:
        Normalized complex polarization vector [Ex, Ey, Ez].
    """

    n3 = refractive_index_para
    X = (plasma_frequency / frequency) ** 2
    # negative sign by convention
    Ys = -jnp.abs(cyclotron_frequency / frequency)
    Y2 = Ys**2
    Y21 = 1 - Y2
    n32 = n3**2
    N120 = 1 - n32
    n1 = jnp.sqrt(N120)

    if mode == "X":
        F = 0.5 * Y2 * N120 * (1 - jnp.sqrt(1 + 4 * n32 / (Y2 * N120**2)))
        F1 = F - N120 * Y2
        Ey: ComplexFloat = 1.0 + 0j
        Ex: ComplexFloat = (
            -1j * Ys / F1 * (n32 + X * (Y2 - F) * (F1 - n32) / (Y21 * F1))
        )
        Ez: ComplexFloat = (
            1j
            * Ys
            * n3
            * n1
            / F1
            * (1 + X / (N120 * Y21) * ((1 + F) / 2 - n32 * F / F1))
        )

    elif mode == "O":
        F = 0.5 * Y2 * N120 * (1 + jnp.sqrt(1 + 4 * n32 / (Y2 * N120**2)))
        Ez = 1.0 + 0j
        Ex = -n3 / n1 * (1 + X * (1 + F - 2 * Y2 / F) / (2 * N120 * Y21))
        Ey = 1j * Ys / F * Ex

    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'X' or 'O'.")

    p = jnp.array([Ex, Ey, Ez])
    p = p / jnp.linalg.norm(p)

    return p
