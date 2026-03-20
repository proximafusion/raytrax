"""Wave polarization vector for O- and X-mode in Stix coordinates."""

from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt

from raytrax.physics import dispersion
from raytrax.types import ScalarFloat


def polarization(
    dielectric_tensor: jt.Complex[jax.Array, "3 3"],
    refractive_index_perp: ScalarFloat,
    refractive_index_para: ScalarFloat,
    frequency: ScalarFloat,
    cyclotron_frequency: ScalarFloat,
    mode: Literal["X", "O"],
) -> jt.Complex[jax.Array, "3"]:
    """Compute the normalized polarization vector in Stix coordinates.

    Solves the 2x2 subsystem of the dispersion tensor: for O-mode, Ez is
    fixed to 1 and (Ex, Ey) are determined; for X-mode, Ey is fixed to 1
    and (Ex, Ez) are determined. The result is normalised to unit length.

    Args:
        dielectric_tensor: Dielectric tensor as a 3x3 complex array.
        refractive_index_perp: Refractive index perpendicular to the magnetic field.
        refractive_index_para: Refractive index parallel to the magnetic field.
        frequency: Wave frequency (unused, kept for API compatibility).
        cyclotron_frequency: Electron cyclotron frequency (unused, kept for API compatibility).
        mode: Polarization mode, either "X" for extraordinary or "O" for ordinary.

    Returns:
        Normalized complex polarization vector [Ex, Ey, Ez].
    """
    if mode not in ("X", "O"):
        raise ValueError(f"mode must be 'X' or 'O', got {mode!r}.")

    D = dispersion.dispersion_tensor_stix(
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        dielectric_tensor=dielectric_tensor,
    )

    def solve_o_mode():
        denom = (-D[0, 0]) * (-D[1, 1]) + D[0, 1] ** 2
        ex = -((-D[1, 1]) * (-D[0, 2]) - D[0, 1] * D[1, 2]) / denom
        ey = ((-D[0, 0]) * D[1, 2] + (-D[0, 2]) * D[0, 1]) / denom
        return jnp.array([ex, ey, 1.0 + 0.0j], dtype=jnp.complex128)

    def solve_x_mode():
        denom = (-D[0, 0]) * (-D[2, 2]) - (-D[0, 2]) ** 2
        ex = ((-D[2, 2]) * D[0, 1] + D[1, 2] * (-D[0, 2])) / denom
        ez = -(D[0, 1] * (-D[0, 2]) + D[1, 2] * (-D[0, 0])) / denom
        return jnp.array([ex, 1.0 + 0.0j, ez], dtype=jnp.complex128)

    p_unnorm = jax.lax.cond(mode == "O", solve_o_mode, solve_x_mode)

    return p_unnorm / jnp.linalg.norm(p_unnorm)
