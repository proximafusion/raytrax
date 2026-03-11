"""Cold plasma ray-tracing Hamiltonian and its gradients ∂H/∂r, ∂H/∂N."""

from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Float

from raytrax.physics import dispersion, quantities


class HamiltonianAux(NamedTuple):
    """Auxiliary quantities computed alongside the Hamiltonian value."""

    magnetic_field: Float[jax.Array, "3"]
    rho: Float[jax.Array, ""]
    electron_density_1e20_per_m3: Float[jax.Array, ""]


def hamiltonian(
    position: Float[jax.Array, "3"],
    refractive_index: Float[jax.Array, "3"],
    magnetic_field_interpolator: Callable[
        [Float[jax.Array, "3"]], Float[jax.Array, "3"]
    ],
    rho_interpolator: Callable[[Float[jax.Array, "3"]], Float[jax.Array, ""]],
    electron_density_profile_interpolator: Callable[
        [Float[jax.Array, ""]], Float[jax.Array, ""]
    ],
    frequency: Float[jax.Array, ""],
    mode: Literal["X", "O"],
) -> tuple[
    Float[jax.Array, ""],
    HamiltonianAux,
]:
    r"""Ray-tracing Hamiltonian $\mathcal{H}(\boldsymbol{r}, \boldsymbol{n}) = |\boldsymbol{n}|^2 - n_\mathrm{AH}^2(\boldsymbol{r}, \boldsymbol{n})$.

    Rays propagate along level sets $\mathcal{H} = 0$. In vacuum
    ($n_e < 10^{-6} \times 10^{20}\,\mathrm{m}^{-3}$) the trivial form
    $\mathcal{H} = |\boldsymbol{n}|^2 - 1$ is used; elsewhere the cold-plasma
    Appleton-Hartree dispersion relation selects the O- or X-mode refractive index squared.

    Args:
        position: Cartesian position vector $\boldsymbol{r}$ in metres.
        refractive_index: Refractive index vector $\boldsymbol{n} = c\boldsymbol{k}/\omega$.
        magnetic_field_interpolator: Callable mapping a Cartesian position to the
            magnetic field vector $\boldsymbol{B}$ in Tesla.
        rho_interpolator: Callable mapping a Cartesian position to the normalised
            effective radius $\rho \in [0, 1]$.
        electron_density_profile_interpolator: Callable mapping $\rho$ to
            the electron density in $10^{20}\,\mathrm{m}^{-3}$.
        frequency: Wave frequency $f$ in Hz.
        mode: Polarisation mode — `"O"` (ordinary) or `"X"` (extraordinary).

    Returns:
        Scalar value of the Hamiltonian. Zero on the dispersion surface.
    """
    magnetic_field = magnetic_field_interpolator(position)
    rho = rho_interpolator(position)
    electron_density_1e20_per_m3 = electron_density_profile_interpolator(rho)
    H = jax.lax.cond(
        electron_density_1e20_per_m3 < 1e-6,
        lambda: _hamiltonian_vacuum(
            refractive_index=refractive_index,
        ),
        lambda: _hamiltonian_cold(
            refractive_index=refractive_index,
            magnetic_field=magnetic_field,
            electron_density_1e20_per_m3=electron_density_1e20_per_m3,
            frequency=frequency,
            mode=mode,
        ),
    )
    return H, HamiltonianAux(
        magnetic_field=magnetic_field,
        rho=rho,
        electron_density_1e20_per_m3=electron_density_1e20_per_m3,
    )


hamiltonian_gradients = jax.grad(hamiltonian, argnums=(0, 1), has_aux=True)
r"""Compute both Hamiltonian gradients $(\partial \mathcal{H}/\partial \boldsymbol{r},\, \partial \mathcal{H}/\partial \boldsymbol{n})$ in a single backward pass.

Signature mirrors `hamiltonian`: `(position, refractive_index,
magnetic_field_interpolator, rho_interpolator,
electron_density_profile_interpolator, frequency, mode)`.

Returns a tuple ``((grad_r, grad_n), HamiltonianAux(...))``
where $\partial \mathcal{H}/\partial \boldsymbol{r}$
and $\partial \mathcal{H}/\partial \boldsymbol{n}$ are computed in one shared
forward+backward pass, halving the number of B-interpolator evaluations compared
to computing each gradient separately.
The internally computed quantities (`magnetic_field`, `rho`,
`electron_density_1e20_per_m3`) are
returned as aux data, which is not differentiated.
"""


def _hamiltonian_vacuum(
    refractive_index: Float[jax.Array, "3"],
) -> Float[jax.Array, ""]:
    """Compute the (trivial) Hamiltonian in vacuum."""
    return jnp.linalg.norm(refractive_index) ** 2 - 1


def _hamiltonian_cold(
    refractive_index: Float[jax.Array, "3"],
    magnetic_field: Float[jax.Array, "3"],
    electron_density_1e20_per_m3: Float[jax.Array, ""],
    frequency: Float[jax.Array, ""],
    mode: Literal["X", "O"],
) -> Float[jax.Array, ""]:
    """Compute the Hamiltonian for a cold plasma."""
    abs_n = jnp.linalg.norm(refractive_index)
    # component parallel to the B field
    tangent = magnetic_field / jnp.linalg.norm(magnetic_field)
    refractive_index_para = jnp.dot(refractive_index, tangent)
    # component perpendicular to the B field
    refractive_index_perp = jnp.linalg.norm(
        refractive_index - refractive_index_para * tangent
    )
    cyclotron_frequency = quantities.electron_cyclotron_frequency(
        magnetic_field_strength=jnp.linalg.norm(magnetic_field)
    )
    plasma_frequency = quantities.electron_plasma_frequency(
        electron_density_1e20_per_m3=electron_density_1e20_per_m3
    )
    n2_dispersion = dispersion.dispersion_cold(
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        frequency=frequency,
        cyclotron_frequency=cyclotron_frequency,
        plasma_frequency=plasma_frequency,
        mode=mode,
    )
    result = abs_n**2 - jnp.real(n2_dispersion)
    # return a scalar
    return jnp.squeeze(result)
