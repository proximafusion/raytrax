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
    r"""Ray-tracing Hamiltonian $\mathcal{H}(\boldsymbol{r}, \boldsymbol{n}) = n_\perp^2 - N_\perp^2(n_\parallel^2, \boldsymbol{r})$.

    Rays propagate along level sets $\mathcal{H} = 0$. In vacuum ($n_e = 0$) the
    trivial form $\mathcal{H} = |\boldsymbol{n}|^2 - 1$ is used; elsewhere
    $N_\perp^2$ is the O- or X-mode root of the cold-plasma dispersion relation at
    the parallel refractive index $n_\parallel = \boldsymbol{n} \cdot \boldsymbol{B} / |\boldsymbol{B}|$
    of the ray, see `dispersion.refractive_index_perp_squared_cold`. This form is
    smooth where rays are reflected, in particular at the O-mode cutoff, and depends
    on $n_\parallel$ only through its square.

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
    # Only at zero density, where the cold Hamiltonian is |n|^2 - 1 too, so that H is
    # continuous across the plasma boundary. Outside the grid B is zero and the cold
    # Hamiltonian undefined, but there rho, and hence the density, is beyond the ramp.
    H = jax.lax.cond(
        electron_density_1e20_per_m3 <= 0.0,
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


hamiltonian_value_and_gradients = jax.value_and_grad(
    hamiltonian, argnums=(0, 1), has_aux=True
)
r"""Compute $\mathcal{H}$ together with both gradients in a single pass.

Returns ``((H, HamiltonianAux(...)), (grad_r, grad_n))``.
"""

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
    """Compute the Hamiltonian for a cold plasma, n_perp^2 - N_perp^2(n_para^2)."""
    magnetic_field_strength = jnp.linalg.norm(magnetic_field)
    tangent = magnetic_field / magnetic_field_strength
    refractive_index_para = jnp.dot(refractive_index, tangent)
    # Squares only: the norm of the perpendicular component is not differentiable
    # where it vanishes, which is where rays are reflected.
    refractive_index_perp_squared = (
        jnp.dot(refractive_index, refractive_index) - refractive_index_para**2
    )
    cyclotron_frequency = quantities.electron_cyclotron_frequency(
        magnetic_field_strength=magnetic_field_strength
    )
    plasma_frequency = quantities.electron_plasma_frequency(
        electron_density_1e20_per_m3=electron_density_1e20_per_m3
    )
    n_perp2_dispersion = dispersion.refractive_index_perp_squared_cold(
        refractive_index_para_squared=refractive_index_para**2,
        X=(plasma_frequency / frequency) ** 2,
        Y=cyclotron_frequency / frequency,
        mode=mode,
    )
    return jnp.squeeze(refractive_index_perp_squared - n_perp2_dispersion)
