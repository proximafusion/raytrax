from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from raytrax import dispersion, quantities, ray
from jaxtyping import Float


def hamiltonian(
    position: Float[jax.Array, "3"],
    refractive_index: Float[jax.Array, "3"],
    magnetic_field_interpolator: Callable[
        [Float[jax.Array, "3"]], Float[jax.Array, "3"]
    ],
    rho_interpolator: Callable[
        [Float[jax.Array, "3"]], Float[jax.Array, ""]
    ],
    electron_density_profile_interpolator: Callable[
        [Float[jax.Array, ""]], Float[jax.Array, ""]
    ],
    frequency: Float[jax.Array, ""],
    mode: Literal["X", "O"],
) -> Float[jax.Array, ""]:
    """Compute the Hamiltonian."""
    magnetic_field = magnetic_field_interpolator(position)
    rho = rho_interpolator(position)
    electron_density_1e20_per_m3 = electron_density_profile_interpolator(rho)
    # FIXME add back cold tracing
    return _hamiltonian_vacuum(refractive_index=refractive_index)
    return jax.lax.cond(
        electron_density_1e20_per_m3 < 1e6,
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


_hamiltonian_gradient_r = jax.grad(hamiltonian, argnums=0)
_hamiltonian_gradient_n = jax.grad(hamiltonian, argnums=1)


def hamiltonian_gradient_r(
    ray_state: ray.RayState,
    ray_setting: ray.RaySetting,
    magnetic_field_interpolator: Callable[
        [Float[jax.Array, "3"]], Float[jax.Array, "3"]
    ],
    rho_interpolator: Callable[
        [Float[jax.Array, "3"]], Float[jax.Array, ""]
    ],
    electron_density_profile_interpolator: Callable[
        [Float[jax.Array, ""]], Float[jax.Array, ""]
    ],
) -> Float[jax.Array, "3"]:
    r"""Compute the gradient of the Hamiltonian with respect to the position vector.

    .. math::
        \frac{\partial \mathcal{H}}{\partial \mathbf{r}}
    """
    return _hamiltonian_gradient_r(
        ray_state.position,
        ray_state.refractive_index,
        magnetic_field_interpolator,
        rho_interpolator,
        electron_density_profile_interpolator,
        ray_setting.frequency,
        ray_setting.mode,
    )


def hamiltonian_gradient_n(
    ray_state: ray.RayState,
    ray_setting: ray.RaySetting,
    magnetic_field_interpolator: Callable[
        [Float[jax.Array, "3"]], Float[jax.Array, "3"]
    ],
    rho_interpolator: Callable[
        [Float[jax.Array, "3"]], Float[jax.Array, ""]
    ],
    electron_density_profile_interpolator: Callable[
        [Float[jax.Array, ""]], Float[jax.Array, ""]
    ],
) -> Float[jax.Array, "3"]:
    r"""Compute the gradient of the Hamiltonian with respect to the refractive index
    vector.

    .. math::
        \frac{\partial \mathcal{H}}{\partial \mathbf{N}}
    """
    return _hamiltonian_gradient_n(
        ray_state.position,
        ray_state.refractive_index,
        magnetic_field_interpolator,
        rho_interpolator,
        electron_density_profile_interpolator,
        ray_setting.frequency,
        ray_setting.mode,
    )


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
