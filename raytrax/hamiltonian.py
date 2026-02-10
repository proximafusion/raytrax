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
    return jax.lax.cond(
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


_hamiltonian_gradients_rn = jax.grad(hamiltonian, argnums=(0, 1))


def hamiltonian_gradients(
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
) -> tuple[Float[jax.Array, "3"], Float[jax.Array, "3"]]:
    r"""Compute both Hamiltonian gradients in a single backward pass.

    Returns (∂H/∂r, ∂H/∂N) from one shared forward+backward pass through the
    Hamiltonian, halving the number of B-interpolator evaluations compared to
    computing each gradient separately.

    Returns:
        Tuple of (grad_r, grad_n) where grad_r = ∂H/∂r and grad_n = ∂H/∂N.
    """
    return _hamiltonian_gradients_rn(
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
