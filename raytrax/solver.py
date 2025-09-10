from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping as jt
from raytrax import absorption, hamiltonian, ray
from scipy import integrate


def _y_to_state(
    y: jt.Float[jax.Array, " n "],
    s: float,
) -> ray.RayState:
    return ray.RayState(
        position=y[:3],
        refractive_index=y[3:6],
        optical_depth=y[6],
        arc_length=jnp.array(s),
    )


def _state_to_y(state: ray.RayState) -> jt.Float[jax.Array, " n "]:
    return jnp.concatenate(
        [
            state.position,
            state.refractive_index,
            jnp.array([state.optical_depth]),
        ]
    )


def _term_to_dydt(term: ray.Term) -> jt.Float[jax.Array, " n "]:
    return jnp.concatenate(
        [
            term.position,
            term.refractive_index,
            jnp.array([term.optical_depth]),
        ]
    )


def _right_hand_side(
    s: float,
    y: jt.Float[jax.Array, " n "],
    setting: ray.RaySetting,
    magnetic_field_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]
    ],
    electron_density_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]
    ],
    electron_temperature_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]
    ],
) -> jt.Float[jax.Array, " n "]:
    r"""Compute the right-hand side of the differential equation.

    .. math::
        \frac{d y}{d s} = f(s, y)

    encoding the ray tracing equations

    .. math::
        \frac{d\mathbf{r}}{ds} = \frac{\partial \mathcal{H}}{\partial \mathbf{N}} \cdot \left| \frac{\partial \mathcal{H}}{\partial \mathbf{N}} \right|^{-1}
    .. math::
        \frac{d\mathbf{N}}{ds} = -\frac{\partial \mathcal{H}}{\partial \mathbf{r}} \cdot \left| \frac{\partial \mathcal{H}}{\partial \mathbf{N}} \right|^{-1}
    .. math::
        \frac{d\tau}{ds} = \alpha
    """  # noqa: E501
    state = _y_to_state(y, s=s)
    hamiltonian_gradient_n = hamiltonian.hamiltonian_gradient_n(
        state, setting, magnetic_field_interpolator, electron_density_interpolator
    )
    hamiltonian_gradient_r = hamiltonian.hamiltonian_gradient_r(
        state, setting, magnetic_field_interpolator, electron_density_interpolator
    )
    norm = jnp.linalg.norm(hamiltonian_gradient_n)
    absorption_coefficient = absorption.absorption_coefficient_conditional(
        refractive_index=state.refractive_index,
        magnetic_field=magnetic_field_interpolator(state.position),
        electron_density_1e20_per_m3=electron_density_interpolator(state.position),
        electron_temperature_keV=electron_temperature_interpolator(state.position),
        frequency=setting.frequency,
        mode=setting.mode,
    )
    right_hand_side = ray.Term(
        position=hamiltonian_gradient_n / norm,
        refractive_index=-hamiltonian_gradient_r / norm,
        optical_depth=jnp.asarray(absorption_coefficient),
    )
    return _term_to_dydt(right_hand_side)


def solve(
    state: ray.RayState,
    setting: ray.RaySetting,
    magnetic_field_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]
    ],
    electron_density_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]
    ],
    electron_temperature_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]
    ],
) -> list[ray.RayState]:
    """Solve the ray tracing equations."""
    fun = partial(
        _right_hand_side,
        setting=setting,
        magnetic_field_interpolator=magnetic_field_interpolator,
        electron_density_interpolator=electron_density_interpolator,
        electron_temperature_interpolator=electron_temperature_interpolator,
    )
    y0 = _state_to_y(state)
    jitted_fun = jax.jit(fun)
    res = integrate.solve_ivp(
        fun=jitted_fun,
        t_span=(0, 0.6),
        y0=y0,
        max_step=0.005,
        first_step=0.005,
    )
    return [_y_to_state(y=jnp.asarray(y), s=t) for t, y in zip(res.t, res.y.T)]


def compute_additional_quantities(
    ray_states: list[ray.RayState],
    setting: ray.RaySetting,
    magnetic_field_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]
    ],
    electron_density_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]
    ],
    electron_temperature_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]
    ],
) -> list[ray.RayQuantities]:
    """Compute additional quantities for the ray states."""
    ray_quantity_list = []

    for state in ray_states:
        position = state.position
        refractive_index = state.refractive_index

        magnetic_field = magnetic_field_interpolator(position)
        electron_density = electron_density_interpolator(position)
        electron_temperature = electron_temperature_interpolator(position)

        absorption_coefficient = absorption.absorption_coefficient_conditional(
            refractive_index=refractive_index,
            magnetic_field=magnetic_field,
            electron_density_1e20_per_m3=electron_density,
            electron_temperature_keV=electron_temperature,
            frequency=setting.frequency,
            mode=setting.mode,
        )

        linear_power_density = absorption_coefficient * jnp.exp(-state.optical_depth)

        ray_quantities = ray.RayQuantities(
            magnetic_field=magnetic_field,
            absorption_coefficient=jnp.asarray(absorption_coefficient),
            electron_density=electron_density,
            electron_temperature=electron_temperature,
            linear_power_density=linear_power_density,
        )
        ray_quantity_list.append(ray_quantities)

    return ray_quantity_list
