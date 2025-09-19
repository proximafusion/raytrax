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
    rho_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]
    ],
    electron_density_profile_interpolator: Callable[
        [jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]
    ],
    electron_temperature_profile_interpolator: Callable[
        [jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]
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
        state, setting, magnetic_field_interpolator, rho_interpolator, electron_density_profile_interpolator
    )
    hamiltonian_gradient_r = hamiltonian.hamiltonian_gradient_r(
        state, setting, magnetic_field_interpolator, rho_interpolator, electron_density_profile_interpolator
    )
    norm = jnp.linalg.norm(hamiltonian_gradient_n)
    # FIXME add back absorption
    absorption_coefficient = 0.0
    # absorption_coefficient = absorption.absorption_coefficient_conditional(
    #     refractive_index=state.refractive_index,
    #     magnetic_field=magnetic_field_interpolator(state.position),
    #     electron_density_1e20_per_m3=electron_density_interpolator(state.position),
    #     electron_temperature_keV=electron_temperature_interpolator(state.position),
    #     frequency=setting.frequency,
    #     mode=setting.mode,
    # )
    right_hand_side = ray.Term(
        position=hamiltonian_gradient_n / norm,
        refractive_index=-hamiltonian_gradient_r / norm,
        optical_depth=jnp.asarray(absorption_coefficient),
    )
    return _term_to_dydt(right_hand_side)


def _straight_line_trace(
    position: jt.Float[jax.Array, "3"],
    direction: jt.Float[jax.Array, "3"],
    magnetic_field_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]
    ],
    rho_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]
    ],
    step_size: float = 0.01,
    max_steps: int = 100,
) -> jt.Float[jax.Array, "3"]:
    """Trace a straight line until finding a position with positive magnetic field and electron density.

    This function performs straight line ray tracing by taking small steps in the given direction
    until both the magnetic field magnitude and electron density are positive, or until the maximum
    number of steps is reached.

    Args:
        position: Starting position vector
        direction: Normalized direction vector for the straight line
        magnetic_field_interpolator: Function to evaluate magnetic field at a position
        rho_interpolator: Function to evaluate radial coordinate at a position
        step_size: Size of each step along the straight line
        max_steps: Maximum number of steps to take

    Returns:
        The final position vector after tracing
    """
    # Check if we're already at a valid position
    initial_B = magnetic_field_interpolator(position)
    initial_rho = rho_interpolator(position)
    initial_valid = jnp.logical_and(jnp.linalg.norm(initial_B) > 0, initial_rho <= 1.0)

    # Condition function for lax.while_loop that determines when to stop the loop
    # Continues if we haven't found a valid point yet and haven't exceeded max_steps
    def cond_fun(state_tuple):
        pos, step_count, found_valid = state_tuple
        return jnp.logical_and(jnp.logical_not(found_valid), step_count < max_steps)

    # Body function for lax.while_loop that performs one step of the straight line trace
    def body_fun(state_tuple):
        pos, step_count, found_valid = state_tuple
        new_pos = pos + step_size * direction

        B = magnetic_field_interpolator(new_pos)
        rho = rho_interpolator(new_pos)
        new_found_valid = jnp.logical_and(jnp.linalg.norm(B) > 0, rho <= 1.0)

        # Return updated state tuple (position, step count, validity flag)
        return (new_pos, step_count + 1, new_found_valid)

    initial_state_tuple = (position, 0, initial_valid)
    final_position, _, found_valid = jax.lax.while_loop(
        cond_fun, body_fun, initial_state_tuple
    )

    # Return the original position if no valid position was found
    return jnp.where(found_valid, final_position, position)


def solve(
    state: ray.RayState,
    setting: ray.RaySetting,
    magnetic_field_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]
    ],
    rho_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]
    ],
    electron_density_profile_interpolator: Callable[
        [jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]
    ],
    electron_temperature_profile_interpolator: Callable[
        [jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]
    ],
    use_straight_line_init: bool = True,
    straight_line_step_size: float = 0.01,
    max_straight_line_steps: int = 100,
) -> list[ray.RayState]:
    """Solve the ray tracing equations.

    This function solves the ray tracing equations with optional straight line initialization
    to find a valid starting point where both magnetic field and electron density are positive.

    Args:
        state: Initial ray state
        setting: Ray tracing settings
        magnetic_field_interpolator: Function to evaluate magnetic field at a position
        rho_interpolator: Function to evaluate radial coordinate at a position
        electron_density_profile_interpolator: Function to evaluate electron density at a radial coordinate
        electron_temperature_profile_interpolator: Function to evaluate electron temperature at a radial coordinate
        use_straight_line_init: Whether to use straight line tracing to find valid starting point
        straight_line_step_size: Size of each step for straight line tracing
        max_straight_line_steps: Maximum number of steps for straight line tracing

    Returns:
        List of ray states representing the ray trajectory
    """

    initial_position = state.position

    if use_straight_line_init:
        # Safely normalize the direction vector
        direction = jnp.asarray(state.refractive_index)

        # Use static_argnames to specify that the interpolator functions are static (non-traceable)
        straight_line_trace_jit = jax.jit(
            _straight_line_trace,
            static_argnames=[
                "magnetic_field_interpolator",
                "rho_interpolator",
            ],
        )
        initial_position = straight_line_trace_jit(
            position=initial_position,
            direction=direction,
            magnetic_field_interpolator=magnetic_field_interpolator,
            rho_interpolator=rho_interpolator,
            step_size=straight_line_step_size,
            max_steps=max_straight_line_steps,
        )
        state = ray.RayState(
            position=initial_position,
            refractive_index=state.refractive_index,
            optical_depth=state.optical_depth,
            arc_length=state.arc_length,
        )

    fun = partial(
        _right_hand_side,
        setting=setting,
        magnetic_field_interpolator=magnetic_field_interpolator,
        rho_interpolator=rho_interpolator,
        electron_density_profile_interpolator=electron_density_profile_interpolator,
        electron_temperature_profile_interpolator=electron_temperature_profile_interpolator,
    )
    y0 = _state_to_y(state)
    jitted_fun = jax.jit(fun)
    res = integrate.solve_ivp(
        fun=jitted_fun,
        # FIXME use a different stopping criterion
        t_span=(0, 0.1),
        y0=y0,
        max_step=0.01,
        min_step=0.001,
        method="LSODA",
    )
    return [_y_to_state(y=jnp.asarray(y), s=t) for t, y in zip(res.t, res.y.T)]


def compute_additional_quantities(
    ray_states: list[ray.RayState],
    setting: ray.RaySetting,
    magnetic_field_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]
    ],
    rho_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]
    ],
    electron_density_profile_interpolator: Callable[
        [jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]
    ],
    electron_temperature_profile_interpolator: Callable[
        [jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]
    ],
) -> list[ray.RayQuantities]:
    """Compute additional quantities for the ray states."""
    ray_quantity_list = []

    for state in ray_states:
        position = state.position
        refractive_index = state.refractive_index

        magnetic_field = magnetic_field_interpolator(position)
        rho = rho_interpolator(position)
        electron_density = electron_density_profile_interpolator(rho)
        electron_temperature = electron_temperature_profile_interpolator(rho)

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
