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
    """Extract RayState from the 7-component ODE state vector.

    State vector structure (7 components):
    - y[0:3]: position
    - y[3:6]: refractive_index
    - y[6]:   optical_depth
    """
    return ray.RayState(
        position=y[:3],
        refractive_index=y[3:6],
        optical_depth=y[6],
        arc_length=jnp.array(s),
    )


def _state_to_y(
    state: ray.RayState,
) -> jt.Float[jax.Array, " n "]:
    """Pack RayState into the 7-component ODE state vector."""
    return jnp.concatenate(
        [
            state.position,
            state.refractive_index,
            jnp.array([state.optical_depth]),
        ]
    )


def _right_hand_side(
    s: float,
    y: jt.Float[jax.Array, " n "],
    setting: ray.RaySetting,
    magnetic_field_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]
    ],
    rho_interpolator: Callable[[jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]],
    electron_density_profile_interpolator: Callable[
        [jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]
    ],
    electron_temperature_profile_interpolator: Callable[
        [jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]
    ],
) -> jt.Float[jax.Array, " n "]:
    r"""Compute the right-hand side of the 7-component ray tracing ODE.

    Integrates (r, N, τ) only. B, ρ, nₑ, Tₑ are not tracked in the ODE
    state — they are recomputed at output points via vectorised post-processing,
    which avoids the jacfwd(B) calls and the tight per-component tolerances that
    previously forced ~2 mm step sizes.
    """
    state = _y_to_state(y, s)

    # Compute both Hamiltonian gradients in a single backward pass
    hamiltonian_gradient_r, hamiltonian_gradient_n = hamiltonian.hamiltonian_gradients(
        state,
        setting,
        magnetic_field_interpolator,
        rho_interpolator,
        electron_density_profile_interpolator,
    )
    norm = jnp.linalg.norm(hamiltonian_gradient_n)

    dr_ds = hamiltonian_gradient_n / norm
    dn_ds = -hamiltonian_gradient_r / norm

    rho = rho_interpolator(state.position)
    ne = electron_density_profile_interpolator(rho)
    te = electron_temperature_profile_interpolator(rho)
    mag = magnetic_field_interpolator(state.position)
    dtau_ds = absorption.absorption_coefficient_conditional(
        refractive_index=state.refractive_index,
        magnetic_field=mag,
        electron_density_1e20_per_m3=ne,
        electron_temperature_keV=te,
        frequency=setting.frequency,
        mode=setting.mode,
    )

    return jnp.concatenate([dr_ds, dn_ds, jnp.array([dtau_ds])])


# Persistent module-level JIT. Using static_argnames for the interpolators means
# JAX caches one compiled trace per unique combination of callable objects. As long
# as the same Python objects are passed across solve() calls, JAX reuses the cached
# trace instead of re-tracing from scratch each call (~370 ms overhead avoided).
_rhs_jitted = jax.jit(
    _right_hand_side,
    static_argnames=(
        "magnetic_field_interpolator",
        "rho_interpolator",
        "electron_density_profile_interpolator",
        "electron_temperature_profile_interpolator",
    ),
)


def _compute_quantities_at_positions(
    positions: jt.Float[jax.Array, "n 3"],
    optical_depths: jt.Float[jax.Array, " n"],
    arc_lengths: jt.Float[jax.Array, " n"],
    magnetic_field_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]
    ],
    rho_interpolator: Callable[[jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]],
    electron_density_profile_interpolator: Callable[
        [jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]
    ],
    electron_temperature_profile_interpolator: Callable[
        [jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]
    ],
) -> list[ray.RayQuantities]:
    """Evaluate all diagnostic quantities at a batch of trajectory positions.

    Uses jax.vmap for a single vectorised XLA call rather than n separate
    JIT-dispatched calls, eliminating per-point Python overhead.
    """
    B_all = jax.vmap(magnetic_field_interpolator)(positions)
    rho_all = jax.vmap(rho_interpolator)(positions)
    ne_all = jax.vmap(electron_density_profile_interpolator)(rho_all)
    te_all = jax.vmap(electron_temperature_profile_interpolator)(rho_all)

    # Derive alpha = dtau/ds from finite differences of the ODE-integrated optical depth
    ds = jnp.diff(arc_lengths)
    dtau = jnp.diff(optical_depths)
    alpha_interior = dtau / jnp.where(ds > 0, ds, 1.0)
    alpha_all = jnp.concatenate([alpha_interior[:1], alpha_interior])
    P_all = alpha_all * jnp.exp(-optical_depths)

    return [
        ray.RayQuantities(
            magnetic_field=B_all[i],
            normalized_effective_radius=rho_all[i],
            electron_density=ne_all[i],
            electron_temperature=te_all[i],
            absorption_coefficient=alpha_all[i],
            linear_power_density=P_all[i],
        )
        for i in range(len(optical_depths))
    ]


def _straight_line_trace(
    position: jt.Float[jax.Array, "3"],
    direction: jt.Float[jax.Array, "3"],
    magnetic_field_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]
    ],
    rho_interpolator: Callable[[jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]],
    step_size: float = 0.01,
    max_steps: int = 100,
) -> tuple[jt.Float[jax.Array, "3"], jt.Float[jax.Array, ""]]:
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
        Tuple of (final_position, distance_traveled)
    """
    # Check if we're already at a valid position
    initial_B = magnetic_field_interpolator(position)
    initial_rho = rho_interpolator(position)
    initial_valid = jnp.logical_and(jnp.linalg.norm(initial_B) > 0, initial_rho <= 1.0)

    def cond_fun(state_tuple):
        pos, step_count, found_valid = state_tuple
        return jnp.logical_and(jnp.logical_not(found_valid), step_count < max_steps)

    def body_fun(state_tuple):
        pos, step_count, found_valid = state_tuple
        new_pos = pos + step_size * direction

        B = magnetic_field_interpolator(new_pos)
        rho = rho_interpolator(new_pos)
        new_found_valid = jnp.logical_and(jnp.linalg.norm(B) > 0, rho <= 1.0)

        return (new_pos, step_count + 1, new_found_valid)

    initial_state_tuple = (position, 0, initial_valid)
    final_position, step_count, found_valid = jax.lax.while_loop(
        cond_fun, body_fun, initial_state_tuple
    )

    distance_traveled = step_count * step_size

    final_pos = jnp.where(found_valid, final_position, position)
    final_dist = jnp.where(found_valid, distance_traveled, 0.0)
    return final_pos, final_dist


_straight_line_trace_jitted = jax.jit(
    _straight_line_trace,
    static_argnames=("magnetic_field_interpolator", "rho_interpolator"),
)


def solve(
    state: ray.RayState,
    setting: ray.RaySetting,
    magnetic_field_interpolator: Callable[
        [jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]
    ],
    rho_interpolator: Callable[[jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]],
    electron_density_profile_interpolator: Callable[
        [jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]
    ],
    electron_temperature_profile_interpolator: Callable[
        [jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]
    ],
    use_straight_line_init: bool = True,
    straight_line_step_size: float = 0.01,
    max_straight_line_steps: int = 100,
) -> tuple[list[ray.RayState], list[ray.RayQuantities]]:
    """Solve the ray tracing equations.

    Integrates a 7-component ODE: (position[3], refractive_index[3], optical_depth[1]).
    Diagnostic quantities (B, ρ, nₑ, Tₑ, α, P) are NOT part of the ODE state; they
    are evaluated in a single vectorised post-processing pass after the integration,
    avoiding the jacfwd(B) calls and tight tolerances that previously caused ~2 mm steps.

    Args:
        state: Initial ray state
        setting: Ray tracing settings
        magnetic_field_interpolator: Function to evaluate magnetic field at a position
        rho_interpolator: Function to evaluate radial coordinate at a position
        electron_density_profile_interpolator: maps rho -> electron density
        electron_temperature_profile_interpolator: maps rho -> electron temperature
        use_straight_line_init: Whether to use straight line tracing to find valid starting point
        straight_line_step_size: Size of each step for straight line tracing
        max_straight_line_steps: Maximum number of steps for straight line tracing

    Returns:
        Tuple of (ray_states, ray_quantities) representing the ray trajectory and associated quantities
    """

    initial_position = state.position
    initial_arc_length = state.arc_length

    if use_straight_line_init:
        direction = jnp.asarray(state.refractive_index)

        initial_position, straight_line_distance = _straight_line_trace_jitted(
            position=initial_position,
            direction=direction,
            magnetic_field_interpolator=magnetic_field_interpolator,
            rho_interpolator=rho_interpolator,
            step_size=straight_line_step_size,
            max_steps=max_straight_line_steps,
        )
        initial_arc_length = state.arc_length + straight_line_distance
        state = ray.RayState(
            position=initial_position,
            refractive_index=state.refractive_index,
            optical_depth=state.optical_depth,
            arc_length=initial_arc_length,
        )

    def ray_exits_plasma(_t, y):
        return float(rho_interpolator(jnp.asarray(y[:3]))) - 1.05

    ray_exits_plasma.terminal = True  # type: ignore
    ray_exits_plasma.direction = 1  # type: ignore

    def ray_fully_absorbed(_t, y):
        return float(jnp.exp(-y[6])) - 1e-3

    ray_fully_absorbed.terminal = True  # type: ignore
    ray_fully_absorbed.direction = -1  # type: ignore

    def ray_out_of_bounds(_t, y):
        return 20.0 - float(jnp.linalg.norm(jnp.asarray(y[:3])))

    ray_out_of_bounds.terminal = True  # type: ignore
    ray_out_of_bounds.direction = -1  # type: ignore

    fun = partial(
        _rhs_jitted,
        setting=setting,
        magnetic_field_interpolator=magnetic_field_interpolator,
        rho_interpolator=rho_interpolator,
        electron_density_profile_interpolator=electron_density_profile_interpolator,
        electron_temperature_profile_interpolator=electron_temperature_profile_interpolator,
    )
    y0 = _state_to_y(state)
    t_start = float(initial_arc_length)
    res = integrate.solve_ivp(
        fun=fun,
        t_span=(t_start, t_start + 20.0),
        y0=y0,
        method="LSODA",
        events=[ray_exits_plasma, ray_fully_absorbed, ray_out_of_bounds],
        dense_output=False,
        max_step=0.05,
        rtol=1e-4,
        atol=1e-6,
    )

    # Extract ray states from the 7-component ODE solution
    ray_states = [
        _y_to_state(y=jnp.asarray(y), s=t)
        for t, y in zip(res.t, res.y.T)
    ]

    # Vectorised post-processing: evaluate all diagnostics in one vmap call
    positions = jnp.stack([s.position for s in ray_states])
    optical_depths = jnp.array([float(s.optical_depth) for s in ray_states])
    arc_lengths = jnp.array([float(s.arc_length) for s in ray_states])
    ray_quantities = _compute_quantities_at_positions(
        positions=positions,
        optical_depths=optical_depths,
        arc_lengths=arc_lengths,
        magnetic_field_interpolator=magnetic_field_interpolator,
        rho_interpolator=rho_interpolator,
        electron_density_profile_interpolator=electron_density_profile_interpolator,
        electron_temperature_profile_interpolator=electron_temperature_profile_interpolator,
    )

    return ray_states, ray_quantities
