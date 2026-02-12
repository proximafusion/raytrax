import diffrax
import interpax
import jax
import jax.numpy as jnp
import jaxtyping as jt

from raytrax import absorption, hamiltonian, ray
from raytrax.interpolate import (
    _apply_B_stellarator_symmetry,
    _map_to_fundamental_domain,
)
from raytrax.types import Interpolators


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


def _right_hand_side(
    s: float,
    y: jt.Float[jax.Array, " n "],
    args: tuple,
) -> jt.Float[jax.Array, " n "]:
    r"""Compute the right-hand side of the 7-component ray tracing ODE.

    Integrates (r, N, τ) only. B, ρ, nₑ, Tₑ are not tracked in the ODE
    state — they are recomputed at output points via vectorised post-processing,
    which avoids the jacfwd(B) calls and the tight per-component tolerances that
    previously forced ~2 mm step sizes.

    Uses diffrax's ``(t, y, args)`` calling convention. The args tuple is
    ``(setting, interpolators, nfp)`` where ``interpolators`` is an
    :class:`Interpolators` pytree.
    """
    setting, interpolators, nfp = args

    state = _y_to_state(y, s)

    # Helper function to evaluate B field interpolator with coordinate transforms
    def eval_B(position: jt.Float[jax.Array, "3"]) -> jt.Float[jax.Array, "3"]:
        r = jnp.sqrt(position[0] ** 2 + position[1] ** 2)
        phi = jnp.arctan2(position[1], position[0])
        z = position[2]
        phi_mapped, z_query, in_second_half = _map_to_fundamental_domain(phi, z, nfp)
        B_grid = interpolators.magnetic_field(r, phi_mapped, z_query)
        return _apply_B_stellarator_symmetry(B_grid, phi_mapped, phi, in_second_half)

    # Helper function to evaluate rho interpolator with coordinate transforms
    def eval_rho(position: jt.Float[jax.Array, "3"]) -> jt.Float[jax.Array, ""]:
        r = jnp.sqrt(position[0] ** 2 + position[1] ** 2)
        phi = jnp.arctan2(position[1], position[0])
        z = position[2]
        phi_mapped, z_query, _ = _map_to_fundamental_domain(phi, z, nfp)
        return interpolators.rho(r, phi_mapped, z_query)

    # Compute both Hamiltonian gradients in a single backward pass
    hamiltonian_gradient_r, hamiltonian_gradient_n = hamiltonian.hamiltonian_gradients(
        state,
        setting,
        eval_B,
        eval_rho,
        interpolators.electron_density,
    )
    norm = jnp.linalg.norm(hamiltonian_gradient_n)

    dr_ds = hamiltonian_gradient_n / norm
    dn_ds = -hamiltonian_gradient_r / norm

    rho = eval_rho(state.position)
    ne = interpolators.electron_density(rho)
    te = interpolators.electron_temperature(rho)
    mag = eval_B(state.position)
    dtau_ds = absorption.absorption_coefficient_conditional(
        refractive_index=state.refractive_index,
        magnetic_field=mag,
        electron_density_1e20_per_m3=ne,
        electron_temperature_keV=te,
        frequency=setting.frequency,
        mode=setting.mode,
    )

    return jnp.concatenate([dr_ds, dn_ds, jnp.array([dtau_ds])])


def _straight_line_trace(
    position: jt.Float[jax.Array, "3"],
    direction: jt.Float[jax.Array, "3"],
    interpolators: Interpolators,
    nfp: int,
    step_size: float = 0.01,
    max_steps: int = 100,
) -> tuple[jt.Float[jax.Array, "3"], jt.Float[jax.Array, ""]]:
    """Trace a straight line until finding a position inside the plasma.

    Takes small steps in the given direction until both the magnetic field
    magnitude is positive and rho <= 1, or until max_steps is reached.
    """

    def eval_B(pos: jt.Float[jax.Array, "3"]) -> jt.Float[jax.Array, "3"]:
        r = jnp.sqrt(pos[0] ** 2 + pos[1] ** 2)
        phi = jnp.arctan2(pos[1], pos[0])
        z = pos[2]
        phi_mapped, z_query, in_second_half = _map_to_fundamental_domain(phi, z, nfp)
        B_grid = interpolators.magnetic_field(r, phi_mapped, z_query)
        return _apply_B_stellarator_symmetry(B_grid, phi_mapped, phi, in_second_half)

    def eval_rho(pos: jt.Float[jax.Array, "3"]) -> jt.Float[jax.Array, ""]:
        r = jnp.sqrt(pos[0] ** 2 + pos[1] ** 2)
        phi = jnp.arctan2(pos[1], pos[0])
        z = pos[2]
        phi_mapped, z_query, _ = _map_to_fundamental_domain(phi, z, nfp)
        return interpolators.rho(r, phi_mapped, z_query)

    # Check if we're already at a valid position
    initial_B = eval_B(position)
    initial_rho = eval_rho(position)
    initial_valid = jnp.logical_and(jnp.linalg.norm(initial_B) > 0, initial_rho <= 1.0)

    def cond_fun(state_tuple):
        pos, step_count, found_valid = state_tuple
        return jnp.logical_and(jnp.logical_not(found_valid), step_count < max_steps)

    def body_fun(state_tuple):
        pos, step_count, found_valid = state_tuple
        new_pos = pos + step_size * direction

        B = eval_B(new_pos)
        rho = eval_rho(new_pos)
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


def _cond_exit(t, y, args, **kwargs):
    """Terminate when ray exits plasma (rho > 1.05)."""
    pos = y[:3]
    r = jnp.sqrt(pos[0] ** 2 + pos[1] ** 2)
    phi = jnp.arctan2(pos[1], pos[0])
    z = pos[2]
    nfp_ = args[2]
    phi_mapped, z_query, _ = _map_to_fundamental_domain(phi, z, nfp_)
    rho_val = args[1].rho(r, phi_mapped, z_query)
    return rho_val - 1.05


def _cond_absorbed(t, y, args, **kwargs):
    """Terminate when ray is fully absorbed (exp(-tau) < 1e-3)."""
    return jnp.exp(-y[6]) - 1e-3


def _cond_oob(t, y, args, **kwargs):
    """Terminate when ray leaves the computational domain (|r| > 20 m)."""
    return 20.0 - jnp.linalg.norm(y[:3])


_event = diffrax.Event(
    cond_fn=[_cond_exit, _cond_absorbed, _cond_oob],
    direction=[True, False, False],
)

_term = diffrax.ODETerm(_right_hand_side)
_solver = diffrax.Tsit5()
_stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6, dtmax=0.05)
_saveat = diffrax.SaveAt(steps=True, t0=True)


@jax.jit
def trace_jitted(
    position: jt.Float[jax.Array, "3"],
    direction: jt.Float[jax.Array, "3"],
    setting: ray.RaySetting,
    interpolators: Interpolators,
    nfp: int,
    rho_1d: jt.Float[jax.Array, " nrho"],
    dvolume_drho: jt.Float[jax.Array, " nrho"],
) -> tuple:
    """Fully JIT-compiled ray trace: vacuum propagation + ODE solve + diagnostics + radial profile.

    Returns fixed-size arrays (padded to max_steps=4096). Invalid entries beyond the
    last integration step are inf/NaN and must be trimmed by the caller.
    """
    # 1. Straight-line trace to plasma entry
    entry_pos, vacuum_dist = _straight_line_trace(
        position,
        direction,
        interpolators,
        nfp,
    )

    # 2. Build ODE initial state
    y0 = jnp.concatenate([entry_pos, direction, jnp.array([0.0])])
    t_start = vacuum_dist
    t_end = t_start + 20.0
    args = (setting, interpolators, nfp)

    # 3. ODE solve
    sol = diffrax.diffeqsolve(
        terms=_term,
        solver=_solver,
        t0=t_start,
        t1=t_end,
        dt0=0.001,
        y0=y0,
        args=args,
        saveat=_saveat,
        stepsize_controller=_stepsize_controller,
        event=_event,
        max_steps=4096,
        throw=False,
    )
    ts = sol.ts
    ys = sol.ys

    # 4. Post-processing: compute diagnostics on the full buffer
    positions = ys[:, :3]
    optical_depths = ys[:, 6]

    def eval_B(pos):
        r = jnp.sqrt(pos[0] ** 2 + pos[1] ** 2)
        phi = jnp.arctan2(pos[1], pos[0])
        z = pos[2]
        phi_mapped, z_query, in_second_half = _map_to_fundamental_domain(phi, z, nfp)
        B_grid = interpolators.magnetic_field(r, phi_mapped, z_query)
        return _apply_B_stellarator_symmetry(B_grid, phi_mapped, phi, in_second_half)

    def eval_rho(pos):
        r = jnp.sqrt(pos[0] ** 2 + pos[1] ** 2)
        phi = jnp.arctan2(pos[1], pos[0])
        z = pos[2]
        phi_mapped, z_query, _ = _map_to_fundamental_domain(phi, z, nfp)
        return interpolators.rho(r, phi_mapped, z_query)

    B_all = jax.vmap(eval_B)(positions)
    rho_all = jax.vmap(eval_rho)(positions)
    ne_all = jax.vmap(interpolators.electron_density)(rho_all)
    te_all = jax.vmap(interpolators.electron_temperature)(rho_all)

    # Derive alpha = dtau/ds from finite differences
    ds = jnp.diff(ts)
    dtau = jnp.diff(optical_depths)
    alpha_interior = dtau / jnp.where(ds > 0, ds, 1.0)
    alpha_all = jnp.concatenate([alpha_interior[:1], alpha_interior])
    P_all = alpha_all * jnp.exp(-optical_depths)

    # 5. Radial profile: dP/dV from finite differences
    drho_ds = jnp.diff(rho_all) / jnp.where(ds > 0, ds, 1.0)
    drho_ds_padded = jnp.concatenate([drho_ds[:1], drho_ds, drho_ds[-1:]])
    drho_ds_avg = 0.5 * (drho_ds_padded[:-1] + drho_ds_padded[1:])

    dV_drho = interpax.interp1d(
        rho_all, rho_1d, dvolume_drho, method="cubic", extrap=True
    )
    dP_drho = P_all / jnp.where(jnp.abs(drho_ds_avg) > 0, jnp.abs(drho_ds_avg), 1.0)
    dP_dV = dP_drho / jnp.where(jnp.abs(dV_drho) > 0, dV_drho, 1.0)

    return ts, ys, B_all, rho_all, ne_all, te_all, alpha_all, P_all, dP_dV
