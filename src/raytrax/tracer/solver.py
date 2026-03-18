"""JIT-compiled ray tracing ODE solver: integrates position, refractive index, and optical depth via diffrax."""

from typing import NamedTuple, cast

import diffrax
import interpax
import jax
import jax.numpy as jnp
import jaxtyping as jt

from raytrax.physics import absorption, hamiltonian
from raytrax.tracer import ray
from raytrax.tracer.buffers import Interpolators, TraceBuffers
from raytrax.types import TracerSettings


def _map_to_fundamental_domain(
    phi: jt.Float[jax.Array, ""],
    z: jt.Float[jax.Array, ""],
    nfp: int,
) -> tuple[
    jt.Float[jax.Array, ""],
    jt.Float[jax.Array, ""],
    jt.Bool[jax.Array, ""],
]:
    """Map toroidal angle and z to the fundamental domain [0, π/nfp] using stellarator symmetry.

    For points in the second half of a field period (phi_mod > π/nfp), the stellarator
    symmetry maps (R, phi, Z) to (R, phi_mapped, -Z), so z must also be reflected.

    Args:
        phi: Toroidal angle in radians (can be any value)
        z: Cylindrical z coordinate
        nfp: Number of field periods

    Returns:
        A tuple of (phi_mapped, z_query, in_second_half) where phi_mapped is in [0, π/nfp],
        z_query is the z to use for the grid lookup, and in_second_half indicates whether
        the original phi was in the second half of a field period.
    """
    period = 2.0 * jnp.pi / nfp
    half_period = jnp.pi / nfp
    phi_mod = phi % period
    in_second_half = phi_mod > half_period
    phi_mapped = jnp.where(in_second_half, period - phi_mod, phi_mod)
    z_query = jnp.where(in_second_half, -z, z)
    return phi_mapped, z_query, in_second_half


def _apply_B_stellarator_symmetry(
    B_cyl: jt.Float[jax.Array, "3"],
    in_second_half: jt.Bool[jax.Array, ""],
) -> jt.Float[jax.Array, "3"]:
    """Apply stellarator symmetry to recover B at the original point from its mirror.

    The grid covers only the fundamental domain [0, pi/nfp].  For phi in the
    second half of a field period the grid was queried at the mirror point
    (phi_mapped, -z).  Stellarator symmetry (phi->-phi, z->-z) gives
    B_R(-phi,-z) = -B_R(phi,z) and B_phi(-phi,-z) = B_phi(phi,z),
    B_Z(-phi,-z) = B_Z(phi,z) (eq. 6.9 in arXiv:2502.04374), so B_R from the
    grid must be sign-flipped to recover the field at the original point.

    Args:
        B_cyl: Cylindrical (B_R, B_phi, B_Z) from the grid at the mirror point
        in_second_half: Whether the query phi is in the second half of a field period

    Returns:
        Cylindrical (B_R, B_phi, B_Z) with symmetry applied
    """
    sign = jnp.where(in_second_half, -1.0, 1.0)
    return jnp.stack([sign * B_cyl[0], B_cyl[1], B_cyl[2]])


def _cylindrical_to_cartesian_B(
    B_cyl: jt.Float[jax.Array, "3"],
    phi: jt.Float[jax.Array, ""],
) -> jt.Float[jax.Array, "3"]:
    """Rotate cylindrical (B_R, B_phi, B_Z) to Cartesian (B_x, B_y, B_z)."""
    cp, sp = jnp.cos(phi), jnp.sin(phi)
    return jnp.stack(
        [
            B_cyl[0] * cp - B_cyl[1] * sp,
            B_cyl[0] * sp + B_cyl[1] * cp,
            B_cyl[2],
        ]
    )


def _eval_magnetic_field(
    position: jt.Float[jax.Array, "3"],
    interpolators: Interpolators,
    nfp: int,
) -> jt.Float[jax.Array, "3"]:
    """Evaluate B field at a Cartesian position."""
    r = jnp.sqrt(position[0] ** 2 + position[1] ** 2)
    phi = jnp.arctan2(position[1], position[0])
    z = position[2]
    if interpolators.is_axisymmetric:
        B_interp_2d = cast(interpax.Interpolator2D, interpolators.magnetic_field)
        B_cyl = B_interp_2d(r, z)
    else:
        phi_mapped, z_query, in_second_half = _map_to_fundamental_domain(phi, z, nfp)
        B_interp_3d = cast(interpax.Interpolator3D, interpolators.magnetic_field)
        B_cyl = B_interp_3d(r, phi_mapped, z_query)
        B_cyl = _apply_B_stellarator_symmetry(B_cyl, in_second_half)
    return _cylindrical_to_cartesian_B(B_cyl, phi)


def _eval_rho(
    position: jt.Float[jax.Array, "3"],
    interpolators: Interpolators,
    nfp: int,
) -> jt.Float[jax.Array, ""]:
    """Evaluate the normalized effective radius at a Cartesian position."""
    r = jnp.sqrt(position[0] ** 2 + position[1] ** 2)
    z = position[2]
    if interpolators.is_axisymmetric:
        rho_interp_2d = cast(interpax.Interpolator2D, interpolators.rho)
        return rho_interp_2d(r, z)
    phi = jnp.arctan2(position[1], position[0])
    phi_mapped, z_query, _ = _map_to_fundamental_domain(phi, z, nfp)
    rho_interp_3d = cast(interpax.Interpolator3D, interpolators.rho)
    return rho_interp_3d(r, phi_mapped, z_query)


def _y_to_state(
    y: jt.Float[jax.Array, " n "],
    s: float | int | jax.Array,
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
    s: float | int | jax.Array,
    y: jt.Float[jax.Array, " n "],
    args: tuple,
) -> jt.Float[jax.Array, " n "]:
    r"""Compute the right-hand side of the 7-component ray tracing ODE.

    Integrates (r, N, τ) only. B, ρ, nₑ, Tₑ are not tracked in the ODE
    state — they are recomputed at output points via vectorised post-processing,
    which avoids the jacfwd(B) calls and the tight per-component tolerances that
    previously forced ~2 mm step sizes.

    Uses diffrax's ``(t, y, args)`` calling convention. The args tuple is
    ``(setting, interpolators, nfp, tracer_settings)`` where ``interpolators`` is an
    :class:`Interpolators` pytree.
    """
    setting, interpolators, nfp, _ = args

    state = _y_to_state(y, s)

    def eval_B(pos):
        return _eval_magnetic_field(pos, interpolators, nfp)

    def eval_rho(pos):
        return _eval_rho(pos, interpolators, nfp)

    # Compute both Hamiltonian gradients in a single backward pass,
    # reusing B-field, rho, and ne from the forward pass (has_aux=True).
    (hamiltonian_gradient_r, hamiltonian_gradient_n), hamiltonian_aux = (
        hamiltonian.hamiltonian_gradients(
            state.position,
            state.refractive_index,
            eval_B,
            eval_rho,
            interpolators.electron_density,
            setting.frequency,
            setting.mode,
        )
    )
    norm = jnp.linalg.norm(hamiltonian_gradient_n)

    dr_ds = hamiltonian_gradient_n / norm
    dn_ds = -hamiltonian_gradient_r / norm

    te = interpolators.electron_temperature(hamiltonian_aux.rho)
    dtau_ds = absorption.absorption_coefficient_conditional(
        refractive_index=state.refractive_index,
        magnetic_field=hamiltonian_aux.magnetic_field,
        electron_density_1e20_per_m3=hamiltonian_aux.electron_density_1e20_per_m3,
        electron_temperature_keV=te,
        frequency=setting.frequency,
        mode=setting.mode,
    )

    return jnp.concatenate([dr_ds, dn_ds, jnp.array([dtau_ds])])


def _cond_exit(t, y, args, **kwargs):
    """Terminate when ray exits plasma (rho > 1.05)."""
    return _eval_rho(y[:3], args[1], args[2]) - 1.05


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

_term = diffrax.ODETerm(_right_hand_side)  # type: ignore[arg-type]
_solver = diffrax.Tsit5()


def _solve(
    position: jt.Float[jax.Array, "3"],
    direction: jt.Float[jax.Array, "3"],
    setting: ray.RaySetting,
    interpolators: Interpolators,
    nfp: int,
    tracer_settings: TracerSettings,
) -> diffrax.Solution:
    """Core ODE solve called by trace_jitted.

    Starts from `position` directly. The vacuum region (ne=0, rho>1) is handled
    automatically: the Hamiltonian switches to _hamiltonian_vacuum when ne<1e-6,
    giving straight-line propagation until the beam enters plasma.

    Output is saved at every accepted step (plus t0).  After an early-exit
    event, diffrax leaves unvisited SaveAt slots as inf; sol.t1 is always the
    scheduled t1 (max_arc_length), not the event time.
    """
    stepsize_controller = diffrax.PIDController(
        rtol=tracer_settings.relative_tolerance,
        atol=tracer_settings.absolute_tolerance,
        dtmax=tracer_settings.max_step_size,
    )
    y0 = jnp.concatenate([position, direction, jnp.array([0.0])])
    return diffrax.diffeqsolve(
        terms=_term,
        solver=_solver,
        t0=0.0,
        t1=tracer_settings.max_arc_length,
        dt0=0.001,
        y0=y0,
        args=(setting, interpolators, nfp, tracer_settings),
        saveat=diffrax.SaveAt(steps=True, t0=True),
        stepsize_controller=stepsize_controller,
        event=_event,
        max_steps=4096,
        throw=False,
    )


class _BeamDiagnostics(NamedTuple):
    """Plasma quantities evaluated along the trajectory."""

    magnetic_field: jt.Float[jax.Array, "nsteps 3"]
    rho: jt.Float[jax.Array, " nsteps"]
    electron_density: jt.Float[jax.Array, " nsteps"]
    electron_temperature: jt.Float[jax.Array, " nsteps"]
    absorption_coefficient: jt.Float[jax.Array, " nsteps"]
    linear_power_density: jt.Float[jax.Array, " nsteps"]


def _compute_beam_diagnostics(
    ts: jt.Float[jax.Array, " nsteps"],
    ys: jt.Float[jax.Array, "nsteps 7"],
    interpolators: Interpolators,
    nfp: int,
) -> _BeamDiagnostics:
    """Evaluate plasma quantities along the trajectory and derive absorption/power."""
    positions = ys[:, :3]
    optical_depths = ys[:, 6]

    B_all = jax.vmap(lambda pos: _eval_magnetic_field(pos, interpolators, nfp))(
        positions
    )
    rho_all = jax.vmap(lambda pos: _eval_rho(pos, interpolators, nfp))(positions)
    ne_all = jax.vmap(interpolators.electron_density)(rho_all)
    te_all = jax.vmap(interpolators.electron_temperature)(rho_all)

    ds = jnp.diff(ts)
    dtau = jnp.diff(optical_depths)
    alpha_interior = dtau / jnp.where(ds > 0, ds, 1.0)
    alpha_all = jnp.concatenate([alpha_interior[:1], alpha_interior])
    P_all = alpha_all * jnp.exp(-optical_depths)

    return _BeamDiagnostics(
        magnetic_field=B_all,
        rho=rho_all,
        electron_density=ne_all,
        electron_temperature=te_all,
        absorption_coefficient=alpha_all,
        linear_power_density=P_all,
    )


@jax.jit
def trace_jitted(
    position: jt.Float[jax.Array, "3"],
    direction: jt.Float[jax.Array, "3"],
    setting: ray.RaySetting,
    interpolators: Interpolators,
    nfp: int,
    tracer_settings: TracerSettings = TracerSettings(),
) -> tuple[TraceBuffers, jax.Array]:
    """Fully JIT-compiled ray trace: ODE solve + diagnostics.

    Returns ``(TraceBuffers, n_valid)`` where ``n_valid`` is the number of
    valid (finite) arc-length entries.  TraceBuffers arrays have shape
    ``(max_steps + 1,)``; entries after the event are left as ``inf`` by diffrax
    (unvisited SaveAt slots).  The caller trims to the first ``n_valid``
    entries.
    """
    sol = _solve(position, direction, setting, interpolators, nfp, tracer_settings)
    ts = cast(jax.Array, sol.ts)
    ys = cast(jax.Array, sol.ys)
    diag = _compute_beam_diagnostics(ts, ys, interpolators, nfp)
    n_valid = jnp.sum(jnp.isfinite(ts)).astype(jnp.int32)
    return (
        TraceBuffers(
            arc_length=ts,
            ode_state=ys,
            magnetic_field=diag.magnetic_field,
            normalized_effective_radius=diag.rho,
            electron_density=diag.electron_density,
            electron_temperature=diag.electron_temperature,
            absorption_coefficient=diag.absorption_coefficient,
            linear_power_density=diag.linear_power_density,
        ),
        n_valid,
    )
