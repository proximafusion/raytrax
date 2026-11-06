"""JIT-compiled ray tracing ODE solver: integrates position, refractive index, optical depth and arc length via diffrax."""

from collections.abc import Callable
from typing import NamedTuple, cast

import diffrax
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import optimistix as optx

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
        B_cyl = interpolators.magnetic_field(r, z)
    else:
        phi_mapped, z_query, in_second_half = _map_to_fundamental_domain(phi, z, nfp)
        B_cyl = interpolators.magnetic_field(r, phi_mapped, z_query)
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
        return interpolators.rho(r, z)
    phi = jnp.arctan2(position[1], position[0])
    phi_mapped, z_query, _ = _map_to_fundamental_domain(phi, z, nfp)
    return interpolators.rho(r, phi_mapped, z_query)


def _y_to_state(y: jt.Float[jax.Array, " n "]) -> ray.RayState:
    """Extract RayState from the 8-component ODE state vector.

    State vector structure (8 components):
    - y[0:3]: position
    - y[3:6]: refractive_index
    - y[6]:   optical_depth
    - y[7]:   arc_length
    """
    return ray.RayState(
        position=y[:3],
        refractive_index=y[3:6],
        optical_depth=y[6],
        arc_length=y[7],
    )


# diffrax traces the RHS several times per solve (shape check and RK stages); a
# nested jit traces it once and reuses the jaxpr.
@jax.jit
def _right_hand_side(
    t: float | int | jax.Array,
    y: jt.Float[jax.Array, " n "],
    args: tuple,
) -> jt.Float[jax.Array, " n "]:
    r"""Compute the right-hand side of the 8-component ray tracing ODE.

    The independent variable is the Hamiltonian time t: dr/dt = ∂H/∂N,
    dN/dt = -∂H/∂r, dτ/dt = α |∂H/∂N|, ds/dt = |∂H/∂N|. Unlike the parametrisation
    by the arc length s, which divides by |∂H/∂N|, this one stays regular where a
    ray is reflected: there the group velocity, and with it ∂H/∂N, goes through
    zero, r(t) turns around smoothly and the path has a cusp, at which s is not a
    smooth coordinate. The arc length rides along as the last component: it is the
    coordinate of the output profiles and of the `max_arc_length` stop, but it takes
    no part in the step-size control (see `_error_norm`).

    Integrates (r, N, τ, s) only. B, ρ, nₑ, Tₑ are not tracked in the ODE
    state — they are recomputed at output points via vectorised post-processing,
    which avoids the jacfwd(B) calls and the tight per-component tolerances that
    previously forced ~2 mm step sizes.

    Uses diffrax's ``(t, y, args)`` calling convention. The args tuple is
    ``(setting, interpolators, nfp, tracer_settings)`` where ``interpolators`` is an
    :class:`Interpolators` pytree.
    """
    setting, interpolators, nfp, _ = args

    state = _y_to_state(y)

    def eval_B(pos):
        return _eval_magnetic_field(pos, interpolators, nfp)

    def eval_rho(pos):
        return _eval_rho(pos, interpolators, nfp)

    # Compute H and both gradients in a single pass,
    # reusing B-field, rho, and ne from the forward pass (has_aux=True).
    (_, hamiltonian_aux), (hamiltonian_gradient_r, hamiltonian_gradient_n) = (
        hamiltonian.hamiltonian_value_and_gradients(
            state.position,
            state.refractive_index,
            eval_B,
            eval_rho,
            interpolators.electron_density,
            setting.frequency,
            setting.mode,
        )
    )
    ds_dt = jnp.linalg.norm(hamiltonian_gradient_n)

    te = interpolators.electron_temperature(hamiltonian_aux.rho)
    dtau_ds = absorption.absorption_coefficient_conditional(
        refractive_index=state.refractive_index,
        magnetic_field=hamiltonian_aux.magnetic_field,
        electron_density_1e20_per_m3=hamiltonian_aux.electron_density_1e20_per_m3,
        electron_temperature_keV=te,
        frequency=setting.frequency,
        mode=setting.mode,
        max_harmonic=setting.max_harmonic,
    )

    return jnp.concatenate(
        [
            hamiltonian_gradient_n,
            -hamiltonian_gradient_r,
            jnp.array([dtau_ds * ds_dt, ds_dt]),
        ]
    )


class _HamiltonianScales(NamedTuple):
    """H at an ODE state with the scales the step-size control needs."""

    value: jt.Float[jax.Array, ""]
    """H."""
    frequency_scale: jt.Float[jax.Array, ""]
    """ω ∂H/∂ω at fixed wave vector. H / frequency_scale is the relative frequency
    mismatch δω/ω by which the wave is off the dispersion surface."""
    velocity: jt.Float[jax.Array, "3"]
    """∂H/∂N = dr/dt, the velocity of the ray in Hamiltonian time; its norm is the
    arc length per unit of Hamiltonian time."""


def _hamiltonian_scales(
    y: jt.Float[jax.Array, " n "], args: tuple
) -> _HamiltonianScales:
    """Evaluate H, ω ∂H/∂ω at fixed wave vector and ∂H/∂N at an ODE state."""
    setting, interpolators, nfp, _ = args
    frequency = jnp.asarray(setting.frequency, dtype=float)

    def h(refractive_index, frequency):
        return hamiltonian.hamiltonian(
            y[:3],
            refractive_index,
            lambda pos: _eval_magnetic_field(pos, interpolators, nfp),
            lambda pos: _eval_rho(pos, interpolators, nfp),
            interpolators.electron_density,
            frequency,
            setting.mode,
        )[0]

    value, (gradient_n, dh_dfrequency) = jax.value_and_grad(h, argnums=(0, 1))(
        y[3:6], frequency
    )
    # N = c k / ω: at fixed k, N scales with 1 / ω as well.
    frequency_scale = frequency * dh_dfrequency - jnp.dot(y[3:6], gradient_n)
    return _HamiltonianScales(value, frequency_scale, gradient_n)


def _error_norm(scaled_error: jt.Float[jax.Array, " n "]) -> jt.Float[jax.Array, ""]:
    """RMS norm of the scaled error estimate over (r, N, τ), leaving out the arc length.

    The arc length is an output, not a dynamical variable: its error does not feed
    back into the ray, and keeping it out of the norm leaves the step sequence the
    same as that of the state without it.
    """
    return optx.rms_norm(scaled_error[:7])


class _DispersionAwareController(diffrax.AbstractAdaptiveStepSizeController):
    """PID control on an error estimate that also includes the change of H.

    H is conserved along exact rays. A step that jumps over a thin layer, such as the
    density ramp at the plasma edge, puts no Runge-Kutta stage inside it, so the
    embedded error estimate misses the layer while H changes. The change of H over
    the step is converted to the relative frequency mismatch
    |ΔH| / |ω ∂H/∂ω| it amounts to, which does not depend on how H is written or
    parametrised, weighted and added to the error of N.

    The ODE runs in Hamiltonian time, while the step limits are arc lengths in
    metres: the initial step and the maximum step are divided by ds/dt = |∂H/∂N|.
    Where a ray is reflected ds/dt goes through zero, so the step predicted from
    ds/dt at the start of a step can turn out longer than the maximum; a step whose
    arc length (the increment of the integrated s) exceeds it by more than half is
    rejected and shortened in proportion.
    """

    pid: diffrax.PIDController
    weight: jt.Float[jax.Array, ""]
    max_step_size: jt.Float[jax.Array, ""]
    # Required by AbstractAdaptiveStepSizeController; copied from the PID controller.
    rtol: int | float | jax.Array | np.ndarray
    atol: int | float | jax.Array | np.ndarray
    norm: Callable

    def __init__(
        self,
        pid: diffrax.PIDController,
        weight: jt.Float[jax.Array, ""],
        max_step_size: jt.Float[jax.Array, ""],
    ):
        self.pid = pid
        self.weight = weight
        self.max_step_size = max_step_size
        self.rtol = pid.rtol
        self.atol = pid.atol
        self.norm = pid.norm

    def wrap(self, direction):
        return _DispersionAwareController(
            self.pid.wrap(direction), self.weight, self.max_step_size
        )

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        scales = _hamiltonian_scales(y0, args)
        t1_next, pid_state = self.pid.init(
            terms,
            t0,
            t1,
            y0,
            dt0 / jnp.linalg.norm(scales.velocity),
            args,
            func,
            error_order,
        )
        # H and ∂H/∂N at the start of the next step; reused until a step is accepted.
        return t1_next, (pid_state, scales)

    def adapt_step_size(
        self, t0, t1, y0, y1_candidate, args, y_error, error_order, controller_state
    ):
        assert y_error is not None  # Bosh3 provides an error estimate
        pid_state, scales_0 = controller_state
        scales = _hamiltonian_scales(y1_candidate, args)
        frequency_error = (
            self.weight
            * jnp.abs(scales.value - scales_0.value)
            / jnp.abs(scales.frequency_scale)
        )
        # The PID scales each component by its own tolerance. Rescale so that the
        # added relative error is measured against the tolerance of a quantity of
        # order one, and add it to the first component of N.
        tolerance_unit = self.pid.atol + self.pid.rtol
        tolerance_0 = self.pid.atol + self.pid.rtol * jnp.maximum(
            jnp.abs(y0[3]), jnp.abs(y1_candidate[3])
        )
        y_error = y_error.at[3].set(
            jnp.hypot(y_error[3], frequency_error * tolerance_0 / tolerance_unit)
        )
        keep_step, t0_next, t1_next, made_jump, pid_state, result = (
            self.pid.adapt_step_size(
                t0, t1, y0, y1_candidate, args, y_error, error_order, pid_state
            )
        )
        arc_length = y1_candidate[7] - y0[7]
        too_long = arc_length > 1.5 * self.max_step_size
        keep_step = keep_step & ~too_long
        t0_next = jnp.where(too_long, t0, t0_next)
        t1_next = jnp.where(
            too_long,
            jnp.minimum(t1_next, t0 + (t1 - t0) * self.max_step_size / arc_length),
            jnp.minimum(
                t1_next,
                t0_next + self.max_step_size / jnp.linalg.norm(scales.velocity),
            ),
        )
        scales_0 = jax.tree_util.tree_map(
            lambda new, old: jnp.where(keep_step, new, old), scales, scales_0
        )
        return keep_step, t0_next, t1_next, made_jump, (pid_state, scales_0), result


def _cond_terminate(t, y, args, **kwargs):
    """Terminate when this falls through zero, i.e. when the first of these happens.

    - the ray exits the plasma (rho > 1.05),
    - the ray is fully absorbed (exp(-tau) < 1e-3),
    - the ray leaves the computational domain (|r| > 20 m),
    - the arc length reaches TracerSettings.max_arc_length; the ray is then
      incomplete, see `trace_jitted`.
    """
    return jnp.minimum(
        jnp.minimum(
            1.05 - _eval_rho(y[:3], args[1], args[2]),
            jnp.exp(-y[6]) - 1e-3,
        ),
        jnp.minimum(
            20.0 - jnp.linalg.norm(y[:3]),
            args[3].max_arc_length - y[7],
        ),
    )


# Resolution of the arc-length stop: the root finder below places the end of the ray
# within 1e-9 units of Hamiltonian time of max_arc_length, i.e. within nanometres. A
# ray whose final arc length is within this tolerance of the cap was stopped by it.
_ARC_LENGTH_TOLERANCE = 1e-6

# The root find stops the ray where the condition crosses zero rather than at the end
# of the step that crossed it, so the end of the ray does not jump with the step
# placement. Bisection on the step's bracket always converges; Newton started at the
# end of the step can diverge when rho is not monotonic along a long vacuum step.
_event = diffrax.Event(
    cond_fn=_cond_terminate,
    # mypy takes Bisection.norm, a ClassVar, for a constructor argument.
    root_finder=optx.Bisection(rtol=1e-9, atol=1e-9),  # type: ignore[call-arg]
    direction=False,
)

_term = diffrax.ODETerm(_right_hand_side)  # type: ignore[arg-type]
# Third order on purpose: the linearly interpolated equilibrium makes the right-hand
# side only piecewise smooth, so higher-order tableaus lose their order and
# Bosh3 reaches a given accuracy with the fewest right-hand-side evaluations.
_solver = diffrax.Bosh3()


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
    automatically: the Hamiltonian switches to _hamiltonian_vacuum when ne=0,
    giving straight-line propagation until the beam enters plasma.

    The independent variable is the Hamiltonian time, which has no bound of its
    own: the solve ends at an event, on the termination conditions or at the arc
    length max_arc_length, or when the step budget max_steps runs out. Output is
    saved at every accepted step (plus t0). After the event, diffrax leaves
    unvisited SaveAt slots as inf.
    """
    # PI rather than diffrax's default I controller: on the non-smooth right-hand side
    # the I controller rejects far more steps and is less accurate per evaluation.
    stepsize_controller = _DispersionAwareController(
        diffrax.PIDController(
            rtol=tracer_settings.relative_tolerance,
            atol=tracer_settings.absolute_tolerance,
            norm=_error_norm,
            # Allow progress across residual jumps of H at micrometre scale.
            dtmin=1e-6,
            pcoeff=0.4,
            icoeff=0.3,
        ),
        jnp.asarray(tracer_settings.dispersion_error_weight),
        jnp.asarray(tracer_settings.max_step_size),
    )
    y0 = jnp.concatenate([position, direction, jnp.array([0.0, 0.0])])
    return diffrax.diffeqsolve(
        terms=_term,
        solver=_solver,
        t0=0.0,
        t1=jnp.inf,
        dt0=0.001,  # metres; the controller converts to Hamiltonian time
        y0=y0,
        args=(setting, interpolators, nfp, tracer_settings),
        saveat=diffrax.SaveAt(steps=True, t0=True),
        stepsize_controller=stepsize_controller,
        event=_event,
        max_steps=tracer_settings.max_steps,
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
    ys: jt.Float[jax.Array, "nsteps 8"],
    interpolators: Interpolators,
    nfp: int,
) -> _BeamDiagnostics:
    """Evaluate plasma quantities along the trajectory and derive absorption/power.

    The absorption coefficient of a segment is Δτ/Δs between consecutive saved
    steps. Entries in the inf padding after the end of the ray stay non-finite.
    """
    positions = ys[:, :3]
    optical_depths = ys[:, 6]
    arc_lengths = ys[:, 7]

    B_all = jax.vmap(lambda pos: _eval_magnetic_field(pos, interpolators, nfp))(
        positions
    )
    rho_all = jax.vmap(lambda pos: _eval_rho(pos, interpolators, nfp))(positions)
    ne_all = jax.vmap(interpolators.electron_density)(rho_all)
    te_all = jax.vmap(interpolators.electron_temperature)(rho_all)

    ds = jnp.diff(arc_lengths)
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
    ``(tracer_settings.max_steps + 1,)``; entries after the event are left as
    ``inf`` by diffrax (unvisited SaveAt slots).  The caller trims to the first
    ``n_valid`` entries.
    """
    sol = _solve(position, direction, setting, interpolators, nfp, tracer_settings)
    ts = cast(jax.Array, sol.ts)
    ys = cast(jax.Array, sol.ys)
    diag = _compute_beam_diagnostics(ys, interpolators, nfp)
    n_valid = jnp.sum(jnp.isfinite(ts)).astype(jnp.int32)
    # Complete only if the ray left the plasma, was absorbed or left the domain. The
    # arc-length cap is a cutoff like the step budget: a ray stopped there, which the
    # root find ends at the cap itself, is incomplete.
    s_end = jnp.max(jnp.where(jnp.isfinite(ys[:, 7]), ys[:, 7], -jnp.inf))
    reached_max_arc_length = (
        s_end >= tracer_settings.max_arc_length - _ARC_LENGTH_TOLERANCE
    )
    incomplete = (sol.result != diffrax.RESULTS.event_occurred) | reached_max_arc_length
    return (
        TraceBuffers(
            arc_length=ys[:, 7],
            ode_state=ys,
            magnetic_field=diag.magnetic_field,
            normalized_effective_radius=diag.rho,
            electron_density=diag.electron_density,
            electron_temperature=diag.electron_temperature,
            absorption_coefficient=diag.absorption_coefficient,
            linear_power_density=diag.linear_power_density,
            incomplete=jnp.asarray(incomplete),
        ),
        n_valid,
    )
