"""Weakly-relativistic electron cyclotron absorption coefficient via momentum-space resonance integration."""

from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt
from scipy.constants import c as speed_of_light

from raytrax.math import bessel
from raytrax.physics import dielectric_tensor as dielectric_tensor_module
from raytrax.physics import distribution_function, polarization, power_flux, quantities
from raytrax.types import ScalarFloat

ScalarInt = int | jt.Int[jax.Array, " "]
ScalarBool = bool | jt.Bool[jax.Array, " "]


def absorption_coefficient_conditional(
    refractive_index: jt.Float[jax.Array, "3"],
    magnetic_field: jt.Float[jax.Array, "3"],
    electron_density_1e20_per_m3: ScalarFloat,
    electron_temperature_keV: ScalarFloat,
    frequency: ScalarFloat,
    mode: Literal["X", "O"],
) -> ScalarFloat:
    """Compute the absorption coefficient if the density and temperature fulfill minimum
    requirements, otherwise return zero.

    Args:
        refractive_index: Refractive index vector in the lab system.
        magnetic_field: Magnetic field vector in the lab system.
        electron_density_1e20_per_m3: Electron density in 10^20 m^-3.
        electron_temperature_keV: Electron temperature in keV.
        frequency: Frequency of the wave in Hz.
        mode: Polarization mode, either "X" for extraordinary or "O" for ordinary.

    Returns:
        Absorption coefficient alpha in m^-1.
    """
    temperature_condition = electron_temperature_keV > 0.01
    density_condition = electron_density_1e20_per_m3 > 0.0
    active = temperature_condition & density_condition

    # lax.cond VJP evaluates both branches; clamp te/ne so the true-branch VJP
    # never sees near-zero values that cause NaN in the Maxwell-Juettner formula.
    safe_te = jnp.where(active, electron_temperature_keV, 1.0)  # 1 keV fallback
    safe_ne = jnp.where(active, electron_density_1e20_per_m3, 0.1)  # 0.1e20 fallback
    return jax.lax.cond(
        active,
        lambda: absorption_coefficient(
            refractive_index=refractive_index,
            magnetic_field=magnetic_field,
            electron_density_1e20_per_m3=safe_ne,
            electron_temperature_keV=safe_te,
            frequency=frequency,
            mode=mode,
        ),
        lambda: 0.0,
    )


def absorption_coefficient(
    refractive_index: jt.Float[jax.Array, "3"],
    magnetic_field: jt.Float[jax.Array, "3"],
    electron_density_1e20_per_m3: ScalarFloat,
    electron_temperature_keV: ScalarFloat,
    frequency: ScalarFloat,
    mode: Literal["X", "O"],
) -> ScalarFloat:
    r"""Compute the absorption coefficient $\alpha$.

    Args:
        refractive_index: Refractive index vector in the lab system.
        magnetic_field: Magnetic field vector in the lab system.
        electron_density_1e20_per_m3: Electron density in 10^20 m^-3.
        electron_temperature_keV: Electron temperature in keV.
        frequency: Frequency of the wave in Hz.
        mode: Polarization mode, either "X" for extraordinary or "O" for ordinary.

    Returns:
        Absorption coefficient $\alpha$ in 1/m.
    """
    cyclotron_frequency = quantities.electron_cyclotron_frequency(
        magnetic_field_strength=jnp.linalg.norm(magnetic_field)
    )
    plasma_frequency = quantities.electron_plasma_frequency(
        electron_density_1e20_per_m3=electron_density_1e20_per_m3
    )
    tangent = magnetic_field / jnp.linalg.norm(magnetic_field)
    refractive_index_para = jnp.dot(refractive_index, tangent)
    refractive_index_perp = jnp.linalg.norm(
        refractive_index - refractive_index_para * tangent
    )
    thermal_velocity = quantities.normalized_electron_thermal_velocity(
        electron_temperature_keV=electron_temperature_keV
    )
    dielectric_tensor = dielectric_tensor_module.weakly_relativistic_dielectric_tensor(
        frequency=frequency,
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
        thermal_velocity=thermal_velocity,
        refractive_index_para=refractive_index_para,
        max_s=1,
        max_k=1,
    )
    polarization_vector = polarization.polarization(
        dielectric_tensor=dielectric_tensor,
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        frequency=frequency,
        cyclotron_frequency=cyclotron_frequency,
        mode=mode,
    )
    power_flux_vector = power_flux.power_flux_vector_stix(
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        dielectric_tensor=dielectric_tensor,
        polarization_vector=polarization_vector,
    )
    eAe = anti_hermitian_dielectric_form(
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
        frequency=frequency,
        refractive_index_para=refractive_index_para,
        refractive_index_perp=refractive_index_perp,
        thermal_velocity=thermal_velocity,
        polarization_vector=polarization_vector,
    )
    # Absorption coefficient: alpha = (omega/c) * eAe / |F|
    # where eAe = ê* · ε_r^A · ê is the anti-Hermitian dielectric form
    # and F is the power flux vector (F = 0.5 * dH/dN).
    omega = 2 * jnp.pi * frequency

    # Avoid division by zero or NaN when power flux is very small or invalid (near cutoff)
    power_flux_magnitude = jnp.linalg.norm(power_flux_vector)
    power_flux_threshold = 1e-10
    # Check for both small magnitude and NaN/Inf
    is_valid = (power_flux_magnitude >= power_flux_threshold) & jnp.isfinite(
        power_flux_magnitude
    )
    return jax.lax.cond(
        is_valid,
        lambda: omega / (2 * speed_of_light) * eAe / power_flux_magnitude,
        lambda: 0.0,
    )


def anti_hermitian_dielectric_form(
    plasma_frequency: ScalarFloat,
    cyclotron_frequency: ScalarFloat,
    frequency: ScalarFloat,
    refractive_index_para: ScalarFloat,
    refractive_index_perp: ScalarFloat,
    thermal_velocity: ScalarFloat,
    polarization_vector: jt.Complex[jax.Array, "3"],
) -> ScalarFloat:
    r"""Compute $\hat{e}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{e}$, the
    anti-Hermitian part of the relative dielectric tensor contracted with the
    polarization vector.

    This is the numerator quantity in the absorption coefficient formula,

    .. math::
        \alpha = \frac{\omega}{c} \frac{\hat{e}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{e}}{|F|}

    where :math:`F` is the power flux vector.  The result is positive for an absorbing
    plasma.

    The anti-Hermitian part is computed via the fully relativistic momentum-space
    resonance integral (Bornatici et al. 1983), summed over cyclotron harmonics:

    .. math::
        \hat{e}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{e}
        = -X_p \cdot 4\pi^2 \sum_s I_s

    where :math:`X_p = (\omega_p/\omega)^2` and :math:`I_s` is the resonance integral
    for harmonic :math:`s` (negative for absorption).

    Args:
        plasma_frequency: Electron plasma frequency in Hz.
        cyclotron_frequency: Electron cyclotron frequency in Hz.
        frequency: Wave frequency in Hz.
        refractive_index_para: Refractive index parallel to the magnetic field.
        refractive_index_perp: Refractive index perpendicular to the magnetic field.
        thermal_velocity: Normalized electron thermal velocity (v_th / c).
        polarization_vector: Normalized polarization vector in Stix coordinates.

    Returns:
        The scalar quadratic form :math:`\hat{e}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{e}` (dimensionless).
    """  # noqa: E501
    resonance_integral = 0.0
    # TODO extend to higher harmonics - currently only 1st and 2nd harmonic
    for harmonic_index in range(1, 3):
        resonance_integral += jax.lax.cond(
            # The resonance condition is given by
            # gamma = nY + n_para * u_para
            # where nY = harmonic_index * cyclotron_frequency / frequency
            # Also, from p = m_0 * c * gamma one can derive that
            # gamma^2 = 1 + u_para^2 + u_perp^2
            # consequently, the resonance condition can be written as the equation
            # for a circle in the u_perp, u_para plane:
            # 1 + u_para^2 + u_perp^2 = n_para^2 + (nY)^2
            # the left and right boundaries of this circle are obtained by setting
            # u_perp = 0 and solving for u_para. If the discriminant of this
            # quadratic equation is negative, the resonance condition cannot be
            # satisfied anywhere in the plane, and the integral is zero.
            (
                refractive_index_para**2
                + (harmonic_index * cyclotron_frequency / frequency) ** 2
                - 1
            )
            < 0,
            lambda: 0.0,
            lambda: compute_resonance_integral(
                harmonic_index=harmonic_index,
                cyclotron_frequency=cyclotron_frequency,
                frequency=frequency,
                refractive_index_para=refractive_index_para,
                refractive_index_perp=refractive_index_perp,
                polarization_vector=polarization_vector,
                thermal_velocity=thermal_velocity,
            ),
        )
    Xp = (plasma_frequency / frequency) ** 2
    return -Xp * 8 * jnp.pi**2 * resonance_integral


def compute_resonance_integral(
    harmonic_index: ScalarInt,
    cyclotron_frequency: ScalarFloat,
    frequency: ScalarFloat,
    refractive_index_para: ScalarFloat,
    refractive_index_perp: ScalarFloat,
    polarization_vector: jt.Complex[jax.Array, "3"],
    thermal_velocity: ScalarFloat,
) -> ScalarFloat:
    r"""Compute the resonance integral for the given harmonic.

    The resonance integral is defined as:
    .. math::
        I = \int_{u_{min}}^{u_{max}} du_\parallel D_{ql} (\frac{\partial f_e}{\partial \gamma} + n_\parallel \frac{\partial f_e}{\partial u_\parallel})

    where :math:`D_{ql}` is the quasilinear diffusion coefficient, :math:`f_e` is the
    electron distribution function, :math:`n_\parallel` is the refractive index parallel
    to the magnetic field, and :math:`u_\parallel` is the normalized parallel momentum
    (:math:`u_\parallel = p_\parallel / (m_0 c)`).

    See eq. (4) of Marushchenko, Maassberg, and Turkin,
    https://doi.org/10.1088/0029-5515/48/5/054002

    Args:
        harmonic_index: Harmonic index (integer).
        cyclotron_frequency: Electron cyclotron frequency in Hz.
        frequency: Wave frequency in Hz.
        refractive_index_para: Refractive index parallel to the magnetic field.
        refractive_index_perp: Refractive index perpendicular to the magnetic field.
        polarization_vector: Normalized polarization vector in Stix coordinates.
        thermal_velocity: Normalized electron thermal velocity (v_th / c).
    """  # noqa: E501
    # Determine the integration bounds
    # See above for the description of the resonance condition as a circle in
    # the u_perp, u_para plane. This determines u_min and u_max as the left and right
    # boundaries of the circle.
    nY = harmonic_index * cyclotron_frequency / frequency
    n_para = refractive_index_para
    denom = 1 - n_para**2
    # 1e-30 floor: at disc=0 (exact 2nd-harmonic resonance), sqrt'(0)=Inf causes
    # 0*Inf=NaN in the adjoint.  The floor keeps the gradient finite.
    # jnp.where zeros delta_u for disc<0; the VJP of the enclosing lax.cond
    # evaluates this code even for the non-resonant branch.
    disc = n_para**2 + nY**2 - 1
    delta_u = jnp.sqrt(jnp.maximum(1e-30, disc)) / jnp.maximum(jnp.abs(denom), 1e-10)
    delta_u = jnp.where(disc < 0, 0.0, delta_u)
    u_res = nY * n_para / denom
    u_min = u_res - delta_u
    u_max = u_res + delta_u

    # No u_gamma_limit = (1-nY)/n_para enforcement: its gradient ~1/n_para²
    # diverges catastrophically when n_para≈0.  The γ²-u²-1<0 guard inside
    # resonance_integrand already zeros the integrand for any u with γ<1.

    # Early return if resonance region doesn't intersect bulk of distribution.
    # intersects with the bulk of the Maxwellian. Use u_cutoff = 5 * thermal_velocity
    # (corresponding to exp(-12.5) ≈ 4e-6 of the distribution peak).
    u_cutoff = 5.0 * thermal_velocity
    resonance_in_bulk = jnp.minimum(jnp.abs(u_min), jnp.abs(u_max)) < u_cutoff

    def compute_integral() -> ScalarFloat:
        # K2_scaled = kve(2, mu) depends only on thermal_velocity, which is
        # constant across all 1000 vmap iterations.  Pre-compute it here so
        # it is evaluated exactly once rather than once per grid point.
        mu = 2 / thermal_velocity**2
        K2_scaled = bessel.kve_jax(2, mu)

        # Grid resolution for the trapezoidal rule.
        # The integrand is peaked on the resonance curve with a 1/e-folding width
        # in u_para of ~ 1 / (mu * n_para), where mu = m_e c^2 / T_e = 2 / vth^2.
        # The integration range is set by the resonance circle geometry and scales
        # as ~ sqrt(n_para^2 + nY^2 - 1) / (1 - n_para^2).  Requiring ~5 points
        # per 1/e-folding gives N_min ~ 3 * mu * n_para^2 / (1 - n_para^2).
        #
        # N=1000 covers the physically relevant ECRH parameter space:
        #   T >= 0.5 keV (mu <= 770),  n_para <= 0.50  ->  N_min <= 770   [safe]
        #   T >= 0.1 keV (mu <= 3100), n_para <= 0.25  ->  N_min <= 640   [safe]
        # At T < 0.5 keV and n_para > 0.3 the integrand is more narrowly peaked,
        # but the absorption coefficient is also exponentially suppressed (sub-thermal
        # resonance), so the absolute error on alpha remains negligible.
        u_grid = jnp.linspace(u_min, u_max, 1000)

        # Compute the integrand values
        integrand_values = jax.vmap(
            lambda u: resonance_integrand(
                harmonic_index=harmonic_index,
                cyclotron_frequency=cyclotron_frequency,
                frequency=frequency,
                refractive_index_para=refractive_index_para,
                refractive_index_perp=refractive_index_perp,
                parallel_momentum=u,
                thermal_velocity=thermal_velocity,
                K2_scaled=K2_scaled,
                polarization_vector=polarization_vector,
            )
        )(u_grid)
        # Integrate using the trapezoidal rule
        return jnp.trapezoid(integrand_values, u_grid)

    return jax.lax.cond(resonance_in_bulk, compute_integral, lambda: 0.0)


def resonance_integrand(
    harmonic_index: ScalarInt,
    cyclotron_frequency: ScalarFloat,
    frequency: ScalarFloat,
    refractive_index_para: ScalarFloat,
    refractive_index_perp: ScalarFloat,
    parallel_momentum: ScalarFloat,
    thermal_velocity: ScalarFloat,
    K2_scaled: ScalarFloat,
    polarization_vector: jt.Complex[jax.Array, "3"],
) -> ScalarFloat:
    """Compute the integrand of the resonance integral.

    Args:
        harmonic_index: Harmonic index (integer).
        cyclotron_frequency: Electron cyclotron frequency in Hz.
        frequency: Wave frequency in Hz.
        refractive_index_para: Refractive index parallel to the magnetic field.
        refractive_index_perp: Refractive index perpendicular to the magnetic field.
        parallel_momentum: Normalized parallel momentum (p_para / (m_0 c)).
        thermal_velocity: Normalized electron thermal velocity (v_th / c).
        K2_scaled: Pre-computed kve(2, mu) = K_2(mu) * exp(mu), where mu = 2/v_th^2.
        polarization_vector: Normalized polarization vector in Stix coordinates.
    """
    # Fix the Lorentz factor from the resonance condition
    # gamma - n * w / w_c - n * u_para = 0
    lorentz_factor = (
        harmonic_index * cyclotron_frequency / frequency
        + refractive_index_para * parallel_momentum
    )
    return jax.lax.cond(
        # If this is negative, no solution in the u_para, u_perp plane exists
        # as u_perp^2 is always positive
        lorentz_factor**2 - parallel_momentum**2 - 1 < 0,
        lambda: 0.0,
        lambda: _resonance_integrand_full(
            harmonic_index=harmonic_index,
            cyclotron_frequency=cyclotron_frequency,
            frequency=frequency,
            refractive_index_perp=refractive_index_perp,
            refractive_index_para=refractive_index_para,
            parallel_momentum=parallel_momentum,
            lorentz_factor=lorentz_factor,
            thermal_velocity=thermal_velocity,
            K2_scaled=K2_scaled,
            polarization_vector=polarization_vector,
        ),
    )


def _resonance_integrand_full(
    harmonic_index: ScalarInt,
    cyclotron_frequency: ScalarFloat,
    frequency: ScalarFloat,
    refractive_index_para: ScalarFloat,
    refractive_index_perp: ScalarFloat,
    parallel_momentum: ScalarFloat,
    lorentz_factor: ScalarFloat,
    thermal_velocity: ScalarFloat,
    K2_scaled: ScalarFloat,
    polarization_vector: jt.Complex[jax.Array, "3"],
) -> ScalarFloat:
    """Compute the resonance integrand in the case where it does not vanish.

    This auxiliary function is needed for the `jax.lax.cond` condition in
    `resonance_integrand` to avoid computing the quasilinear diffusion coefficient
    when the resonance condition is not satisfied.
    """
    Dql = quasilinear_diffusion_coefficient(
        harmonic_index=harmonic_index,
        cyclotron_frequency=cyclotron_frequency,
        frequency=frequency,
        lorentz_factor=lorentz_factor,
        refractive_index_perp=refractive_index_perp,
        parallel_momentum=parallel_momentum,
        polarization_vector=polarization_vector,
    )
    # FIXME hard-coded Maxwellian for now
    df_dgamma = distribution_function.maxwell_juettner_distribution_dgamma_precomputed(
        lorentz_factor, thermal_velocity=thermal_velocity, K2_scaled=K2_scaled
    )
    # Fidone operator: Lf = df/dw + N_par * df/dq_par|_w
    # For an isotropic Maxwellian f(gamma), df/dq_par|_w = 0 (energy is constant),
    # so Lf = df/dgamma.
    return Dql * df_dgamma


def quasilinear_diffusion_coefficient(
    harmonic_index: ScalarInt,
    cyclotron_frequency: ScalarFloat,
    frequency: ScalarFloat,
    refractive_index_perp: ScalarFloat,
    lorentz_factor: ScalarFloat,
    parallel_momentum: ScalarFloat,
    polarization_vector: jt.Complex[jax.Array, "3"],
) -> ScalarFloat:
    """Compute the quasilinear diffusion coefficient Dql.

    The quasilinear diffusion coefficient is given in
    Marushchenko, Maassberg, and Turkin,
    https://doi.org/10.1088/0029-5515/48/5/054002, and was originally published in
    Fidone, Granata and Johner, https://inis.iaea.org/records/6av5q-vn438

    Args:
        harmonic_index: Harmonic index (integer).
        cyclotron_frequency: Electron cyclotron frequency in Hz.
        frequency: Wave frequency in Hz.
        refractive_index_perp: Refractive index perpendicular to the magnetic field.
        lorentz_factor: Lorentz factor of the electrons.
        parallel_momentum: Normalized parallel momentum (p_para / (m_0 c)).
        polarization_vector: Normalized polarization vector in Stix coordinates.

    Returns:
        The quasilinear diffusion coefficient Dql.
    """
    # u_perp = p_perp / (m_0 c).  Two forms to avoid 0*Inf=NaN in the adjoint:
    #   perp_mom_sq       = max(0, x)       → gradient=0 at boundary (used in return)
    #   perpendicular_momentum = sqrt(max(1e-30, x)) → finite gradient (used in kperp_rho)
    perp_mom_sq = jnp.maximum(0.0, lorentz_factor**2 - parallel_momentum**2 - 1)
    perpendicular_momentum = jnp.sqrt(
        jnp.maximum(1e-30, lorentz_factor**2 - parallel_momentum**2 - 1)
    )

    # this is the perpendicular wave number times the electron Larmor radius
    # k_perp = n_perp * omega / c
    # rho_e = p_perp / (e * B) = p_perp / (omega_c * m_0)
    # p_perp = u_perp * (m_0 * c)
    kperp_rho = (
        refractive_index_perp * perpendicular_momentum * frequency / cyclotron_frequency
    )

    # Safety check for small kperp_rho to avoid division by zero
    # For small x, Jn(x)/x = (x/2)^(n-1) / (2 * n!) for n >= 1
    kperp_rho_threshold = 1e-10
    kperp_rho_safe = jnp.where(
        kperp_rho < kperp_rho_threshold, kperp_rho_threshold, kperp_rho
    )

    Jn = bessel.jv_jax(harmonic_index, kperp_rho_safe)
    dJn = bessel.djv_jax(harmonic_index, kperp_rho_safe)

    An1 = harmonic_index * Jn / kperp_rho_safe
    An2 = 1j * dJn
    # Avoid division by zero when perpendicular_momentum is very small
    perp_safe = jnp.where(
        perpendicular_momentum < kperp_rho_threshold, 1.0, perpendicular_momentum
    )
    An3 = (parallel_momentum / perp_safe) * Jn
    An3 = jnp.where(perpendicular_momentum < kperp_rho_threshold, 0.0, An3)

    An = jnp.array(
        [
            [An1 * An1, An1 * An2, An1 * An3],
            [-An1 * An2, -An2 * An2, -An2 * An3],
            [An1 * An3, An2 * An3, An3 * An3],
        ],
        dtype=jnp.complex128,
    )

    # evaluate Hermitian form: pe^H * An * pe
    # vdot does conjugation on first argument
    D = jnp.vdot(polarization_vector, An @ polarization_vector)
    return perp_mom_sq * D.real
