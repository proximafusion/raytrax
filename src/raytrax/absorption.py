from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt
from raytrax import bessel
from raytrax import dielectric_tensor as dielectric_tensor_module
from raytrax import (
    distribution_function,
    polarization,
    power_flux,
    quantities,
)

ScalarFloat = float | jt.Float[jax.Array, " "]
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
    return jax.lax.cond(
        temperature_condition & density_condition,
        lambda: absorption_coefficient(
            refractive_index=refractive_index,
            magnetic_field=magnetic_field,
            electron_density_1e20_per_m3=electron_density_1e20_per_m3,
            electron_temperature_keV=electron_temperature_keV,
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
    """Compute the absorption coefficient.

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
    # Absorption coefficient from Fidone et al., Phys. Fluids 1988:
    #   alpha = (omega/(8*pi)) * Xp * I / |S|
    # where S is the Poynting flux and I is the resonance integral.
    #
    # We use the power flux F = 0.5 * dH/dN instead of the full Poynting flux
    # S = (c/(16*pi)) * dH/dN, giving a factor of (c/(8*pi)) difference:
    #   alpha = omega*Xp/c * I / |F|
    #
    # After accounting for Maxwellian distribution normalization, where the
    # resonance integral I includes a factor of sqrt(2*pi/mu):
    #   alpha = -omega*Xp*sqrt(mu/(2*pi))/c * resonance_integral / |F|
    #
    # The resonance_integral < 0 for absorption, giving alpha > 0.
    from scipy.constants import c as speed_of_light

    omega = 2 * jnp.pi * frequency
    Xp = (plasma_frequency / frequency) ** 2
    mu = 2 / thermal_velocity**2
    prefactor = -omega * Xp * jnp.sqrt(mu / (2 * jnp.pi)) / speed_of_light

    # Avoid division by zero or NaN when power flux is very small or invalid (near cutoff)
    power_flux_magnitude = jnp.linalg.norm(power_flux_vector)
    power_flux_threshold = 1e-10
    # Check for both small magnitude and NaN/Inf
    is_valid = (power_flux_magnitude >= power_flux_threshold) & jnp.isfinite(
        power_flux_magnitude
    )
    return jax.lax.cond(
        is_valid,
        lambda: prefactor * resonance_integral / power_flux_magnitude,
        lambda: 0.0,
    )


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
    delta_u = jnp.sqrt(n_para**2 + nY**2 - 1) / jnp.abs(denom)
    u_res = nY * n_para / denom
    u_min = u_res - delta_u
    u_max = u_res + delta_u

    # Enforce physical constraint: gamma = nY + N_para * u_para >= 1
    # This gives: u_para >= (1 - nY) / N_para (if N_para > 0)
    #         or: u_para <= (1 - nY) / N_para (if N_para < 0)
    u_gamma_limit = (1 - nY) / n_para
    u_min = jnp.where(n_para > 0, jnp.maximum(u_min, u_gamma_limit), u_min)
    u_max = jnp.where(n_para < 0, jnp.minimum(u_max, u_gamma_limit), u_max)

    # Early return if resonance region doesn't intersect bulk of distribution.
    # intersects with the bulk of the Maxwellian. Use u_cutoff = 5 * thermal_velocity
    # (corresponding to exp(-12.5) ≈ 4e-6 of the distribution peak).
    u_cutoff = 5.0 * thermal_velocity
    resonance_in_bulk = jnp.minimum(jnp.abs(u_min), jnp.abs(u_max)) < u_cutoff

    def compute_integral():
        # Define grid in u_para
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
    df_dgamma = distribution_function.maxwell_juettner_distribution_dgamma(
        lorentz_factor, thermal_velocity=thermal_velocity
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
    # normalized perpendicular momentum u_perp = p_perp / (m_0 *c)
    # the radicand cannot be negative as we checked this in `resonance_integrand`
    perpendicular_momentum = jnp.sqrt(
        lorentz_factor**2 - parallel_momentum**2 - 1,
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
    return perpendicular_momentum**2 * D.real
