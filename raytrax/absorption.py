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
    # TODO(dstraub): implement higher harmonics
    for harmonic_index in range(1):
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
            <= 0,
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
    prefactor = (2 * jnp.pi * frequency) / 8
    return prefactor * resonance_integral / jnp.linalg.norm(power_flux_vector)


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
    # determine the integration bounds
    # see above for the description of the resonance condition as a circle in
    # the u_perp, u_para plane. This determins u_min and u_max as the left and right
    # boundaries of the circle.
    nY = harmonic_index * cyclotron_frequency / frequency
    n_para = refractive_index_para
    denom = 1 - n_para**2
    delta_u = jnp.sqrt(n_para**2 + nY**2 - 1) / jnp.abs(denom)
    u_res = nY * n_para / denom
    u_min = u_res - delta_u
    u_max = u_res + delta_u

    # define grid in u_para
    u_grid = jnp.linspace(u_min, u_max, 1000)

    # compute the integrand values
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

    # integrate using the trapezoidal rule
    return jnp.trapezoid(integrand_values, u_grid)


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
    df_dupara = 0.0
    return Dql * (df_dgamma + refractive_index_para * df_dupara)


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

    Jn = bessel.jv_jax(harmonic_index, kperp_rho)
    dJn = bessel.djv_jax(harmonic_index, kperp_rho)

    An1 = harmonic_index * Jn / kperp_rho
    An2 = 1j * dJn
    An3 = (parallel_momentum / perpendicular_momentum) * Jn

    An = jnp.array(
        [
            [An1 * An1, An1 * An2, An1 * An3],
            [-An1 * An2, -An2 * An2, -An2 * An3],
            [An1 * An3, -An2 * An3, An3 * An3],
        ],
        dtype=jnp.complex128,
    )

    # evaluate Hermitian form: pe^H * An * pe
    # vdot does conjugation on first argument
    D = jnp.vdot(polarization_vector, An @ polarization_vector)
    return perpendicular_momentum**2 * D.real
