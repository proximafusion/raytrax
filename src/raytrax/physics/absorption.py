"""Relativistic electron cyclotron absorption coefficient via momentum-space resonance integration."""

import math
from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from scipy.constants import c as speed_of_light
from scipy.special import lambertw

from raytrax.math import bessel
from raytrax.physics import dielectric_tensor as dielectric_tensor_module
from raytrax.physics import distribution_function, polarization, power_flux, quantities
from raytrax.types import ScalarFloat

ScalarBool = bool | jt.Bool[jax.Array, " "]


def absorption_coefficient_conditional(
    refractive_index: jt.Float[jax.Array, "3"],
    magnetic_field: jt.Float[jax.Array, "3"],
    electron_density_1e20_per_m3: ScalarFloat,
    electron_temperature_keV: ScalarFloat,
    frequency: ScalarFloat,
    mode: Literal["X", "O"],
    max_harmonic: int = 2,
    resonance_rtol: float = 1e-14,
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
        max_harmonic: Highest cyclotron harmonic to include in the absorption
            calculation. Determines the KO tensor FLR order and the number of
            resonance integrals evaluated. Must be a Python integer (compile-time
            constant for JIT).
        resonance_rtol: Approximate relative accuracy of each resonance integral;
            sets the integration window and the number of quadrature nodes. Must be
            a Python float (compile-time constant for JIT).

    Returns:
        Absorption coefficient alpha in m^-1.
    """
    temperature_condition = electron_temperature_keV > 0.01
    density_condition = electron_density_1e20_per_m3 > 0.0
    # Off resonance alpha is exactly zero; deciding that from a few scalars skips
    # the dielectric tensor, polarization and power flux, most of the cost.
    active = (
        temperature_condition
        & density_condition
        & _any_harmonic_resonant(
            refractive_index,
            magnetic_field,
            electron_temperature_keV,
            frequency,
            max_harmonic,
        )
    )

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
            max_harmonic=max_harmonic,
            resonance_rtol=resonance_rtol,
        ),
        lambda: 0.0,
    )


def _resonance_circle(
    nY: ScalarFloat, n_para: ScalarFloat, thermal_velocity: ScalarFloat
) -> tuple[ScalarFloat, ScalarFloat, ScalarBool]:
    """Ends of the resonance circle in u_∥ and whether it reaches the bulk of the distribution."""
    denom = 1 - n_para**2
    # 1e-30 floor: at disc=0 (exact 2nd-harmonic resonance), sqrt'(0)=Inf causes
    # 0*Inf=NaN in the adjoint.  The floor keeps the gradient finite.
    # jnp.where zeros delta_u for disc<0; the VJP of the enclosing lax.cond
    # evaluates this code even for the non-resonant branch.
    disc = n_para**2 + nY**2 - 1
    delta_u = jnp.sqrt(jnp.maximum(1e-30, disc)) / jnp.maximum(jnp.abs(denom), 1e-10)
    delta_u = jnp.where(disc < 0, 0.0, delta_u)
    u_res = nY * n_para / denom
    # Use u_cutoff = 7 * thermal_velocity (exp(-24.5) ≈ 2e-11 of distribution peak).
    u_cutoff = 7.0 * thermal_velocity
    # Use the unclamped circle so that wide circles that straddle u_∥ = 0 are not
    # mistakenly rejected.
    u_min = u_res - delta_u
    u_max = u_res + delta_u
    in_bulk = (disc >= 0) & (jnp.minimum(jnp.abs(u_min), jnp.abs(u_max)) < u_cutoff)
    return u_min, u_max, in_bulk


def _any_harmonic_resonant(
    refractive_index: jt.Float[jax.Array, "3"],
    magnetic_field: jt.Float[jax.Array, "3"],
    electron_temperature_keV: ScalarFloat,
    frequency: ScalarFloat,
    max_harmonic: int,
) -> ScalarBool:
    """Whether any harmonic's resonance reaches the bulk; if none does, alpha is zero."""
    b_norm = jnp.linalg.norm(magnetic_field)
    cyclotron_frequency = quantities.electron_cyclotron_frequency(
        magnetic_field_strength=b_norm
    )
    n_para = jnp.dot(refractive_index, magnetic_field) / b_norm
    thermal_velocity = quantities.normalized_electron_thermal_velocity(
        electron_temperature_keV=jnp.maximum(electron_temperature_keV, 0.0)
    )
    resonant = jnp.array(False)
    for harmonic_index in range(1, max_harmonic + 1):
        nY = harmonic_index * cyclotron_frequency / frequency
        resonant = resonant | _resonance_circle(nY, n_para, thermal_velocity)[2]
    return resonant


def absorption_coefficient(
    refractive_index: jt.Float[jax.Array, "3"],
    magnetic_field: jt.Float[jax.Array, "3"],
    electron_density_1e20_per_m3: ScalarFloat,
    electron_temperature_keV: ScalarFloat,
    frequency: ScalarFloat,
    mode: Literal["X", "O"],
    max_harmonic: int = 2,
    resonance_rtol: float = 1e-14,
) -> ScalarFloat:
    r"""Compute the absorption coefficient $\alpha$.

    Args:
        refractive_index: Refractive index vector in the lab system.
        magnetic_field: Magnetic field vector in the lab system.
        electron_density_1e20_per_m3: Electron density in 10^20 m^-3.
        electron_temperature_keV: Electron temperature in keV.
        frequency: Frequency of the wave in Hz.
        mode: Polarization mode, either "X" for extraordinary or "O" for ordinary.
        max_harmonic: Highest cyclotron harmonic to include. Determines the KO
            tensor FLR order and the number of resonance integrals evaluated.
            Must be a Python integer (compile-time constant for JIT).
        resonance_rtol: Approximate relative accuracy of each resonance integral;
            sets the integration window and the number of quadrature nodes. Must be
            a Python float (compile-time constant for JIT).

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
    # The FLR expansion order must be at least as large as the resonant harmonic
    # so that the polarization eigenvector captures the leading-order FLR correction
    # at the resonance.  (max_s and max_k must be Python constants for JIT.)
    dielectric_tensor = dielectric_tensor_module.weakly_relativistic_dielectric_tensor(
        frequency=frequency,
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
        thermal_velocity=thermal_velocity,
        refractive_index_para=refractive_index_para,
        refractive_index_perp=refractive_index_perp,
        max_s=max_harmonic,
        max_k=max_harmonic,
    )
    polarization_vector = polarization.polarization(
        dielectric_tensor=dielectric_tensor,
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        frequency=frequency,
        cyclotron_frequency=cyclotron_frequency,
        mode=mode,
    )
    power_flux_vector = power_flux.cold_power_flux_vector_stix(
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        frequency=frequency,
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
        mode=mode,
    )
    eAe = anti_hermitian_dielectric_form(
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
        frequency=frequency,
        refractive_index_para=refractive_index_para,
        refractive_index_perp=refractive_index_perp,
        thermal_velocity=thermal_velocity,
        polarization_vector=polarization_vector,
        max_harmonic=max_harmonic,
        resonance_rtol=resonance_rtol,
    )
    # Absorption coefficient: alpha = (omega/c) * eAe / |F|
    # where eAe = ê* · ε_r^A · ê is the anti-Hermitian dielectric form
    # and F = -0.5 * dH/dN is the power flux vector.
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
        lambda: omega / speed_of_light * eAe / power_flux_magnitude,
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
    max_harmonic: int = 2,
    resonance_rtol: float = 1e-14,
) -> ScalarFloat:
    r"""Compute $\hat{e}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{e}$, the
    anti-Hermitian part of the relative dielectric tensor contracted with the
    polarization vector.

    Here $\boldsymbol{\varepsilon}_r^A = (\boldsymbol{\varepsilon}_r -
    \boldsymbol{\varepsilon}_r^\dagger)/(2i)$. This is the numerator quantity in the
    absorption coefficient formula,

    $$\alpha = \frac{\omega}{c} \frac{\hat{e}^* \cdot \boldsymbol{\varepsilon}_r^A
    \cdot \hat{e}}{|F|}$$

    where $F$ is the power flux vector.  The result is positive for an absorbing
    plasma.

    The anti-Hermitian part is computed via the fully relativistic momentum-space
    resonance integral (Bornatici et al. 1983), summed over cyclotron harmonics:

    $$\hat{e}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{e}
    = -2\pi^2 X_p \sum_s I_s$$

    where $X_p = (\omega_p/\omega)^2$ and $I_s$ is the resonance integral
    for harmonic $s$ (negative for absorption).

    Args:
        plasma_frequency: Electron plasma frequency in Hz.
        cyclotron_frequency: Electron cyclotron frequency in Hz.
        frequency: Wave frequency in Hz.
        refractive_index_para: Refractive index parallel to the magnetic field.
        refractive_index_perp: Refractive index perpendicular to the magnetic field.
        thermal_velocity: Normalized electron thermal velocity (v_th / c).
        polarization_vector: Normalized polarization vector in Stix coordinates.
        max_harmonic: Highest cyclotron harmonic to include in the sum.
        resonance_rtol: Approximate relative accuracy of each resonance integral;
            sets the integration window and the number of quadrature nodes. Must be
            a Python float (compile-time constant for JIT).

    Returns:
        The scalar quadratic form $\hat{e}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{e}$ (dimensionless).
    """  # noqa: E501
    resonance_integral = 0.0
    for harmonic_index in range(1, max_harmonic + 1):
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
                resonance_rtol=resonance_rtol,
            ),
        )
    Xp = (plasma_frequency / frequency) ** 2
    return -Xp * 2 * jnp.pi**2 * resonance_integral


def compute_resonance_integral(
    harmonic_index: int,
    cyclotron_frequency: ScalarFloat,
    frequency: ScalarFloat,
    refractive_index_para: ScalarFloat,
    refractive_index_perp: ScalarFloat,
    polarization_vector: jt.Complex[jax.Array, "3"],
    thermal_velocity: ScalarFloat,
    resonance_rtol: float = 1e-14,
) -> ScalarFloat:
    r"""Compute the resonance integral for the given harmonic.

    The resonance integral is defined as:

    $$I = \int_{u_{min}}^{u_{max}} du_\parallel D_{ql} \left(\frac{\partial f_e}{\partial \gamma} + n_\parallel \frac{\partial f_e}{\partial u_\parallel}\right)$$

    where $D_{ql}$ is the quasilinear diffusion coefficient, $f_e$ is the
    electron distribution function, $n_\parallel$ is the refractive index parallel
    to the magnetic field, and $u_\parallel$ is the normalized parallel momentum
    ($u_\parallel = p_\parallel / (m_0 c)$).

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
        resonance_rtol: Approximate relative accuracy of each resonance integral;
            sets the integration window and the number of quadrature nodes. Must be
            a Python float (compile-time constant for JIT).
    """  # noqa: E501
    # Integration bounds from the resonance circle.  The conditions
    # γ = nY + N_∥·u_∥ and γ² = 1 + u_∥² + u_⊥² define a circle in the
    # (u_∥, u_⊥) plane; setting u_⊥=0 gives the endpoints u_min and u_max.
    nY = harmonic_index * cyclotron_frequency / frequency
    n_para = refractive_index_para
    denom = 1 - n_para**2
    u_min_raw, u_max_raw, resonance_in_bulk = _resonance_circle(
        nY, n_para, thermal_velocity
    )

    # The resonant electrons lie on a curve in momentum space along which
    # γ = nY + N_∥·u_∥ grows linearly, so the integrand is largest at the
    # low-γ end of the curve (u_min_raw if N_∥ > 0, else u_max_raw) and decays
    # like exp(-μ|N_∥|·distance) from there, μ = 2/v_th². Integrate from that end
    # only as far as needed to reach the requested relative accuracy, or the
    # whole curve if it is shorter.
    mu = 2 / thermal_velocity**2
    n_decay_lengths, n_nodes = _resonance_quadrature(resonance_rtol, harmonic_index)
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    decay_length = 1 / (mu * jnp.maximum(jnp.abs(n_para), 1e-12))
    window_length = n_decay_lengths * decay_length
    u_min = jnp.where(
        n_para < 0, jnp.maximum(u_min_raw, u_max_raw - window_length), u_min_raw
    )
    u_max = jnp.where(
        n_para < 0, u_max_raw, jnp.minimum(u_max_raw, u_min_raw + window_length)
    )
    # For N_∥ → 1 the resonance circle diverges; zero the contribution since the
    # quasi-longitudinal regime is outside ECH validity.
    near_lightline = jnp.abs(denom) < 1e-4
    u_min = jnp.where(near_lightline, 0.0, u_min)
    u_max = jnp.where(near_lightline, 0.0, u_max)

    # The u_gamma_limit = (1-nY)/N_∥ bound is not enforced explicitly; the
    # u_perp^2 clamp in quasilinear_diffusion_coefficient zeros the integrand for γ<1.

    def compute_integral() -> ScalarFloat:
        # K2_scaled = kve(2, mu) depends only on thermal_velocity, which is
        # constant across all vmap iterations.  Pre-compute it here so
        # it is evaluated exactly once rather than once per node.
        K2_scaled = bessel.kve_jax(2, mu)

        # The integrand is analytic on the window and vanishes at the circle
        # ends, so Gauss-Legendre converges exponentially.
        half_width = 0.5 * (u_max - u_min)
        u_grid = half_width * nodes + 0.5 * (u_max + u_min)

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
        # Integrate using Gauss-Legendre quadrature
        return half_width * jnp.dot(weights, integrand_values)

    return jax.lax.cond(resonance_in_bulk, compute_integral, lambda: 0.0)


def _resonance_quadrature(rtol: float, harmonic_index: int) -> tuple[float, int]:
    """Window length in decay lengths and Gauss-Legendre node count for relative accuracy rtol.

    Near its low-γ end the integrand behaves like t^n e^-t (t in decay lengths, n the
    harmonic), so the tail beyond a window K is ~K^n e^-K, and N Gauss-Legendre nodes
    integrate e^-t over [0, K] to ~(K/4)^(2N) / (2N)! ≈ (eK / 8N)^(2N).
    """
    # Below ln(1/rtol) = 1 the log term flips sign and the window can vanish.
    if not 0 < rtol < 1 / math.e:
        raise ValueError(f"resonance_rtol must be in (0, 1/e), got {rtol}")
    log_inv_rtol = math.log(1 / rtol)
    window = log_inv_rtol + harmonic_index * math.log(log_inv_rtol)
    growth = math.exp(lambertw(4 * log_inv_rtol / (math.e * window)).real)
    return window, math.ceil(math.e * window / 8 * growth)


def resonance_integrand(
    harmonic_index: int,
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
    # For |N_∥| > 1 the resonance curve is a hyperbola whose second branch has γ < 1,
    # where exp(-μγ) overflows and 0·inf = NaN. Evaluate such nodes at γ = 1, where
    # Dql is zero and everything is finite, so that neither the value nor the gradient
    # picks up the NaN, and drop them.
    physical = lorentz_factor >= 1
    lorentz_factor = jnp.where(physical, lorentz_factor, 1.0)
    Dql = quasilinear_diffusion_coefficient(
        harmonic_index=harmonic_index,
        cyclotron_frequency=cyclotron_frequency,
        frequency=frequency,
        lorentz_factor=lorentz_factor,
        refractive_index_perp=refractive_index_perp,
        parallel_momentum=parallel_momentum,
        polarization_vector=polarization_vector,
    )
    df_dgamma = distribution_function.maxwell_juettner_distribution_dgamma_precomputed(
        lorentz_factor, thermal_velocity=thermal_velocity, K2_scaled=K2_scaled
    )
    # Fidone operator: Lf = df/dw + N_par * df/dq_par|_w
    # For an isotropic Maxwellian f(gamma), df/dq_par|_w = 0 (energy is constant),
    # so Lf = df/dgamma.
    return jnp.where(physical, Dql * df_dgamma, 0.0)


def quasilinear_diffusion_coefficient(
    harmonic_index: int,
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
    # Dql = u_perp^2 |Pi . e|^2, Pi = (-n J_{-n}(z)/z, i J_{-n}'(z), u_par/u_perp J_{-n}(z)),
    # z = k_perp rho_L = -kappa u_perp: negative for the electron helicity, and its sign
    # fixes the third component relative to the first two. With J_m(z) = (z/2)^m S_m(q),
    # q = z^2/4, the components share the factor (-1)^n z^(n-1) / 2^n, so
    #   Dql = u_perp^2 q^(n-1) / 4 |a . e|^2,
    #   a = (-n S_n, i (n S_n - 2 q S_{n+1}), -kappa u_par S_n),
    # a polynomial in u_par: no square root, no 1/z, no branch. Outside the resonance
    # circle (u_perp^2 < 0) the clamp makes Dql zero.
    u_perp_sq = jnp.maximum(lorentz_factor**2 - parallel_momentum**2 - 1, 0.0)
    kappa = refractive_index_perp * frequency / cyclotron_frequency
    q = kappa**2 * u_perp_sq / 4
    s_n = bessel.jv_series_q(harmonic_index, q)
    s_n1 = bessel.jv_series_q(harmonic_index + 1, q)
    a1 = -harmonic_index * s_n
    a2 = harmonic_index * s_n - 2 * q * s_n1
    a3 = -kappa * parallel_momentum * s_n
    e = polarization_vector
    a_dot_e = a1 * e[0] + 1j * a2 * e[1] + a3 * e[2]
    return u_perp_sq * q ** (harmonic_index - 1) / 4 * jnp.abs(a_dot_e) ** 2
