import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from raytrax.math import bessel
from raytrax.physics import absorption, distribution_function, quantities
from raytrax.physics.dielectric_tensor import (
    cold_dielectric_tensor,
    weakly_relativistic_dielectric_tensor,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

# W7-X-like 2nd-harmonic X-mode scenario (B=2.52 T -> f_ce≈70.5 GHz, f=140 GHz)
_FREQ = 140e9  # Hz
_B_FIELD = jnp.array([0.0, 0.0, 2.52])  # T
_N_PARA = 0.3
_N_PERP = 0.7
_N_VEC = jnp.array([_N_PERP, 0.0, _N_PARA])
_NE = 0.3  # 1e20 m^-3
_TE = 2.0  # keV
_MODE = "X"


def test_maxwell_juettner_normalization():
    r"""The Maxwell-Juettner distribution must integrate to 1 over :math:`d^3u`.

    .. math::

        \int f(\gamma(\mathbf{u}))\, d^3u
        = 4\pi \int_0^\infty u^2\, f(\gamma(u))\, du = 1

    where :math:`\gamma(u) = \sqrt{1 + u^2}` and :math:`u = p/(m_0 c)` is the
    normalized momentum magnitude.

    Tests several temperatures spanning non-relativistic to mildly relativistic
    regimes (:math:`\mu = 2/v_{th}^2` from ~22 to ~200).
    """
    jitted_maxwell_juettner_distribution = jax.jit(
        distribution_function.maxwell_juettner_distribution
    )
    # v_th/c values corresponding to T ≈ 2.5, 10, 23 keV
    for thermal_velocity in [0.1, 0.2, 0.3]:

        def integrand(u):
            gamma = np.sqrt(1 + u**2)
            f = float(jitted_maxwell_juettner_distribution(gamma, thermal_velocity))
            return 4 * np.pi * u**2 * f

        result, _ = quad(integrand, 0, np.inf)
        np.testing.assert_allclose(
            result,
            1.0,
            rtol=1e-4,
            err_msg=f"Normalization integral = {result:.6f} for v_th/c = {thermal_velocity}",
        )


@pytest.mark.parametrize("thermal_velocity", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("lorentz_factor", [1.0, 1.5, 2.0, 5.0])
def test_dgamma_precomputed_matches_jax_grad(lorentz_factor, thermal_velocity):
    r"""maxwell_juettner_distribution_dgamma_precomputed must agree with jax.grad.

    The pre-computed variant accepts K2_scaled = kve(2, \\mu) as an explicit
    argument to avoid redundant Bessel evaluations inside vmap loops.  It must
    return the same value as the JAX-differentiated baseline for all
    (\\gamma, v_th) combinations.
    """
    mu = 2 / thermal_velocity**2
    K2_scaled = float(bessel.kve_jax(2, mu))

    dgamma_ref = float(
        distribution_function.maxwell_juettner_distribution_dgamma(
            lorentz_factor, thermal_velocity
        )
    )
    dgamma_pre = float(
        distribution_function.maxwell_juettner_distribution_dgamma_precomputed(
            lorentz_factor, thermal_velocity, K2_scaled
        )
    )

    np.testing.assert_allclose(
        dgamma_pre,
        dgamma_ref,
        rtol=1e-12,
        err_msg=(
            f"Mismatch at v_th/c={thermal_velocity}, gamma={lorentz_factor}: "
            f"precomputed={dgamma_pre:.6e}, jax.grad={dgamma_ref:.6e}"
        ),
    )


def test_absorption():
    refractive_index_perp = 0.6
    refractive_index_para = 0.8
    refractive_index = jnp.array([refractive_index_perp, 0.0, refractive_index_para])

    magnetic_field = jnp.array([0.0, 0.0, 8.3])

    frequency = 220e9
    electron_density_1e20_per_m3 = 0.1
    electron_temperature_keV = 1.0
    mode = "X"

    alpha = absorption.absorption_coefficient(
        refractive_index=refractive_index,
        magnetic_field=magnetic_field,
        electron_density_1e20_per_m3=electron_density_1e20_per_m3,
        electron_temperature_keV=electron_temperature_keV,
        frequency=frequency,
        mode=mode,
    )

    print(f"Absorption coefficient: {alpha}")

    assert not np.isnan(alpha), "Absorption coefficient is NaN"
    assert alpha >= 0, "Absorption coefficient should be non-negative"
    print(alpha)


# ---------------------------------------------------------------------------
# Tests for max_harmonic parameter
# ---------------------------------------------------------------------------


def test_absorption_coefficient_default_max_harmonic_is_two():
    """Explicit max_harmonic=2 must give the same result as the default (2)."""
    alpha_default = absorption.absorption_coefficient(
        refractive_index=_N_VEC,
        magnetic_field=_B_FIELD,
        electron_density_1e20_per_m3=_NE,
        electron_temperature_keV=_TE,
        frequency=_FREQ,
        mode=_MODE,
    )
    alpha_explicit = absorption.absorption_coefficient(
        refractive_index=_N_VEC,
        magnetic_field=_B_FIELD,
        electron_density_1e20_per_m3=_NE,
        electron_temperature_keV=_TE,
        frequency=_FREQ,
        mode=_MODE,
        max_harmonic=2,
    )
    np.testing.assert_allclose(
        float(alpha_default), float(alpha_explicit), rtol=0, atol=0
    )


def test_absorption_coefficient_max_harmonic_one():
    """max_harmonic=1 must produce a finite, non-negative result at the fundamental."""
    # At 1st harmonic resonance (B ~ 5 T for 140 GHz -> f_ce ≈ 140 GHz)
    B_1st = jnp.array([0.0, 0.0, 5.0])
    alpha = absorption.absorption_coefficient(
        refractive_index=_N_VEC,
        magnetic_field=B_1st,
        electron_density_1e20_per_m3=_NE,
        electron_temperature_keV=_TE,
        frequency=_FREQ,
        mode=_MODE,
        max_harmonic=1,
    )
    assert np.isfinite(float(alpha)), "alpha must be finite for max_harmonic=1"
    assert float(alpha) >= 0, "alpha must be non-negative"


def test_absorption_coefficient_max_harmonic_three():
    """max_harmonic=3 must produce a finite, non-negative result."""
    alpha = absorption.absorption_coefficient(
        refractive_index=_N_VEC,
        magnetic_field=_B_FIELD,
        electron_density_1e20_per_m3=_NE,
        electron_temperature_keV=_TE,
        frequency=_FREQ,
        mode=_MODE,
        max_harmonic=3,
    )
    assert np.isfinite(float(alpha)), "alpha must be finite for max_harmonic=3"
    assert float(alpha) >= 0, "alpha must be non-negative"


def test_absorption_coefficient_higher_harmonic_same_order_of_magnitude():
    """max_harmonic=3 and max_harmonic=2 must give results of the same order of magnitude.

    Note: alpha is *not* guaranteed to increase monotonically with max_harmonic,
    because increasing max_harmonic also changes the KO dielectric tensor (and
    thus the polarization eigenvector), which can slightly modify the result.
    We only require both to be finite and within a factor of 2 of each other.
    """
    alpha2 = float(
        absorption.absorption_coefficient(
            refractive_index=_N_VEC,
            magnetic_field=_B_FIELD,
            electron_density_1e20_per_m3=_NE,
            electron_temperature_keV=_TE,
            frequency=_FREQ,
            mode=_MODE,
            max_harmonic=2,
        )
    )
    alpha3 = float(
        absorption.absorption_coefficient(
            refractive_index=_N_VEC,
            magnetic_field=_B_FIELD,
            electron_density_1e20_per_m3=_NE,
            electron_temperature_keV=_TE,
            frequency=_FREQ,
            mode=_MODE,
            max_harmonic=3,
        )
    )
    assert np.isfinite(alpha2) and np.isfinite(alpha3)
    assert alpha2 > 0 and alpha3 > 0
    ratio = alpha3 / alpha2
    assert 0.5 <= ratio <= 2.0, (
        f"max_harmonic=3 alpha ({alpha3:.4e}) should be within factor 2 of "
        f"max_harmonic=2 alpha ({alpha2:.4e}), ratio={ratio:.3f}"
    )


def test_absorption_coefficient_conditional_zero_for_cold_plasma():
    """absorption_coefficient_conditional must return 0 below temperature threshold."""
    alpha = absorption.absorption_coefficient_conditional(
        refractive_index=_N_VEC,
        magnetic_field=_B_FIELD,
        electron_density_1e20_per_m3=_NE,
        electron_temperature_keV=0.001,  # below 0.01 keV threshold
        frequency=_FREQ,
        mode=_MODE,
    )
    np.testing.assert_allclose(float(alpha), 0.0, atol=0, rtol=0)


def test_absorption_coefficient_conditional_zero_for_zero_density():
    """absorption_coefficient_conditional must return 0 for zero density."""
    alpha = absorption.absorption_coefficient_conditional(
        refractive_index=_N_VEC,
        magnetic_field=_B_FIELD,
        electron_density_1e20_per_m3=0.0,
        electron_temperature_keV=_TE,
        frequency=_FREQ,
        mode=_MODE,
    )
    np.testing.assert_allclose(float(alpha), 0.0, atol=0, rtol=0)


def test_absorption_coefficient_conditional_matches_full_above_threshold():
    """absorption_coefficient_conditional must equal absorption_coefficient above threshold."""
    alpha_cond = float(
        absorption.absorption_coefficient_conditional(
            refractive_index=_N_VEC,
            magnetic_field=_B_FIELD,
            electron_density_1e20_per_m3=_NE,
            electron_temperature_keV=_TE,
            frequency=_FREQ,
            mode=_MODE,
        )
    )
    alpha_full = float(
        absorption.absorption_coefficient(
            refractive_index=_N_VEC,
            magnetic_field=_B_FIELD,
            electron_density_1e20_per_m3=_NE,
            electron_temperature_keV=_TE,
            frequency=_FREQ,
            mode=_MODE,
        )
    )
    np.testing.assert_allclose(alpha_cond, alpha_full, rtol=1e-12)


def test_absorption_coefficient_conditional_respects_max_harmonic():
    """max_harmonic is correctly threaded through absorption_coefficient_conditional."""
    alpha2 = float(
        absorption.absorption_coefficient_conditional(
            refractive_index=_N_VEC,
            magnetic_field=_B_FIELD,
            electron_density_1e20_per_m3=_NE,
            electron_temperature_keV=_TE,
            frequency=_FREQ,
            mode=_MODE,
            max_harmonic=2,
        )
    )
    alpha3 = float(
        absorption.absorption_coefficient_conditional(
            refractive_index=_N_VEC,
            magnetic_field=_B_FIELD,
            electron_density_1e20_per_m3=_NE,
            electron_temperature_keV=_TE,
            frequency=_FREQ,
            mode=_MODE,
            max_harmonic=3,
        )
    )
    # The two values need not be equal (different harmonics), but both must be valid
    assert np.isfinite(alpha2) and np.isfinite(alpha3)
    # Check that conditional result for max_harmonic=2 matches the non-conditional version
    alpha_full2 = float(
        absorption.absorption_coefficient(
            refractive_index=_N_VEC,
            magnetic_field=_B_FIELD,
            electron_density_1e20_per_m3=_NE,
            electron_temperature_keV=_TE,
            frequency=_FREQ,
            mode=_MODE,
            max_harmonic=2,
        )
    )
    np.testing.assert_allclose(alpha2, alpha_full2, rtol=1e-12)


def test_absorption_coefficient_smooth_in_n_par():
    """Absorption coefficient must be smooth and finite as n_par -> 0.

    This is the physics-level regression test for Bug 2 (Shkarofsky F_{7/2}
    instability).  Before the fix, the Shkarofsky recurrence divided by
    n_par^2 (= psi) near-perpendicular propagation, producing catastrophic
    cancellation that made alpha spike to near-zero or NaN for small |n_par|.

    Requirements tested:
    - alpha is finite and non-negative for all n_par in [-0.3, 0.3]
    - alpha varies smoothly: no jumps larger than 5x the median step size
    - alpha(n_par=0) agrees with alpha(n_par=±0.01) to within 1%
      (the psi->0 limit must be continuous)
    """
    # 2nd-harmonic X-mode near ECR (B chosen so f == 2*f_ce)
    freq = 140e9
    B = jnp.array([0.0, 0.0, 2.52])  # f_ce ≈ 70.5 GHz
    ne = 0.3
    Te = 2.0
    n_perp = 0.7

    n_par_values = jnp.linspace(-0.3, 0.3, 61)
    alphas = []
    for n_par in n_par_values:
        n_vec = jnp.array([n_perp, 0.0, float(n_par)])
        alpha = float(
            absorption.absorption_coefficient(
                refractive_index=n_vec,
                magnetic_field=B,
                electron_density_1e20_per_m3=ne,
                electron_temperature_keV=Te,
                frequency=freq,
                mode="X",
            )
        )
        assert np.isfinite(alpha), (
            f"alpha is not finite at n_par={float(n_par):.4f}: got {alpha}"
        )
        assert alpha >= 0.0, (
            f"alpha is negative at n_par={float(n_par):.4f}: got {alpha:.4e}"
        )
        alphas.append(alpha)

    alphas = np.array(alphas)

    # No step should be larger than 5x the median absolute step (smoothness)
    steps = np.abs(np.diff(alphas))
    median_step = np.median(steps)
    max_step = np.max(steps)
    assert max_step <= 5.0 * median_step + 1e-6, (
        f"alpha has a large jump: max_step={max_step:.3e}, "
        f"median_step={median_step:.3e} — suggests numerical instability near n_par=0"
    )

    # alpha at n_par=0 (index 30) must be close to its tiny-n_par neighbours
    alpha_zero = alphas[30]
    alpha_near = 0.5 * (alphas[29] + alphas[31])
    np.testing.assert_allclose(
        alpha_zero,
        alpha_near,
        rtol=0.01,
        err_msg=(
            f"alpha at n_par=0 ({alpha_zero:.4e}) differs from neighbours "
            f"({alpha_near:.4e}) by more than 1% — psi->0 limit is discontinuous"
        ),
    )


def test_anti_hermitian_dielectric_form_positive():
    """eAe must be strictly positive in an absorbing plasma."""
    B_magnitude = float(jnp.linalg.norm(_B_FIELD))
    cyclotron_freq = quantities.electron_cyclotron_frequency(B_magnitude)
    plasma_freq = quantities.electron_plasma_frequency(_NE)
    thermal_vel = quantities.normalized_electron_thermal_velocity(_TE)

    # Build a simple polarization vector (unit vector in x – rough placeholder)
    pol = jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex128)

    eAe = absorption.anti_hermitian_dielectric_form(
        plasma_frequency=plasma_freq,
        cyclotron_frequency=cyclotron_freq,
        frequency=_FREQ,
        refractive_index_para=_N_PARA,
        refractive_index_perp=_N_PERP,
        thermal_velocity=thermal_vel,
        polarization_vector=pol,
        max_harmonic=2,
    )
    assert np.isfinite(float(eAe)), "eAe must be finite"
    assert float(eAe) >= 0, "eAe must be non-negative (absorbing plasma)"


# ---------------------------------------------------------------------------
# KO dielectric tensor cold-plasma limit
# ---------------------------------------------------------------------------


def test_ko_tensor_reduces_to_cold_in_low_temperature_limit():
    """The KO warm tensor must converge element-wise to the cold tensor as T → 0.

    Both functions use the KO electron-sign convention (Y_e = −f_ce/f < 0),
    so all nine elements — including the off-diagonal ε[0,1] — must match
    in both real and imaginary parts as T → 0.
    """
    frequency = 200e9
    plasma_frequency = 50e9  # X_p = (f_p/f)^2 = 0.0625
    cyclotron_frequency = 400e9  # Y = 2, far from every resonance
    refractive_index_para = 0.5
    refractive_index_perp = 0.3

    eps_cold = cold_dielectric_tensor(
        frequency=frequency,
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
    )
    eps_warm = weakly_relativistic_dielectric_tensor(
        frequency=frequency,
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
        thermal_velocity=quantities.normalized_electron_thermal_velocity(
            electron_temperature_keV=1e-4
        ),
        refractive_index_para=refractive_index_para,
        refractive_index_perp=refractive_index_perp,
        max_s=2,
        max_k=2,
    )

    # S and P diagonal elements
    for i in range(3):
        np.testing.assert_allclose(
            float(eps_warm[i, i].real),
            float(eps_cold[i, i].real),
            rtol=1e-3,
            err_msg=f"eps[{i},{i}].real should match cold",
        )
    # Anti-Hermitian diagonal must vanish (no absorption far from resonance)
    for i in range(3):
        np.testing.assert_allclose(
            float(eps_warm[i, i].imag),
            0.0,
            atol=1e-6,
            err_msg=f"eps[{i},{i}].imag should be ~0 far from resonance",
        )
    # Off-diagonal 1–3 and 2–3 blocks vanish
    for i, j in [(0, 2), (1, 2), (2, 0), (2, 1)]:
        np.testing.assert_allclose(
            float(abs(eps_warm[i, j])),
            0.0,
            atol=1e-4,
            err_msg=f"eps[{i},{j}] should vanish in cold limit",
        )
    # In-plane off-diagonal D: both sign AND magnitude must match
    np.testing.assert_allclose(
        float(eps_warm[0, 1].imag),
        float(eps_cold[0, 1].imag),
        rtol=1e-3,
        err_msg="eps[0,1].imag (D element) must match cold value",
    )
    np.testing.assert_allclose(
        float(eps_warm[1, 0].imag),
        float(eps_cold[1, 0].imag),
        rtol=1e-3,
        err_msg="eps[1,0].imag (-D element) must match cold value",
    )
