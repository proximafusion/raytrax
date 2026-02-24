import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from raytrax import absorption, distribution_function

jax.config.update("jax_enable_x64", True)


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
    # v_th/c values corresponding to T ≈ 2.5, 10, 23 keV
    for thermal_velocity in [0.1, 0.2, 0.3]:

        def integrand(u):
            gamma = np.sqrt(1 + u**2)
            f = float(
                distribution_function.maxwell_juettner_distribution(
                    gamma, thermal_velocity
                )
            )
            return 4 * np.pi * u**2 * f

        result, _ = quad(integrand, 0, np.inf)
        np.testing.assert_allclose(
            result,
            1.0,
            rtol=1e-4,
            err_msg=f"Normalization integral = {result:.6f} for v_th/c = {thermal_velocity}",
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
