import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root_scalar

from raytrax.physics import power_flux, quantities

jax.config.update("jax_enable_x64", True)


def test_hamiltonian():
    refractive_index_perp = 0.6
    frequency = 220e9
    plasma_frequency = 2e9
    cyclotron_frequency = 232e9
    electron_temperature_keV = 1.0
    mode = "X"
    thermal_velocity = quantities.normalized_electron_thermal_velocity(
        electron_temperature_keV=electron_temperature_keV
    )

    @jax.jit
    def _h(n_para):
        return power_flux.power_flux_hamiltonian_stix(
            refractive_index=jnp.array([refractive_index_perp, 0.0, n_para]),
            frequency=frequency,
            plasma_frequency=plasma_frequency,
            cyclotron_frequency=cyclotron_frequency,
            thermal_velocity=thermal_velocity,
            mode=mode,
            max_s=1,
            max_k=1,
        )

    # determine n_para such that hamiltonian is zero
    sol = root_scalar(_h, bracket=[0.1, 1.0], method="bisect")
    refractive_index_para = sol.root

    np.testing.assert_allclose(
        _h(refractive_index_para),
        0.0,
        rtol=0,
        atol=1e-6,
    )


def test_power_flux_vector():
    refractive_index_perp = 0.6
    refractive_index_para = 0.8
    frequency = 220e9
    plasma_frequency = 2e9
    cyclotron_frequency = 232e9
    electron_temperature_keV = 1.0
    mode = "X"
    thermal_velocity = quantities.normalized_electron_thermal_velocity(
        electron_temperature_keV=electron_temperature_keV
    )
    power_flux_vector = power_flux.power_flux_vector_stix(
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        frequency=frequency,
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
        thermal_velocity=thermal_velocity,
        mode=mode,
        max_s=1,
        max_k=1,
    )

    assert power_flux_vector.shape == (3,)
    assert np.isfinite(power_flux_vector).all()
