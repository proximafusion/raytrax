import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import root_scalar

from raytrax.physics import (
    dielectric_tensor,
    dispersion,
    polarization,
    power_flux,
    quantities,
)

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


@pytest.mark.parametrize("mode", ["O", "X"])
@pytest.mark.parametrize(
    ("X", "Y", "refractive_index_para"),
    [(0.3, 0.6, 0.2), (0.8, 0.45, 0.4), (0.1, 1.3, 0.0), (0.15, 0.7, 0.5)],
)
def test_cold_power_flux_closed_form_matches_derivative_on_dispersion_surface(
    mode, X, Y, refractive_index_para
):
    """The closed form equals -1/2 dH/dN with the polarization differentiated too.

    On the dispersion surface D e = 0, so the derivative of the polarization does
    not contribute; the reference differentiates through it anyway.
    """
    frequency = 140e9
    n_perp_squared = dispersion.refractive_index_perp_squared_cold(
        refractive_index_para_squared=refractive_index_para**2, X=X, Y=Y, mode=mode
    )
    assert n_perp_squared > 0
    refractive_index = jnp.array([jnp.sqrt(n_perp_squared), 0.0, refractive_index_para])
    eps = dielectric_tensor.cold_dielectric_tensor(
        frequency=frequency,
        plasma_frequency=np.sqrt(X) * frequency,
        cyclotron_frequency=Y * frequency,
    )

    def hamiltonian(n):
        e = polarization.polarization(
            dielectric_tensor=eps,
            refractive_index_perp=n[0],
            refractive_index_para=n[2],
            frequency=frequency,
            cyclotron_frequency=Y * frequency,
            mode=mode,
        )
        d = dispersion.dispersion_tensor_stix(
            refractive_index_perp=n[0],
            refractive_index_para=n[2],
            dielectric_tensor=eps,
        )
        return jnp.real(e.conj() @ d @ e)

    closed_form = power_flux.cold_power_flux_vector_stix(
        refractive_index_perp=refractive_index[0],
        refractive_index_para=refractive_index[2],
        frequency=frequency,
        plasma_frequency=np.sqrt(X) * frequency,
        cyclotron_frequency=Y * frequency,
        mode=mode,
    )
    np.testing.assert_allclose(hamiltonian(refractive_index), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        closed_form, -0.5 * jax.grad(hamiltonian)(refractive_index), atol=1e-12
    )
