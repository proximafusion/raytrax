import jax
import jax.numpy as jnp

from raytrax.physics import dielectric_tensor, polarization

jax.config.update("jax_enable_x64", True)


def test_polarization():
    eps = dielectric_tensor.cold_dielectric_tensor(
        frequency=220.0,
        plasma_frequency=260.0,
        cyclotron_frequency=232.0,
    )
    pol = polarization.polarization(
        dielectric_tensor=eps,
        refractive_index_perp=1 - 0.4**2,
        refractive_index_para=0.4,
        frequency=220.0,
        cyclotron_frequency=232.0,
        mode="X",
    )
    assert pol.shape == (3,)
    assert jnp.isclose(jnp.linalg.norm(pol), 1.0)
    # TODO(dstraub): more meaningful functional tests
