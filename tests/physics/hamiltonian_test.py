import jax
import jax.numpy as jnp
import numpy as np

from raytrax.equilibrium.interpolate import (
    _VACUUM_RAMP_WIDTH,
    build_electron_density_profile_interpolator,
)
from raytrax.physics import hamiltonian
from raytrax.types import RadialProfiles

jax.config.update("jax_enable_x64", True)


def test_hamiltonian_is_continuous_where_the_density_reaches_zero():
    """H has no step at the outer end of the edge ramp, where ne goes to zero.

    Here rho is the x coordinate, so moving along x crosses the end of the ramp.
    """
    rho = jnp.linspace(0.0, 1.0, 40)
    ne_interp = build_electron_density_profile_interpolator(
        RadialProfiles(
            rho=rho,
            electron_density=0.4 * (1 - rho**2) + 0.2,
            electron_temperature=jnp.ones_like(rho),
        )
    )
    refractive_index = jnp.array([0.6, 0.0, 0.8])

    def h(x):
        return hamiltonian.hamiltonian(
            position=jnp.array([x, 0.0, 0.0]),
            refractive_index=refractive_index,
            magnetic_field_interpolator=lambda p: jnp.array([0.0, 0.0, 2.5]),
            rho_interpolator=lambda p: p[0],
            electron_density_profile_interpolator=ne_interp,
            frequency=140e9,
            mode="O",
        )[0]

    # Close to the end of the ramp, where ne falls through 1e-6 down to 0, with a
    # spacing fine enough that a smooth H changes by far less than 1e-8 per point.
    ramp_end = 1.0 + _VACUUM_RAMP_WIDTH
    x = jnp.linspace(ramp_end - 5e-5, ramp_end + 1e-5, 6001)
    values = np.asarray(jax.vmap(h)(x))
    np.testing.assert_allclose(values[-1], 0.0, atol=1e-15)  # |n|^2 - 1 in vacuum
    np.testing.assert_array_less(np.abs(np.diff(values)), 1e-8)
