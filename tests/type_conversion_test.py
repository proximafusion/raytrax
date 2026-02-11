import jax.numpy as jnp
import numpy as np
from raytrax.type_conversion import (
    ray_states_to_beam_profile,
    ray_states_to_radial_profile,
)
from raytrax.ray import RayState, RayQuantities
from tests.fixtures import torus_wout


def test_ray_states_to_beam_profile():
    # Create some test ray states
    states = [
        RayState(
            position=jnp.array([1.0, 2.0, 3.0]),
            refractive_index=jnp.array([0.1, 0.2, 0.3]),
            optical_depth=jnp.array(0.5),
            arc_length=jnp.array(1.0),
        ),
        RayState(
            position=jnp.array([4.0, 5.0, 6.0]),
            refractive_index=jnp.array([0.4, 0.5, 0.6]),
            optical_depth=jnp.array(1.0),
            arc_length=jnp.array(2.0),
        ),
    ]

    # Create test ray quantities
    quantities = [
        RayQuantities(
            absorption_coefficient=jnp.array(0.01),
            electron_density=jnp.array(1.0e19),
            electron_temperature=jnp.array(1.0e3),
            magnetic_field=jnp.array([1.0, 0.0, 0.0]),
            linear_power_density=jnp.array(5.0),
            normalized_effective_radius=jnp.array(0.5),
        ),
        RayQuantities(
            absorption_coefficient=jnp.array(0.02),
            electron_density=jnp.array(2.0e19),
            electron_temperature=jnp.array(2.0e3),
            magnetic_field=jnp.array([0.0, 2.0, 0.0]),
            linear_power_density=jnp.array(10.0),
            normalized_effective_radius=jnp.array(0.8),
        ),
    ]

    # Convert to beam profile
    beam_profile = ray_states_to_beam_profile(states, quantities)

    # Test positions
    np.testing.assert_array_equal(
        beam_profile.position, jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )

    # Test arc lengths
    np.testing.assert_array_equal(beam_profile.arc_length, jnp.array([1.0, 2.0]))

    # Test refractive indices
    np.testing.assert_array_equal(
        beam_profile.refractive_index, jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    )

    # Test optical depths
    np.testing.assert_array_equal(beam_profile.optical_depth, jnp.array([0.5, 1.0]))

    # Test absorption coefficients
    np.testing.assert_array_equal(
        beam_profile.absorption_coefficient, jnp.array([0.01, 0.02])
    )

    # Test electron densities
    np.testing.assert_array_equal(
        beam_profile.electron_density, jnp.array([1.0e19, 2.0e19])
    )

    # Test electron temperatures
    np.testing.assert_array_equal(
        beam_profile.electron_temperature, jnp.array([1.0e3, 2.0e3])
    )

    # Test magnetic fields
    np.testing.assert_array_equal(
        beam_profile.magnetic_field, jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    )

    # Test linear power density
    np.testing.assert_array_equal(
        beam_profile.linear_power_density, jnp.array([5.0, 10.0])
    )


def test_ray_states_to_radial_profile(torus_wout):
    # Create some test ray states
    states = [
        RayState(
            position=jnp.array([3.0, 4.0, 0.0]),
            refractive_index=jnp.array([0.1, 0.2, 0.3]),
            optical_depth=jnp.array(0.5),
            arc_length=jnp.array(1.0),
        ),
        RayState(
            position=jnp.array([0.0, 0.0, 5.0]),
            refractive_index=jnp.array([0.4, 0.5, 0.6]),
            optical_depth=jnp.array(1.0),
            arc_length=jnp.array(2.0),
        ),
    ]

    # Create test ray quantities
    quantities = [
        RayQuantities(
            absorption_coefficient=jnp.array(0.01),
            electron_density=jnp.array(1.0e19),
            electron_temperature=jnp.array(1.0e3),
            magnetic_field=jnp.array([1.0, 0.0, 0.0]),
            linear_power_density=jnp.array(5.0),
            normalized_effective_radius=jnp.array(0.6),
        ),
        RayQuantities(
            absorption_coefficient=jnp.array(0.02),
            electron_density=jnp.array(2.0e19),
            electron_temperature=jnp.array(2.0e3),
            magnetic_field=jnp.array([0.0, 2.0, 0.0]),
            linear_power_density=jnp.array(10.0),
            normalized_effective_radius=jnp.array(0.9),
        ),
    ]

    # Compute volume derivative arrays
    from raytrax.fourier import dvolume_drho

    rho_1d = jnp.linspace(0, 1, 200)
    dv_drho = dvolume_drho(torus_wout, rho_1d)

    # Convert to radial profile
    radial_profile = ray_states_to_radial_profile(states, quantities, rho_1d, dv_drho)

    # Check that we have the ray points (not a binned grid)
    assert len(radial_profile.rho) == 2  # Same as number of ray points
    np.testing.assert_array_equal(
        radial_profile.rho,
        jnp.array([0.6, 0.9]),  # normalized_effective_radius from RayQuantities
    )

    # Check volumetric power density field exists
    assert hasattr(radial_profile, "volumetric_power_density")
    assert radial_profile.volumetric_power_density.shape == radial_profile.rho.shape
    assert jnp.all(jnp.isfinite(radial_profile.volumetric_power_density))
    assert jnp.all(
        radial_profile.volumetric_power_density >= 0
    )  # Should be non-negative
