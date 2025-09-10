import jax.numpy as jnp
import numpy as np
from raytrax.type_conversion import ray_states_to_beam_profile, ray_states_to_radial_profile
from raytrax.ray import RayState, RayQuantities


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
        ),
        RayQuantities(
            absorption_coefficient=jnp.array(0.02),
            electron_density=jnp.array(2.0e19),
            electron_temperature=jnp.array(2.0e3),
            magnetic_field=jnp.array([0.0, 2.0, 0.0]),
        ),
    ]
    
    # Convert to beam profile
    beam_profile = ray_states_to_beam_profile(states, quantities)
    
    # Test positions
    np.testing.assert_array_equal(
        beam_profile.position,
        jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    
    # Test arc lengths
    np.testing.assert_array_equal(
        beam_profile.arc_length,
        jnp.array([1.0, 2.0])
    )
    
    # Test refractive indices
    np.testing.assert_array_equal(
        beam_profile.refractive_index,
        jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    )
    
    # Test optical depths
    np.testing.assert_array_equal(
        beam_profile.optical_depth,
        jnp.array([0.5, 1.0])
    )
    
    # Test absorption coefficients
    np.testing.assert_array_equal(
        beam_profile.absorption_coefficient,
        jnp.array([0.01, 0.02])
    )
    
    # Test electron densities
    np.testing.assert_array_equal(
        beam_profile.electron_density,
        jnp.array([1.0e19, 2.0e19])
    )
    
    # Test electron temperatures
    np.testing.assert_array_equal(
        beam_profile.electron_temperature,
        jnp.array([1.0e3, 2.0e3])
    )
    
    # Test magnetic fields
    np.testing.assert_array_equal(
        beam_profile.magnetic_field,
        jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    )


def test_ray_states_to_radial_profile():
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
        ),
        RayQuantities(
            absorption_coefficient=jnp.array(0.02),
            electron_density=jnp.array(2.0e19),
            electron_temperature=jnp.array(2.0e3),
            magnetic_field=jnp.array([0.0, 2.0, 0.0]),
        ),
    ]
    
    # Convert to radial profile
    radial_profile = ray_states_to_radial_profile(states, quantities)
    
    # Test rho values (position norms)
    np.testing.assert_array_equal(
        radial_profile.rho,
        jnp.array([5.0, 5.0])  # √(3² + 4² + 0²) = 5.0, √(0² + 0² + 5²) = 5.0
    )
