import jax.numpy as jnp
import numpy as np
from raytrax.type_conversion import ray_states_to_beam_profile, ray_states_to_radial_profile
from raytrax.ray import RayState


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
    
    # Convert to beam profile
    beam_profile = ray_states_to_beam_profile(states)
    
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
    
    # Convert to radial profile
    radial_profile = ray_states_to_radial_profile(states)
    
    # Test rho values (position norms)
    np.testing.assert_array_equal(
        radial_profile.rho,
        jnp.array([5.0, 5.0])  # √(3² + 4² + 0²) = 5.0, √(0² + 0² + 5²) = 5.0
    )
