import jax
import jax.numpy as jnp
from raytrax import ray, solver

jax.config.update("jax_enable_x64", True)


def test_y_to_state_roundtrip():
    # Test with 7-component ODE state vector
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    state = solver._y_to_state(y, s=0.0)
    y_reconstructed = solver._state_to_y(state)
    assert jnp.allclose(y, y_reconstructed)


def test_ray_tracing():
    state = ray.RayState(
        position=jnp.array([0.0, 0.0, 0.0]),
        refractive_index=jnp.array([1.0, 1.0, 1.0]),
        optical_depth=jnp.array(0.0),
        arc_length=jnp.array(0.0),
    )
    setting = ray.RaySetting(
        frequency=jnp.array(238e9),
        mode="X",
    )

    def magnetic_field_interpolator(position):
        return jnp.array([10.0, 0.0, 0.0])

    def rho_interpolator(position):
        return jnp.array(0.5)

    def electron_density_profile_interpolator(rho):
        return jnp.array(0.1)

    def electron_temperature_profile_interpolator(rho):
        return jnp.array(1.0)

    ray_states, ray_quantities = solver.solve(
        state,
        setting,
        magnetic_field_interpolator,
        rho_interpolator,
        electron_density_profile_interpolator,
        electron_temperature_profile_interpolator,
    )
    print(f"Ray states: {len(ray_states)}")
    print(f"Ray quantities: {len(ray_quantities)}")
    assert len(ray_states) > 0
    assert len(ray_quantities) == len(ray_states)


def test_quantities_computed_during_solve():
    """Test that quantities are computed correctly during ODE solve (augmented state)."""
    # Set up initial conditions
    state = ray.RayState(
        position=jnp.array([0.0, 0.0, 0.0]),
        refractive_index=jnp.array([1.0, 1.0, 1.0]),
        optical_depth=jnp.array(0.0),
        arc_length=jnp.array(0.0),
    )
    setting = ray.RaySetting(
        frequency=jnp.array(238e9),
        mode="X",
    )

    # Define interpolators for testing
    def magnetic_field_interpolator(position):
        # Return a 3D vector regardless of input shape
        return jnp.array([10.0, 0.0, 0.0])

    def rho_interpolator(position):
        # Return a scalar
        return jnp.array(0.5)

    def electron_density_profile_interpolator(rho):
        # Return a scalar
        return jnp.array(0.1)

    def electron_temperature_profile_interpolator(rho):
        # Return a scalar
        return jnp.array(1.0)

    # Solve ray equations - now returns both states and quantities
    ray_states, ray_quantities = solver.solve(
        state,
        setting,
        magnetic_field_interpolator,
        rho_interpolator,
        electron_density_profile_interpolator,
        electron_temperature_profile_interpolator,
    )

    # Check that we have the right number of quantities
    assert len(ray_quantities) == len(ray_states)
    assert len(ray_states) > 0

    # Check that the first quantity has the correct values
    first_quantities = ray_quantities[0]
    assert jnp.allclose(first_quantities.magnetic_field, jnp.array([10.0, 0.0, 0.0]))
    assert jnp.allclose(first_quantities.electron_density, jnp.array(0.1))
    assert jnp.allclose(first_quantities.electron_temperature, jnp.array(1.0))
    assert jnp.allclose(first_quantities.normalized_effective_radius, jnp.array(0.5))
