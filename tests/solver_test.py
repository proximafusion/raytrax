import diffrax
import interpax
import jax
import jax.numpy as jnp
import numpy as np

from raytrax.tracer import ray, solver
from raytrax.tracer.buffers import Interpolators

jax.config.update("jax_enable_x64", True)


def test_dispersion_controller_preserves_smooth_flow_gradient(monkeypatch):
    """An analytic conserved flow checks AD with the extra error term on and off."""
    monkeypatch.setattr(
        solver, "_hamiltonian_value", lambda y, args: jnp.sum(y[:6] ** 2) - 1
    )

    def rhs(t, y, args):
        # Harmonic oscillator: x^2 + p^2 is conserved; tau integrates x^2.
        return jnp.array([y[3], 0.0, 0.0, -y[0], 0.0, 0.0, y[0] ** 2])

    def integral(theta, weight):
        controller = solver._DispersionAwareController(
            diffrax.PIDController(rtol=1e-7, atol=1e-9), weight
        )
        y0 = jnp.array([jnp.cos(theta), 0.0, 0.0, jnp.sin(theta), 0.0, 0.0, 0.0])
        result = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            diffrax.Bosh3(),
            t0=0.0,
            t1=0.5,
            dt0=0.1,
            y0=y0,
            stepsize_controller=controller,
            max_steps=4096,
        )
        return result.ys[-1, -1]

    theta, end = 1.2, 0.5
    expected_value = end / 2 + (np.sin(2 * end - 2 * theta) + np.sin(2 * theta)) / 4
    expected_gradient = (np.cos(2 * theta) - np.cos(2 * end - 2 * theta)) / 2
    value_and_grad = jax.jit(jax.value_and_grad(integral))
    for weight in (0.0, 30.0):
        value, gradient = value_and_grad(theta, jnp.asarray(weight))
        np.testing.assert_allclose(value, expected_value, rtol=2e-6)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-6)


def _mock_interpolators():
    """Create mock interpolators for testing."""
    # Magnetic field: constant 10 T in R direction (cylindrical coords → Cartesian)
    magnetic_field_interpolator = interpax.Interpolator3D(
        x=jnp.array([0.0, 10.0]),  # r
        y=jnp.array([0.0, 1.0]),  # phi
        z=jnp.array([0.0, 1.0]),  # z
        f=jnp.array(
            [
                [
                    [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                    [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                ],
                [
                    [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                    [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                ],
            ]
        ),
        method="linear",
    )

    # Rho: constant 0.5
    rho_interpolator = interpax.Interpolator3D(
        x=jnp.array([0.0, 10.0]),
        y=jnp.array([0.0, 1.0]),
        z=jnp.array([0.0, 1.0]),
        f=jnp.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]),
        method="linear",
    )

    # Electron density profile: zero everywhere so the vacuum Hamiltonian
    # (ne = 0) is used, avoiding NaN in the warm-plasma gradient at the
    # mock's unphysical initial refractive index N=(1,1,1)/√3.
    electron_density_interpolator = interpax.Interpolator1D(
        x=jnp.array([0.0, 1.0]),
        f=jnp.array([0.0, 0.0]),
        method="linear",
    )

    # Electron temperature profile
    electron_temperature_interpolator = interpax.Interpolator1D(
        x=jnp.array([0.0, 1.0]),
        f=jnp.array([1.0, 1.0]),
        method="linear",
    )

    return Interpolators(
        magnetic_field=magnetic_field_interpolator,
        rho=rho_interpolator,
        electron_density=electron_density_interpolator,
        electron_temperature=electron_temperature_interpolator,
    )


def test_ray_tracing():
    setting = ray.RaySetting(
        frequency=jnp.array(238e9),
        mode="X",
    )
    interpolators = _mock_interpolators()

    position = jnp.array([1.0, 0.0, 0.0])
    direction = jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3.0)

    result, n_valid = solver.trace_jitted(
        position,
        direction,
        setting,
        interpolators,
        5,
    )

    assert n_valid.item() > 0
    assert result.ode_state.shape[1] == 7


def test_quantities_computed_during_solve():
    """Test that diagnostics are computed correctly in post-processing."""
    setting = ray.RaySetting(
        frequency=jnp.array(238e9),
        mode="X",
    )
    interpolators = _mock_interpolators()

    position = jnp.array([1.0, 0.0, 0.0])
    direction = jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3.0)

    result, n_valid = solver.trace_jitted(
        position,
        direction,
        setting,
        interpolators,
        5,
    )

    assert n_valid.item() > 0

    # Check first valid point has expected values from mock interpolators.
    # index 0 corresponds to the first saved arc-length point near s=0,
    # where the mock gives: B_R = 10 T at phi=0 → B_cart = [10, 0, 0];
    # rho = 0.5; ne = 0; Te = 1 keV.
    assert jnp.allclose(result.magnetic_field[0], jnp.array([10.0, 0.0, 0.0]), atol=0.1)
    assert jnp.allclose(result.electron_density[0], jnp.array(0.0), atol=1e-4)
    assert jnp.allclose(result.electron_temperature[0], jnp.array(1.0), atol=1e-4)
    assert jnp.allclose(
        result.normalized_effective_radius[0], jnp.array(0.5), atol=1e-4
    )
