import jax
import jax.numpy as jnp
import interpax
from raytrax import ray, solver
from raytrax.types import Interpolators

jax.config.update("jax_enable_x64", True)


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

    # Electron density profile
    electron_density_interpolator = interpax.Interpolator1D(
        x=jnp.array([0.0, 1.0]),
        f=jnp.array([0.1, 0.1]),
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

    position = jnp.array([0.0, 0.0, 0.0])
    direction = jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3.0)
    rho_1d = jnp.linspace(0, 1, 50)
    dvolume_drho = jnp.ones(50)

    ts, ys, B_all, rho_all, ne_all, te_all, alpha_all, P_all, dP_dV = (
        solver.trace_jitted(
            position,
            direction,
            setting,
            interpolators,
            5,
            rho_1d,
            dvolume_drho,
        )
    )

    n = int(jnp.sum(jnp.isfinite(ts)).item())
    assert n > 0
    assert ys.shape[1] == 7


def test_quantities_computed_during_solve():
    """Test that diagnostics are computed correctly in post-processing."""
    setting = ray.RaySetting(
        frequency=jnp.array(238e9),
        mode="X",
    )
    interpolators = _mock_interpolators()

    position = jnp.array([0.0, 0.0, 0.0])
    direction = jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3.0)
    rho_1d = jnp.linspace(0, 1, 50)
    dvolume_drho = jnp.ones(50)

    ts, ys, B_all, rho_all, ne_all, te_all, alpha_all, P_all, dP_dV = (
        solver.trace_jitted(
            position,
            direction,
            setting,
            interpolators,
            5,
            rho_1d,
            dvolume_drho,
        )
    )

    n = int(jnp.sum(jnp.isfinite(ts)).item())
    assert n > 0

    # Check first valid point has expected values from mock interpolators
    assert jnp.allclose(B_all[0], jnp.array([10.0, 0.0, 0.0]), atol=0.1)
    assert jnp.allclose(ne_all[0], jnp.array(0.1), atol=1e-4)
    assert jnp.allclose(te_all[0], jnp.array(1.0), atol=1e-4)
    assert jnp.allclose(rho_all[0], jnp.array(0.5), atol=1e-4)
