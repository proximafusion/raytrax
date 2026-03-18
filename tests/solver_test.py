import interpax
import jax
import jax.numpy as jnp

from raytrax.tracer import ray, solver
from raytrax.tracer.buffers import Interpolators
from raytrax.tracer.solver import _apply_B_stellarator_symmetry, _eval_magnetic_field

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

    # Electron density profile: zero everywhere so the vacuum Hamiltonian
    # (ne < 1e-6) is used, avoiding NaN in the warm-plasma gradient at the
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


def test_stellarator_symmetry_sign_convention():
    """B_R is even, B_phi and B_Z are odd under stellarator symmetry.

    With in_second_half=True the old code flipped B_R and kept B_phi/B_Z,
    which is wrong. This test would fail under that convention.
    """
    B_cyl = jnp.array([2.0, 3.0, 4.0])

    result_second = _apply_B_stellarator_symmetry(B_cyl, jnp.array(True))
    assert jnp.allclose(result_second, jnp.array([2.0, -3.0, -4.0]))

    result_first = _apply_B_stellarator_symmetry(B_cyl, jnp.array(False))
    assert jnp.allclose(result_first, B_cyl)


def test_stellarator_symmetry_applied_in_second_half_period():
    """_eval_magnetic_field applies the symmetry correctly for phi > pi/nfp.

    nfp=1, period=2pi, half_period=pi.  A position at phi=3pi/2 (second half)
    is mapped to phi_mapped=pi/2, z -> -z.  With a uniform grid storing
    B_cyl=[1, 2, 3], the symmetry flips B_phi and B_Z before the Cartesian
    rotation, so the returned Cartesian field differs from the first-half value.
    """
    # Uniform B_cyl = [1, 2, 3] everywhere on the grid
    B_interp = interpax.Interpolator3D(
        x=jnp.array([0.0, 10.0]),
        y=jnp.array([0.0, jnp.pi]),
        z=jnp.array([-5.0, 5.0]),
        f=jnp.ones((2, 2, 2, 3)) * jnp.array([1.0, 2.0, 3.0]),
        method="linear",
    )
    rho_interp = interpax.Interpolator3D(
        x=jnp.array([0.0, 10.0]),
        y=jnp.array([0.0, jnp.pi]),
        z=jnp.array([-5.0, 5.0]),
        f=jnp.ones((2, 2, 2)) * 0.5,
        method="linear",
    )
    ne_interp = interpax.Interpolator1D(
        x=jnp.array([0.0, 1.0]), f=jnp.array([0.0, 0.0]), method="linear"
    )
    Te_interp = interpax.Interpolator1D(
        x=jnp.array([0.0, 1.0]), f=jnp.array([1.0, 1.0]), method="linear"
    )
    interps = Interpolators(
        magnetic_field=B_interp,
        rho=rho_interp,
        electron_density=ne_interp,
        electron_temperature=Te_interp,
    )

    # phi = -pi/2  =>  phi_mod = 3pi/2 > pi  =>  in_second_half=True
    # phi_mapped = 2pi - 3pi/2 = pi/2,  z_query = -z = -0.5
    # Grid returns B_cyl=[1,2,3]; symmetry gives [1,-2,-3]
    # Cartesian at phi=-pi/2: Bx = 1*cos(-pi/2) - (-2)*sin(-pi/2) = 0 - 2 = -2
    #                         By = 1*sin(-pi/2) + (-2)*cos(-pi/2) = -1 + 0 = -1
    #                         Bz = -3
    position_second_half = jnp.array([0.0, -2.0, 0.5])
    B_second = _eval_magnetic_field(position_second_half, interps, nfp=1)
    assert jnp.allclose(B_second, jnp.array([-2.0, -1.0, -3.0]), atol=1e-5)

    # phi = pi/2  =>  in_second_half=False
    # B_cyl=[1,2,3] unchanged; Cartesian at phi=pi/2:
    # Bx = 1*0 - 2*1 = -2,  By = 1*1 + 2*0 = 1,  Bz = 3
    position_first_half = jnp.array([0.0, 2.0, 0.5])
    B_first = _eval_magnetic_field(position_first_half, interps, nfp=1)
    assert jnp.allclose(B_first, jnp.array([-2.0, 1.0, 3.0]), atol=1e-5)


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
