import jax
import jax.numpy as jnp
import numpy as np
import pytest

from raytrax.physics.dispersion import (
    _dispersion_appleton_hartee,
    dispersion_cold,
    dispersion_relation_cold,
    dispersion_tensor_stix,
    refractive_index_perp_squared_cold,
)

jax.config.update("jax_enable_x64", True)
_MACHINE_PRECISION = float(np.finfo(float).eps)


def test_cold_dispersion_tensor_vacuum():
    """Check that, in vacuum, N^2 = 1."""

    # In vacuum, plasma frequency and cyclotron frequency are zero
    # Test various directions of propagation and both modes
    test_cases = [
        {"refractive_index_perp": 0.0, "refractive_index_para": 1.0, "mode": "X"},
        {"refractive_index_perp": 1.0, "refractive_index_para": 0.0, "mode": "X"},
        {"refractive_index_perp": 0.707, "refractive_index_para": 0.707, "mode": "X"},
        {"refractive_index_perp": 0.0, "refractive_index_para": 1.0, "mode": "O"},
        {"refractive_index_perp": 1.0, "refractive_index_para": 0.0, "mode": "O"},
        {"refractive_index_perp": 0.707, "refractive_index_para": 0.707, "mode": "O"},
    ]

    frequency = 1.0e9  # 1 GHz, arbitrary non-zero value
    plasma_frequency = 0.0  # Zero in vacuum
    cyclotron_frequency = 0.0  # Zero in vacuum

    for case in test_cases:
        n_squared = dispersion_cold(
            refractive_index_perp=case["refractive_index_perp"],
            refractive_index_para=case["refractive_index_para"],
            frequency=frequency,
            cyclotron_frequency=cyclotron_frequency,
            plasma_frequency=plasma_frequency,
            mode=case["mode"],
        )

        # In vacuum, N^2 should be exactly 1
        np.testing.assert_allclose(
            n_squared,
            1.0,
            rtol=0,
            atol=_MACHINE_PRECISION,
            err_msg=f"Failed for {case}",
        )


@pytest.mark.parametrize("sin2theta", [0.1, 0.5, 0.9, 1.0])
def test_appleton_hartree_o_mode_at_cutoff(sin2theta):
    """The O-mode N^2 and its derivative are finite at the cutoff X = 1."""
    Y = 0.5

    def n2(X):
        return _dispersion_appleton_hartee(X=X, Y=Y, sin2theta=sin2theta, mode="O")

    np.testing.assert_allclose(n2(1.0), 0.0, rtol=0, atol=_MACHINE_PRECISION)
    slope = jax.grad(n2)(1.0)
    assert np.isfinite(slope)
    # Near the cutoff, N^2 = slope * (X - 1) to machine precision.
    for eps in [1e-9, 1e-11, 1e-13]:
        for X in [1.0 - eps, 1.0 + eps]:
            np.testing.assert_allclose(
                n2(X), slope * (X - 1.0), rtol=0, atol=4 * _MACHINE_PRECISION
            )


@pytest.mark.parametrize("mode", ["O", "X"])
def test_appleton_hartree_matches_textbook_form(mode):
    """Away from X = 1, N^2 agrees with the textbook Appleton-Hartree form."""
    X, Y, sin2theta = np.meshgrid(
        [0.1, 0.4, 0.7, 1.3], [0.3, 0.6, 1.2], [0.0, 0.3, 0.8, 1.0], indexing="ij"
    )
    cos2theta = 1 - sin2theta
    sign = -1 if mode == "X" else +1
    G = np.sqrt(Y**4 * sin2theta**2 + 4 * (1 - X) ** 2 * Y**2 * cos2theta)
    expected = 1 - 2 * X * (1 - X) / (2 * (1 - X) - Y**2 * sin2theta + sign * G)
    n2 = _dispersion_appleton_hartee(X=X, Y=Y, sin2theta=sin2theta, mode=mode)
    np.testing.assert_allclose(n2, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("mode", ["O", "X"])
@pytest.mark.parametrize("Y", [0.3, 0.95, 1.0, 1.05, 2.0])
def test_perp_root_lies_on_the_appleton_hartree_sheet_of_its_mode(mode, Y):
    """Where the mode propagates, (N_perp, N_par) from the root at fixed N_par solves
    the Appleton-Hartree relation of the same mode at the corresponding angle, below
    and above Y = 1."""
    X, n_para2 = np.meshgrid(
        np.linspace(0.02, 1.0, 50), np.linspace(0.0, 1.5, 31), indexing="ij"
    )
    n_perp2 = np.asarray(refractive_index_perp_squared_cold(n_para2, X, Y, mode))
    propagating = n_perp2 > 1e-9
    assert propagating.sum() > 100
    n2 = n_perp2 + n_para2
    sin2theta = n_perp2 / n2
    n2_ah = np.asarray(_dispersion_appleton_hartee(X, Y, sin2theta, mode))
    np.testing.assert_allclose(n2[propagating], n2_ah[propagating], rtol=0, atol=1e-9)


@pytest.mark.parametrize("mode", ["O", "X"])
@pytest.mark.parametrize("Y", [0.0, 0.5, 2.0])
def test_perp_root_in_vacuum(mode, Y):
    """At zero density N_perp^2 = 1 - N_par^2 exactly, and the value and the
    derivative are continuous into the plasma: the discriminant is of order X^2 and
    its evaluation must not cancel."""
    n_para2 = 0.3
    np.testing.assert_allclose(
        refractive_index_perp_squared_cold(n_para2, 0.0, Y, mode),
        1 - n_para2,
        rtol=0,
        atol=_MACHINE_PRECISION,
    )
    slope = jax.grad(refractive_index_perp_squared_cold, argnums=1)
    slope_0 = slope(n_para2, 0.0, Y, mode)
    assert np.isfinite(slope_0) and slope_0 != 0
    for X in [1e-14, 1e-10, 1e-6]:
        np.testing.assert_allclose(
            refractive_index_perp_squared_cold(n_para2, X, Y, mode),
            1 - n_para2 + slope_0 * X,
            rtol=0,
            atol=1e-11,
        )
        np.testing.assert_allclose(slope(n_para2, X, Y, mode), slope_0, rtol=1e-5)


@pytest.mark.parametrize("n_para2", [0.0, 0.2, 0.5, 0.8])
@pytest.mark.parametrize("Y", [0.5, 1.5])
def test_o_mode_perp_root_is_smooth_where_the_mode_is_reflected(n_para2, Y):
    """N_perp^2 of the O-mode goes through zero linearly, with a continuous
    derivative, where the mode is reflected: at X = 1 for N_par^2 < Y / (1 + Y), and
    at X = (1 + Y)(1 - N_par^2) < 1 beyond that, where the O-mode turns into the
    L-wave. The Appleton-Hartree branch point at X = 1, N_perp = 0 is absent."""
    X_reflect = min(1.0, (1 + Y) * (1 - n_para2))

    def n_perp2(X):
        return refractive_index_perp_squared_cold(n_para2, X, Y, "O")

    np.testing.assert_allclose(n_perp2(X_reflect), 0.0, rtol=0, atol=_MACHINE_PRECISION)
    slope = jax.grad(n_perp2)(X_reflect)
    assert np.isfinite(slope) and slope < 0
    for eps in [1e-9, 1e-6, 1e-3]:
        for X in [X_reflect - eps, X_reflect + eps]:
            # Linear up to the quadratic term, which is of relative size ~eps.
            np.testing.assert_allclose(
                n_perp2(X), slope * (X - X_reflect), rtol=10 * eps, atol=1e-15
            )
            np.testing.assert_allclose(jax.grad(n_perp2)(X), slope, rtol=10 * eps)


def _fold_discriminant(n_para2, X, Y):
    return Y**2 * (1 - n_para2) ** 2 + 4 * n_para2 * (1 - X)


@pytest.mark.parametrize("mode", ["O", "X"])
@pytest.mark.parametrize("Y", [0.3, 0.95, 1.05, 2.0])
def test_dispersion_relation_vanishes_on_the_root_of_its_mode(mode, Y):
    """D_sigma is zero at N_perp^2 = n_sigma(N_par^2, X, Y) wherever that root exists,
    and its slope in N_perp^2 there is 2 sqrt(g) / (sqrt(g) + sqrt(g0)): the
    Hamiltonian n_perp^2 - n_sigma reparametrised, with the factor one at zero
    density and zero at the fold."""
    X, n_para2 = np.meshgrid(
        np.linspace(0.02, 1.0 + 0.9 * Y, 60), np.linspace(0.0, 1.5, 31), indexing="ij"
    )
    g = _fold_discriminant(n_para2, X, Y)
    root = np.asarray(refractive_index_perp_squared_cold(n_para2, X, Y, mode))
    exists = (g > 1e-6) & np.isfinite(root) & (np.abs(root) < 50)
    assert exists.sum() > 500
    value, slope = jax.vmap(
        jax.value_and_grad(dispersion_relation_cold), in_axes=(0, 0, 0, None, None)
    )(root[exists], n_para2[exists], X[exists], Y, mode)
    np.testing.assert_allclose(value, 0.0, rtol=0, atol=1e-11)
    g0 = _fold_discriminant(n_para2[exists], 0.0, Y)
    expected = 2 * np.sqrt(g[exists]) / (np.sqrt(g[exists]) + np.sqrt(g0))
    np.testing.assert_allclose(slope, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("mode", ["O", "X"])
@pytest.mark.parametrize("Y", [0.0, 0.5, 2.0])
def test_dispersion_relation_in_vacuum(mode, Y):
    """At zero density D_sigma = N^2 - 1 exactly, on and off the surface, and value and
    gradient are continuous into the plasma, where on the surface it agrees with
    n_perp^2 - n_sigma of the root to second order in X."""
    n_para2 = 0.3
    for n_perp2 in [1 - n_para2, 0.4, 1.2]:
        np.testing.assert_allclose(
            dispersion_relation_cold(n_perp2, n_para2, 0.0, Y, mode),
            n_perp2 + n_para2 - 1,
            rtol=0,
            atol=_MACHINE_PRECISION,
        )
    # exactly on the vacuum surface: 0/0, replaced by its limit, gradient finite
    grads = jax.grad(dispersion_relation_cold, argnums=(0, 1, 2))
    assert np.all(np.isfinite(grads(1 - n_para2, n_para2, 0.0, Y, mode)))
    slope_root = jax.grad(refractive_index_perp_squared_cold, argnums=1)(
        n_para2, 0.0, Y, mode
    )
    for X in [1e-14, 1e-10, 1e-6, 1e-3]:
        # on the vacuum surface the Hamiltonian of the root is -slope_root * X + O(X^2)
        np.testing.assert_allclose(
            dispersion_relation_cold(1 - n_para2, n_para2, X, Y, mode),
            -slope_root * X,
            rtol=10 * X + 1e-12,
            atol=1e-15,
        )
        d_perp2, _, d_X = grads(1 - n_para2, n_para2, X, Y, mode)
        np.testing.assert_allclose(d_perp2, 1.0, rtol=10 * X + 1e-12)
        np.testing.assert_allclose(d_X, -slope_root, rtol=10 * X + 1e-10)


def test_dispersion_relation_is_regular_across_the_fold():
    """The X mode at Y > 1 beyond the cutoff: the two roots coincide at the fold
    X_fold = 1 + Y^2 (1 - N_par^2)^2 / (4 N_par^2), where the root at fixed N_par has
    a branch point and is NaN beyond. D_sigma is zero on both roots up to the fold,
    smooth with a finite, nonzero gradient at the fold, and finite beyond it."""
    Y, n_para2 = 1.3, 0.6
    X_fold = 1 + Y**2 * (1 - n_para2) ** 2 / (4 * n_para2)
    grads = jax.grad(dispersion_relation_cold, argnums=(0, 1, 2))
    for X in X_fold - np.array([1e-1, 1e-2, 1e-4, 1e-8]):
        for mode in ["O", "X"]:
            root = refractive_index_perp_squared_cold(n_para2, X, Y, mode)
            np.testing.assert_allclose(
                dispersion_relation_cold(root, n_para2, X, Y, "X"), 0.0, atol=1e-12
            )
    S = 1 - Y**2 - X_fold
    b = (
        (S + (1 - Y**2) * (1 - X_fold)) * n_para2
        - (1 - Y - X_fold) * (1 + Y - X_fold)
        - (1 - X_fold) * S
    )
    double_root = -b / (2 * S)
    assert np.isnan(refractive_index_perp_squared_cold(n_para2, X_fold + 1e-3, Y, "X"))
    np.testing.assert_allclose(
        dispersion_relation_cold(double_root, n_para2, X_fold, Y, "X"), 0.0, atol=1e-12
    )
    d_perp2, _, d_X = grads(double_root, n_para2, X_fold, Y, "X")
    assert np.isfinite(d_perp2) and np.isfinite(d_X) and d_X != 0
    # the slope in N_perp^2 vanishes at the fold: the ray turns in N_perp^2 there
    np.testing.assert_allclose(d_perp2, 0.0, atol=1e-8)
    for X in X_fold + np.array([1e-8, 1e-3, 0.1]):
        value = dispersion_relation_cold(double_root, n_para2, X, Y, "X")
        assert np.isfinite(value) and value != 0
        assert np.all(np.isfinite(grads(double_root, n_para2, X, Y, "X")))


@pytest.mark.parametrize("mode", ["O", "X"])
def test_dispersion_relation_at_the_upper_hybrid_layer(mode):
    """At X = 1 - Y^2 the quadratic degenerates (the X-mode root diverges); D_sigma
    stays finite with finite gradients and vanishes on the remaining root."""
    Y, n_para2 = 0.6, 0.2
    X = 1 - Y**2
    root_O = refractive_index_perp_squared_cold(n_para2, X, Y, "O")
    assert np.isfinite(root_O)
    np.testing.assert_allclose(
        dispersion_relation_cold(root_O, n_para2, X, Y, "O"), 0.0, atol=1e-12
    )
    for n_perp2 in [0.0, 0.5, 2.0, 50.0]:
        value, grads = jax.value_and_grad(dispersion_relation_cold, argnums=(0, 1, 2))(
            n_perp2, n_para2, X, Y, mode
        )
        assert np.isfinite(value) and np.all(np.isfinite(grads))


def test_dispersion_tensor_stix():
    """Test the dispersion tensor in Stix coordinates."""
    refractive_index_perp = 0.6
    refractive_index_para = 0.4
    dielectric_tensor = jnp.array(
        [[1.0, 0.1j, 0.1], [-0.1j, 1.0, 0.0], [0.1, 0.0, 1.2]],
        dtype=jnp.complex128,
    )

    dispersion_tensor = dispersion_tensor_stix(
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        dielectric_tensor=dielectric_tensor,
    )

    assert dispersion_tensor.shape == (3, 3)

    # compare the vectorial form of the equation (Stix convention)
    # D_{ij} = \epsilon_{ij} - N^2 \delta_{ij} + N_i N_j
    # to the one implemented in the tested function
    N = jnp.array(
        [refractive_index_perp, 0, refractive_index_para], dtype=jax.numpy.complex128
    )
    nn = jnp.eye(3) * jnp.linalg.norm(N) ** 2 - jnp.einsum("i,j->ij", N, N)
    expected_dispersion_tensor = dielectric_tensor - nn

    np.testing.assert_allclose(
        dispersion_tensor,
        expected_dispersion_tensor,
        rtol=0,
        atol=_MACHINE_PRECISION,
    )
