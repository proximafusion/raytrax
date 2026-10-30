import jax
import jax.numpy as jnp
import numpy as np
import pytest

from raytrax.physics.dispersion import (
    _dispersion_appleton_hartee,
    dispersion_cold,
    dispersion_tensor_stix,
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
