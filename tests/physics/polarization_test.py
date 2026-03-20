import jax
import jax.numpy as jnp
import numpy as np
import pytest

from raytrax.physics import dielectric_tensor, dispersion, polarization

jax.config.update("jax_enable_x64", True)

_FREQUENCY = 220.0e9  # Hz
_PLASMA_FREQUENCY = 160.0e9  # Hz
_CYCLOTRON_FREQUENCY = 232.0e9  # Hz


def _cold_setup(mode: str):
    """Return (n_perp, n_para, eps) at the Appleton-Hartree root for the given mode."""
    X = (_PLASMA_FREQUENCY / _FREQUENCY) ** 2
    Y = _CYCLOTRON_FREQUENCY / _FREQUENCY
    # perpendicular propagation (sin²θ = 1): simplest non-trivial case
    n2 = dispersion._dispersion_appleton_hartee(X=X, Y=Y, sin2theta=1.0, mode=mode)
    n_perp = float(jnp.sqrt(n2))
    n_para = 0.0
    eps = dielectric_tensor.cold_dielectric_tensor(
        frequency=_FREQUENCY,
        plasma_frequency=_PLASMA_FREQUENCY,
        cyclotron_frequency=_CYCLOTRON_FREQUENCY,
    )
    return n_perp, n_para, eps


@pytest.mark.parametrize("mode", ["X", "O"])
def test_polarization_shape_and_norm(mode):
    n_perp, n_para, eps = _cold_setup(mode)
    pol = polarization.polarization(
        dielectric_tensor=eps,
        refractive_index_perp=n_perp,
        refractive_index_para=n_para,
        frequency=_FREQUENCY,
        cyclotron_frequency=_CYCLOTRON_FREQUENCY,
        mode=mode,
    )
    assert pol.shape == (3,)
    np.testing.assert_allclose(jnp.linalg.norm(pol), 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mode", ["X", "O"])
def test_polarization_is_null_vector_of_dispersion_tensor(mode):
    """D(N) @ e ≈ 0: the polarization vector must lie in the null space of D."""
    n_perp, n_para, eps = _cold_setup(mode)
    pol = polarization.polarization(
        dielectric_tensor=eps,
        refractive_index_perp=n_perp,
        refractive_index_para=n_para,
        frequency=_FREQUENCY,
        cyclotron_frequency=_CYCLOTRON_FREQUENCY,
        mode=mode,
    )
    D = dispersion.dispersion_tensor_stix(
        refractive_index_perp=n_perp,
        refractive_index_para=n_para,
        dielectric_tensor=eps,
    )
    residual = D @ pol
    np.testing.assert_allclose(jnp.abs(residual), 0.0, rtol=0, atol=1e-10)
