"""Electromagnetic power flux vector (group velocity direction) in Stix coordinates."""

from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt

from raytrax.math import utils
from raytrax.physics import dielectric_tensor as dielectric_tensor_module
from raytrax.physics import dispersion
from raytrax.physics import polarization as polarization_module
from raytrax.types import ScalarFloat


def power_flux_hamiltonian_stix(
    refractive_index: jt.Float[jax.Array, "3"],
    frequency: ScalarFloat,
    plasma_frequency: ScalarFloat,
    cyclotron_frequency: ScalarFloat,
    thermal_velocity: ScalarFloat,
    mode: Literal["X", "O"],
    max_s: int = 2,
    max_k: int = 1,
) -> ScalarFloat:
    r"""Compute the Hamiltonian :math:`H = \hat{e}^* \cdot D^H(\mathbf{N}) \cdot \hat{e}`.

    Both the dielectric tensor :math:`\varepsilon(\mathbf{N})` and the polarization
    vector :math:`\hat{e}(\mathbf{N})` are recomputed from physical parameters at
    every evaluation of :math:`\mathbf{N}`, so that :func:`jax.grad` captures the full
    :math:`\partial\varepsilon/\partial\mathbf{N}` and
    :math:`\partial\hat{e}/\partial\mathbf{N}` contributions required for a correct
    power-flux gradient in a warm plasma.

    Args:
        refractive_index: Refractive index in Stix coordinates ``[N_perp, 0, N_para]``.
        frequency: Wave frequency in Hz.
        plasma_frequency: Electron plasma frequency in Hz.
        cyclotron_frequency: Electron cyclotron frequency in Hz.
        thermal_velocity: Normalized electron thermal velocity (v_th / c).
        mode: Wave polarization mode, ``"X"`` for extraordinary or ``"O"`` for ordinary.
        max_s: Maximum cyclotron harmonic for the KO tensor.
        max_k: Maximum Larmor-radius order for the KO tensor.

    Returns:
        Scalar real Hamiltonian value.
    """
    refractive_index_perp = refractive_index[0]
    refractive_index_para = refractive_index[2]

    eps = dielectric_tensor_module.weakly_relativistic_dielectric_tensor(
        frequency=frequency,
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
        thermal_velocity=thermal_velocity,
        refractive_index_para=refractive_index_para,
        refractive_index_perp=refractive_index_perp,
        max_s=max_s,
        max_k=max_k,
    )
    polarization_vector = polarization_module.polarization(
        dielectric_tensor=eps,
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        frequency=frequency,
        cyclotron_frequency=cyclotron_frequency,
        mode=mode,
    )
    dispersion_tensor = dispersion.dispersion_tensor_stix(
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        dielectric_tensor=eps,
    )
    dispersion_tensor_h = utils.hermitian_part(dispersion_tensor)
    return jnp.real(
        polarization_vector.conj() @ dispersion_tensor_h @ polarization_vector
    )


_grad_hamiltonian = jax.grad(power_flux_hamiltonian_stix, argnums=0)


def power_flux_vector_stix(
    refractive_index_perp: ScalarFloat,
    refractive_index_para: ScalarFloat,
    frequency: ScalarFloat,
    plasma_frequency: ScalarFloat,
    cyclotron_frequency: ScalarFloat,
    thermal_velocity: ScalarFloat,
    mode: Literal["X", "O"],
    max_s: int = 2,
    max_k: int = 1,
) -> jt.Float[jax.Array, "3"]:
    r"""Compute the dielectric power flux vector in Stix coordinates.

    In SI units (with Stix convention :math:`D = \varepsilon - N^2 I + NN`), the
    time-averaged Poynting vector is (see absorption.md):

    .. math::
        \langle\mathbf{S}\rangle = -\frac{\varepsilon_0\omega}{4}|A|^2
            \frac{\partial}{\partial \mathbf{k}}\left( e_i^* D_{ij}^{H} e_j \right)

    With :math:`\partial/\partial k = (c/\omega)\,\partial/\partial N`, this becomes
    :math:`\langle S\rangle = (\varepsilon_0 c / 2)|A|^2 \mathbf{F}` where

    .. math::
        \mathbf{F} = -\frac{1}{2} \frac{\partial}{\partial \mathbf{N}}
            \left( e_i^* D_{ij}^{H}(\mathbf{N}) e_j \right)

    Both the dielectric tensor :math:`\varepsilon(\mathbf{N})` and the polarization
    vector :math:`\hat{e}(\mathbf{N})` are recomputed at each :math:`\mathbf{N}` so
    that :func:`jax.grad` captures the full
    :math:`\partial\varepsilon/\partial\mathbf{N}` and
    :math:`\partial\hat{e}/\partial\mathbf{N}` contributions in a warm plasma.

    Args:
        refractive_index_perp: Perpendicular refractive index.
        refractive_index_para: Parallel refractive index.
        frequency: Wave frequency in Hz.
        plasma_frequency: Electron plasma frequency in Hz.
        cyclotron_frequency: Electron cyclotron frequency in Hz.
        thermal_velocity: Normalized electron thermal velocity (v_th / c).
        mode: Wave polarization mode, ``"X"`` for extraordinary or ``"O"`` for ordinary.
        max_s: Maximum cyclotron harmonic for the KO tensor.
        max_k: Maximum Larmor-radius order for the KO tensor.

    Returns:
        Power flux vector :math:`\mathbf{F} = -\tfrac{1}{2}\,\partial_\mathbf{N}(e^* D^H e)`
        in Stix coordinates. Dimensionless — it is the gradient of the Hamiltonian
        with respect to the refractive index vector :math:`\mathbf{N}` and carries
        no SI units. To obtain the physical Poynting vector, multiply by
        :math:`(\varepsilon_0 c / 2)|A|^2`.
    """
    refractive_index = jnp.array(
        [refractive_index_perp, 0.0, refractive_index_para],
        dtype=jnp.float64,
    )
    grad_N = _grad_hamiltonian(
        refractive_index,
        frequency,
        plasma_frequency,
        cyclotron_frequency,
        thermal_velocity,
        mode,
        max_s,
        max_k,
    )
    return -0.5 * grad_N
