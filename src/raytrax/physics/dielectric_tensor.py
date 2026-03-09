"""Cold and weakly-relativistic warm plasma dielectric tensors in Stix coordinates."""

import jax
import jax.numpy as jnp
import jaxtyping as jt
from jax.scipy.special import gamma

from raytrax.math import shkarofsky
from raytrax.types import ScalarFloat

ScalarInt = int | jt.Int[jax.Array, " "]


def cold_dielectric_tensor(
    frequency: ScalarFloat,
    plasma_frequency: ScalarFloat,
    cyclotron_frequency: ScalarFloat,
) -> jt.Complex[jax.Array, "3 3"]:
    """Returns the cold plasma dielectric tensor in Stix coordinates.

    Args:
        frequency: Wave frequency in Hz
        plasma_frequency: Electron plasma frequency in Hz
        cyclotron_frequency: Electron cyclotron frequency in Hz

    Returns:
        3x3 complex dielectric tensor in Stix coordinates
    """
    X = (plasma_frequency / frequency) ** 2
    Y = cyclotron_frequency / frequency

    R = 1 - X / (1 + Y)
    L = 1 - X / (1 - Y)
    S = 0.5 * (R + L)
    D = 0.5 * (R - L)
    P = 1 - X

    return jnp.array([[S, -1j * D, 0], [1j * D, S, 0], [0, 0, P]], dtype=jnp.complex128)


def weakly_relativistic_dielectric_tensor(
    frequency: ScalarFloat,
    plasma_frequency: ScalarFloat,
    cyclotron_frequency: ScalarFloat,
    thermal_velocity: ScalarFloat,
    refractive_index_para: ScalarFloat,
    max_s: int = 1,
    max_k: int = 1,
) -> jt.Complex[jax.Array, "3 3"]:
    """Computes the dielectric tensor in weakly relativistic approximation.

    Taken from Krivenski and Orefice, J. Plasma Physics (1983), vol. 30, part 1, pp.
    125-131.

    Args:
        frequency: Wave frequency in Hz
        plasma_frequency: Electron plasma frequency in Hz
        cyclotron_frequency: Electron cyclotron frequency in Hz
        thermal_velocity: electron thermal velocity normalized to c
        refractive_index_para: Refractive index parallel to the magnetic field
        max_s: Maximum value of s for the Shkarofsky functions
        max_k: Maximum value of k for the Shkarofsky functions

    Returns:
        3x3 complex dielectric tensor in Stix coordinates
    """
    w_p = 2 * jnp.pi * plasma_frequency
    w = 2 * jnp.pi * frequency
    w_c = 2 * jnp.pi * cyclotron_frequency
    # mu = m_0 c^2 / T_e = m_0 c^2 / (1/2 m_0 v_th^2) = 2 / (v_th / c)^2
    mu = 2 / thermal_velocity**2
    D = jnp.zeros((3, 3), dtype=jnp.complex128)
    n_par = refractive_index_para
    lam = (n_par * w / w_c) ** 2 / mu
    # lam*mu = (n_par*w/w_c)², so sqrt(lam*mu) = |n_par|*w/w_c exactly.
    # jnp.abs avoids sqrt'(0)=Inf at n_par=0 (which causes 0*Inf=NaN in adjoint).
    sqrt_lam_mu = jnp.abs(n_par) * w / w_c

    # computing the Shkarofsky functions F_{q+1/2} at s=0 for all k required
    q_max = max_k + 3  # because we need up to F_{q+2}
    Fq = shkarofsky.shkarofsky(0, mu=mu, n_par=n_par, w=w, w_c=w_c, q_max=q_max)

    # D_33 is the only non-zero component since b(0, 0) = 0
    # Q^+_{0,0}(h=2)
    q = 1  # s + k + 1 = 0 + 0 + 1
    Q_h2 = Fq[q + 1] / mu + n_par**2 * (Fq[q + 2] + Fq[q] - 2 * Fq[q + 1])
    D = D.at[2, 2].set(mu * a_shkarofsky(0, 0) * Q_h2)

    # s = 0, sum over k
    for k in range(1, max_k + 1):
        b_0k_lam = b_shkarofsky(0, k) * lam ** (k - 1)
        a_0k_lam = a_shkarofsky(0, k) * lam ** (k - 1)
        Q_h0 = Fq[q]
        Q_h1 = (Fq[q] - Fq[q + 1]) * n_par
        Q_h2 = Fq[q + 1] / mu + (Fq[q + 2] + Fq[q] - 2 * Fq[q + 1]) * n_par**2
        # D_11, D_12, D_13 vanish
        # D_22
        D = D.at[1, 1].set(-mu * b_0k_lam * Q_h0)
        # D_23
        D = D.at[1, 2].set(1j * sqrt_lam_mu * k * a_0k_lam * Q_h1)
        # D_33
        D = D.at[2, 2].add(-mu * lam * a_0k_lam * Q_h2)

    # sum over s > 0, k >= 0
    for s in range(1, max_s + 1):
        # computing the Shkarofsky functions F_{q+1/2} for all k required
        q_max = s + max_k + 3  # because we need up to F_{q+2}
        Fq_s = shkarofsky.shkarofsky(s, mu=mu, n_par=n_par, w=w, w_c=w_c, q_max=q_max)
        Fq_minus_s = shkarofsky.shkarofsky(
            -s, mu=mu, n_par=n_par, w=w, w_c=w_c, q_max=q_max
        )

        for k in range(max_k + 1):
            a_sk_lam = a_shkarofsky(s, k) * lam ** (s + k - 1)
            b_sk_lam = b_shkarofsky(s, k) * lam ** (s + k - 1)

            q = s + k + 1
            Q_h0_s = Fq_s[q]
            Q_h0_minus_s = Fq_minus_s[q]
            Q_h1_s = (Fq_s[q] - Fq_s[q + 1]) * n_par
            Q_h1_minus_s = (Fq_minus_s[q] - Fq_minus_s[q + 1]) * n_par
            Q_h2_s = (
                Fq_s[q + 1] / mu + (Fq_s[q + 2] + Fq_s[q] - 2 * Fq_s[q + 1]) * n_par**2
            )
            Q_h2_minus_s = (
                Fq_minus_s[q + 1] / mu
                + (Fq_minus_s[q + 2] + Fq_minus_s[q] - 2 * Fq_minus_s[q + 1]) * n_par**2
            )
            D = D.at[0, 0].add(s**2 * a_sk_lam * (Q_h0_s + Q_h0_minus_s))
            # sign error in Travis?
            D = D.at[0, 1].add(-1j * s * (s + k) * a_sk_lam * (Q_h0_s - Q_h0_minus_s))
            D = D.at[1, 1].add(b_sk_lam * (Q_h0_s + Q_h0_minus_s))
            D = D.at[0, 2].add(sqrt_lam_mu * s * a_sk_lam * (Q_h1_s - Q_h1_minus_s))
            # sign error in Travis?
            D = D.at[1, 2].add(
                1j * sqrt_lam_mu * (s + k) * a_sk_lam * (Q_h1_s + Q_h1_minus_s)
            )
            D = D.at[2, 2].add(lam * mu * a_sk_lam * (Q_h2_s + Q_h2_minus_s))

    # set other off-diagonal components according to symmetry
    # D_21 = -D_12, eq. (44)
    D = D.at[1, 0].set(-D[0, 1])
    # D_31 = D_13, eq. (46)
    D = D.at[2, 0].set(D[0, 2])
    # D_32 = -D_23, eq. (47)
    D = D.at[2, 1].set(-D[1, 2])

    eye = jnp.eye(3, dtype=jnp.complex128)
    return eye - mu * (w_p**2 / w**2) * D  # eq. (42)


def a_shkarofsky(n: int, k: int) -> ScalarFloat:
    """The function a defined in eq. (32) of Shkarofsky [1].

    It is related to the function a in Krivenski and Orefice [2] by:

    .. math::
        a_S(n, k) = (n + k)! 2^{n + k} a_{KO}(n, k)

    [1] Shkarofsky, . Plasma Physics (1986), vol. 35, part 2, pp. 319-331
    [2] Krivenski, A. and Orefice, A. J. Plasma Physics (1983), vol. 30, part 1,
        pp. 125-131.
    """
    sign = (-1) ** k
    numerator = gamma(k + n + 0.5)
    denominator = gamma(k + 1) * gamma(n + k / 2 + 1) * gamma(k / 2 + n + 0.5) * (2**n)
    return sign * numerator / denominator


def b_shkarofsky(n: int, k: int) -> ScalarFloat:
    """The analogon of the function b in Krivenski and Orefice [2] when using the
    function a defined in Shkarofsky [1].

    It is related to the function b in Krivenski and Orefice [2] by:

    .. math::
        b_S(n, k) = (n + k)! 2^{n + k} b_{KO}(n, k)

    [1] Shkarofsky, . Plasma Physics (1986), vol. 35, part 2, pp. 319-331
    [2] Krivenski, A. and Orefice, A. J. Plasma Physics (1983), vol. 30, part 1,
        pp. 125-131.
    """
    prefactor = (k + n) ** 2 - k * (k + 2 * n) / (2 * n + 2 * k - 1)
    return prefactor * a_shkarofsky(n, k)
