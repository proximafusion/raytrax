from functools import partial

import jax
import jax.numpy as jnp

# Number of terms in the ascending power series for J_v(z).
# 50 terms gives ~1e-14 relative accuracy for |z| up to 10.
_JV_SERIES_NTERMS = 50


def _jv_series(v, z):
    """Bessel J_v(z) via ascending power series for non-negative integer order.

    J_v(z) = sum_{k=0}^{N} (-1)^k / (k! * Gamma(k+v+1)) * (z/2)^(2k+v)

    Pure JAX implementation — no callbacks, fully differentiable, vmap-friendly.
    Handles both scalar and array z inputs via broadcasting.
    """
    half_z = z / 2.0
    k = jnp.arange(_JV_SERIES_NTERMS, dtype=jnp.float64)
    # Reshape k for broadcasting: (N, 1, 1, ...) to match z's dimensions
    k = k.reshape((-1,) + (1,) * jnp.ndim(z))
    # Compute in log-space for numerical stability
    log_terms = (2 * k + v) * jnp.log(jnp.maximum(jnp.abs(half_z), 1e-300)) - (
        jax.scipy.special.gammaln(k + 1) + jax.scipy.special.gammaln(k + v + 1)
    )
    signs = (-1.0) ** k
    return jnp.sum(signs * jnp.exp(log_terms), axis=0)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def jv_jax(v, z):
    """Compute the Bessel function of the first kind of integer order.

    Uses a pure JAX ascending power series — no scipy callbacks, so this is
    fully compatible with ``jax.vmap`` and ``jax.jit`` without sequential
    callback overhead.

    Args:
        v: integer order of the Bessel function
        z: real argument
    """
    # For negative integer order: J_{-n}(z) = (-1)^n J_n(z)
    abs_v = abs(v)
    # For negative z: J_n(-z) = (-1)^n J_n(z) for integer n
    abs_z = jnp.abs(z)
    result = _jv_series(abs_v, abs_z)
    # Apply sign for negative z: (-1)^abs_v
    result = jnp.where(z < 0, (-1.0) ** abs_v * result, result)
    # Apply sign for negative order: (-1)^abs_v
    return jnp.where(v < 0, (-1.0) ** abs_v * result, result)


@jv_jax.defjvp
def jv_jax_jvp(v, primals, tangents):
    """Custom JVP rule: dJ_v/dz = 0.5 * (J_{v-1}(z) - J_{v+1}(z))."""
    (z,) = primals
    (dz,) = tangents
    jv = jv_jax(v, z)
    djv_dz = 0.5 * (jv_jax(v - 1, z) - jv_jax(v + 1, z))
    return jv, djv_dz * dz


djv_jax = jax.grad(jv_jax, argnums=1)


# --- K_v power series for small z (DLMF 10.31.1) ---

_EULER_MASCHERONI = 0.5772156649015328606065120900824024

# Number of terms for the K_v power series (small z).
_KV_SERIES_NTERMS = 50

# Switch from series to asymptotic expansion at this z.
_KV_SWITCHOVER = 10.0


def _k0_series(z):
    """K_0(z) via power series (DLMF 10.31.1 with n=0).

    K_0(z) = -(ln(z/2) + gamma) I_0(z) + sum_{k=1}^N H_k (z/2)^{2k} / (k!)^2

    where H_k = 1 + 1/2 + ... + 1/k are harmonic numbers.
    """
    half_z = z / 2.0
    log_half_z = jnp.log(jnp.maximum(jnp.abs(half_z), 1e-300))

    k = jnp.arange(_KV_SERIES_NTERMS, dtype=jnp.float64)
    k_r = k.reshape((-1,) + (1,) * jnp.ndim(z))

    # (z/2)^{2k} / (k!)^2
    log_terms = 2 * k_r * log_half_z - 2 * jax.scipy.special.gammaln(k_r + 1)
    terms = jnp.exp(log_terms)

    I0 = jnp.sum(terms, axis=0)

    # Harmonic numbers: H_0=0, H_k = 1 + 1/2 + ... + 1/k
    H = jnp.concatenate(
        [
            jnp.array([0.0]),
            jnp.cumsum(1.0 / jnp.arange(1, _KV_SERIES_NTERMS, dtype=jnp.float64)),
        ]
    )
    H_r = H.reshape((-1,) + (1,) * jnp.ndim(z))

    harm_sum = jnp.sum(H_r * terms, axis=0)

    return -(log_half_z + _EULER_MASCHERONI) * I0 + harm_sum


def _k1_series(z):
    """K_1(z) via power series (DLMF 10.31.1 with n=1).

    K_1(z) = 1/z + ln(z/2) I_1(z)
             - (z/4) sum_{k=0}^N [psi(k+1)+psi(k+2)] (z^2/4)^k / (k!(k+1)!)
    """
    half_z = z / 2.0
    log_half_z = jnp.log(jnp.maximum(jnp.abs(half_z), 1e-300))

    k = jnp.arange(_KV_SERIES_NTERMS, dtype=jnp.float64)
    k_r = k.reshape((-1,) + (1,) * jnp.ndim(z))

    # I_1(z) = sum_k (z/2)^{2k+1} / (k! (k+1)!)
    log_I1_terms = (2 * k_r + 1) * log_half_z - (
        jax.scipy.special.gammaln(k_r + 1) + jax.scipy.special.gammaln(k_r + 2)
    )
    I1 = jnp.sum(jnp.exp(log_I1_terms), axis=0)

    # Harmonic numbers
    H = jnp.concatenate(
        [
            jnp.array([0.0]),
            jnp.cumsum(1.0 / jnp.arange(1, _KV_SERIES_NTERMS, dtype=jnp.float64)),
        ]
    )

    # psi(k+1) + psi(k+2) = -2*gamma + 2*H_k + 1/(k+1)
    psi_sum = -2 * _EULER_MASCHERONI + 2 * H + 1.0 / (k + 1)
    psi_sum_r = psi_sum.reshape((-1,) + (1,) * jnp.ndim(z))

    # (z/2)^{2k} / (k! (k+1)!)
    log_sum_terms = 2 * k_r * log_half_z - (
        jax.scipy.special.gammaln(k_r + 1) + jax.scipy.special.gammaln(k_r + 2)
    )
    series = jnp.sum(psi_sum_r * jnp.exp(log_sum_terms), axis=0)

    return 1.0 / z + log_half_z * I1 - (z / 4.0) * series


def _kv_series(v, z):
    """K_v(z) for small z using K_0/K_1 series + upward recurrence.

    Uses the power series for K_0 and K_1 (DLMF 10.31.1) and the recurrence
    K_{n+1}(z) = K_{n-1}(z) + (2n/z) K_n(z) for higher integer orders.
    v must be a Python integer (passed via nondiff_argnums).
    """
    abs_v = abs(v)  # K_{-v} = K_v
    k0 = _k0_series(z)
    k1 = _k1_series(z)

    if abs_v == 0:
        return k0
    if abs_v == 1:
        return k1

    # Upward recurrence: K_{n+1} = K_{n-1} + (2n/z) K_n
    k_prev, k_curr = k0, k1
    for n in range(1, abs_v):
        k_next = k_prev + (2.0 * n / z) * k_curr
        k_prev, k_curr = k_curr, k_next
    return k_curr


# --- Asymptotic expansion for K_v(z) * exp(z) (large z) ---

# Number of terms in the asymptotic expansion for K_v(z) * exp(z).
_KVE_ASYMP_NTERMS = 20


def _kve_asymptotic(v, z):
    """Asymptotic expansion for kve(v, z) = K_v(z) * exp(z).

    kve(v, z) ~ sqrt(pi / (2z)) * sum_{k=0}^{N} a_k(v) / z^k

    where a_k(v) = prod_{j=1}^{k} (4v^2 - (2j-1)^2) / (8^k * k!)

    Accurate for large z (our application has z = mu = 2/v_th^2 > 50).
    Pure JAX implementation — no callbacks, fully vmap-compatible.
    """
    four_v2 = 4.0 * v * v

    # Build coefficients a_k iteratively
    # a_0 = 1, a_k = a_{k-1} * (4v^2 - (2k-1)^2) / (8k)
    def body(carry, k):
        a_prev = carry
        a_k = a_prev * (four_v2 - (2.0 * k - 1.0) ** 2) / (8.0 * k)
        return a_k, a_k

    _, coeffs = jax.lax.scan(
        body, jnp.ones_like(z), jnp.arange(1, _KVE_ASYMP_NTERMS + 1, dtype=jnp.float64)
    )
    # coeffs has shape (N, *z.shape); prepend a_0=1
    a0 = jnp.ones_like(z)[jnp.newaxis]
    all_coeffs = jnp.concatenate([a0, coeffs], axis=0)  # (N+1, *z.shape)

    # Compute sum: a_0 + a_1/z + a_2/z^2 + ...
    k = jnp.arange(_KVE_ASYMP_NTERMS + 1, dtype=jnp.float64)
    # Reshape k for broadcasting
    k = k.reshape((-1,) + (1,) * jnp.ndim(z))
    powers = z ** (-k)
    result = jnp.sum(all_coeffs * powers, axis=0)
    return jnp.sqrt(jnp.pi / (2.0 * z)) * result


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def kv_jax(v, z):
    """Compute the modified Bessel function of the second kind of integer order.

    Uses a power series for small z and an asymptotic expansion for large z.
    Pure JAX implementation — fully compatible with ``jax.vmap`` and ``jax.jit``.

    Args:
        v: integer order of the Bessel function
        z: real positive argument
    """
    return jnp.where(
        z < _KV_SWITCHOVER,
        _kv_series(v, z),
        _kve_asymptotic(v, z) * jnp.exp(-z),
    )


@kv_jax.defjvp
def kv_jax_jvp(v, primals, tangents):
    """Custom JVP rule for the modified Bessel function of the second kind."""
    (z,) = primals
    (dz,) = tangents
    kv_val = kv_jax(v, z)
    dkv_dz = -0.5 * (kv_jax(v - 1, z) + kv_jax(v + 1, z))
    return kv_val, dkv_dz * dz


dkv_jax = jax.grad(kv_jax, argnums=1)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def kve_jax(v, z):
    """Compute the exponentially scaled modified Bessel function: kve(v, z) = kv(v, z) * exp(z).

    Uses a pure JAX implementation that switches between power series (small z)
    and asymptotic expansion (large z) — no scipy callbacks, fully
    compatible with ``jax.vmap`` and ``jax.jit``.

    Avoids underflow for large z where kv(v, z) would underflow to 0.
    """
    return jnp.where(
        z < _KV_SWITCHOVER,
        _kv_series(v, z) * jnp.exp(z),
        _kve_asymptotic(v, z),
    )


@kve_jax.defjvp
def kve_jax_jvp(v, primals, tangents):
    """Custom JVP rule for the exponentially scaled modified Bessel function.

    d/dz [kv(v,z)*exp(z)] = [-0.5*(kv(v-1,z)+kv(v+1,z)) + kv(v,z)] * exp(z)
                           = -0.5*(kve(v-1,z)+kve(v+1,z)) + kve(v,z)
    """
    (z,) = primals
    (dz,) = tangents
    kve_val = kve_jax(v, z)
    dkve_dz = -0.5 * (kve_jax(v - 1, z) + kve_jax(v + 1, z)) + kve_val
    return kve_val, dkve_dz * dz
