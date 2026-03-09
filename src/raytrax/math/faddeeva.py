"""Plasma dispersion function Z(ζ) and its derivative, computed via the Faddeeva function.

The Faddeeva function w(z) = exp(-z²)·erfc(-iz) is implemented in pure JAX
using Poppe & Wijers (1990) Algorithm 680.

Reference:
    G. P. M. Poppe and C. M. J. Wijers, "More efficient computation of the
    complex error function", ACM Trans. Math. Softw. 16(1), 38-46 (1990).
"""

import jax
import jax.numpy as jnp

_ERRF = 1.12837916709551257  # 2/sqrt(pi)
_XLIM = 5.33
_YLIM = 4.29


def _wofz_region2(x: jax.Array, y: jax.Array):
    """9-step backward continued fraction for |z| outside the inner region."""
    xh = y
    yh = x

    def body(i, state):
        n = jnp.float64(10 - i)  # i=1..9  →  n=9..1
        Rx, Ry = state
        Tx = xh + n * Rx
        Ty = yh - n * Ry
        Tn = Tx * Tx + Ty * Ty
        return 0.5 * Tx / Tn, 0.5 * Ty / Tn

    Rx, Ry = jax.lax.fori_loop(1, 10, body, (jnp.zeros_like(x), jnp.zeros_like(x)))
    return _ERRF * Rx, _ERRF * Ry


def _wofz_region1(x: jax.Array, y: jax.Array):
    """Inner-region Shuman-sum algorithm for y < 4.29 and x < 5.33.

    The loop runs at most 31 steps (nu ≤ 31); a static fori_loop with
    masking replaces the data-dependent while_loop.
    """
    q = (1.0 - y / _YLIM) * jnp.sqrt(jnp.maximum(0.0, 1.0 - (x / _XLIM) ** 2))
    h = 1.0 / (3.2 * q)
    nc = 7.0 + jnp.floor(23.0 * q)  # ∈ [7, 30]
    nu = 10.0 + jnp.floor(21.0 * q)  # ∈ [10, 31]
    xl = jnp.power(h, 1.0 - nc)
    xh = y + 0.5 / h
    yh = x

    _MAX = 32

    def body(i, state):
        n = jnp.float64(_MAX - 1 - i)
        Rx, Ry, xl_, Sx, Sy = state

        Tx = xh + n * Rx
        Ty = yh - n * Ry
        Tn = Tx * Tx + Ty * Ty
        Rx2 = 0.5 * Tx / Tn
        Ry2 = 0.5 * Ty / Tn

        active = (n >= 1.0) & (n <= nu)
        accum = active & (n <= nc)

        Saux = Sx + xl_
        Sx2 = jnp.where(accum, Rx2 * Saux - Ry2 * Sy, Sx)
        Sy2 = jnp.where(accum, Rx2 * Sy + Ry2 * Saux, Sy)
        xl2 = jnp.where(accum, h * xl_, xl_)
        Rx2 = jnp.where(active, Rx2, Rx)
        Ry2 = jnp.where(active, Ry2, Ry)

        return Rx2, Ry2, xl2, Sx2, Sy2

    init = (
        jnp.zeros_like(x),
        jnp.zeros_like(x),
        xl,
        jnp.zeros_like(x),
        jnp.zeros_like(x),
    )
    _, _, _, Sx, Sy = jax.lax.fori_loop(0, _MAX, body, init)
    return _ERRF * Sx, _ERRF * Sy


@jax.custom_jvp
def wofz_jax(z):
    """Compute the Faddeeva function w(z) = exp(-z²)·erfc(-iz) for complex z."""
    z = jnp.asarray(z, dtype=jnp.complex128)
    x_re = jnp.real(z)
    x_im = jnp.imag(z)
    x = jnp.abs(x_re)
    y = jnp.abs(x_im)

    inner = (y < _YLIM) & (x < _XLIM)
    Wx1, Wy1 = _wofz_region1(x, y)
    Wx2, Wy2 = _wofz_region2(x, y)
    Wx = jnp.where(inner, Wx1, Wx2)
    Wy = jnp.where(inner, Wy1, Wy2)

    # Im(z) = 0: Re(w(t)) = exp(-t²)
    Wx = jnp.where(y == 0.0, jnp.exp(-x * x), Wx)

    # Odd symmetry in Re(z): Im(w(-x+iy)) = -Im(w(x+iy))
    Wy = jnp.sign(x_re) * Wy

    # Reflection for Im(z) < 0: w(a-iy) = 2exp(-(a-iy)²) - w(-a+iy)
    e2 = 2.0 * jnp.exp(y * y - x * x)
    Wx_neg = e2 * jnp.cos(2.0 * x * y) - Wx
    Wy_neg = jnp.sign(x_re) * e2 * jnp.sin(2.0 * x * y) + Wy
    Wx = jnp.where(x_im < 0.0, Wx_neg, Wx)
    Wy = jnp.where(x_im < 0.0, Wy_neg, Wy)

    return Wx + 1j * Wy


@wofz_jax.defjvp
def wofz_jax_jvp(primals, tangents):
    """Custom JVP rule: w'(z) = 2i/√π - 2z·w(z)."""
    (z,) = primals
    (dz,) = tangents
    w_z = wofz_jax(z)
    w_prime = 2j / jnp.sqrt(jnp.pi) - 2 * z * w_z
    return w_z, w_prime * dz


def plasma_dispersion_function(z):
    """Compute the plasma dispersion function Z for a complex argument z."""
    return 1j * jnp.sqrt(jnp.pi) * wofz_jax(z)


def plasma_dispersion_function_derivative(z):
    """Compute the derivative of the plasma dispersion function Z'."""
    return -2 * (1 + z * plasma_dispersion_function(z))
