"""Tests for the Faddeeva function implementation.

Covers the real axis, all four quadrants, near-origin, large |z|, and
physics-relevant Shkarofsky argument ranges (|z| ~ 1–10).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import wofz as scipy_wofz

jax.config.update("jax_enable_x64", True)

from raytrax.math.faddeeva import wofz_jax  # noqa: E402

# ---------------------------------------------------------------------------
# Test points covering the full complex plane
# ---------------------------------------------------------------------------

# fmt: off
_CASES = [
    # (z,                       description)
    # Real axis
    ( 0.0+0j,                  "origin"),
    ( 0.1+0j,                  "real small pos"),
    ( 1.0+0j,                  "real unit"),
    ( 2.0+0j,                  "real moderate"),
    ( 5.0+0j,                  "real large"),
    (10.0+0j,                  "real very large"),
    (-0.5+0j,                  "real small neg"),
    (-2.0+0j,                  "real neg moderate"),
    (-5.0+0j,                  "real large neg"),
    # Pure imaginary
    ( 0+0.1j,                  "imag small pos"),
    ( 0+1.0j,                  "imag unit"),
    ( 0+3.0j,                  "imag moderate"),
    ( 0+8.0j,                  "imag large"),
    ( 0-0.5j,                  "imag small neg"),
    ( 0-2.0j,                  "imag neg"),
    ( 0-8.0j,                  "imag large neg"),
    # Q1: Re>0, Im>0
    ( 1.0+1.0j,                "Q1 symmetric"),
    ( 2.0+0.5j,                "Q1 small Im"),
    ( 0.5+3.0j,                "Q1 large Im"),
    ( 3.0+3.0j,                "Q1 equal large"),
    ( 1.5+2.5j,                "Q1 general"),
    # Q2: Re<0, Im>0
    (-1.0+1.0j,                "Q2 symmetric"),
    (-3.0+0.5j,                "Q2 small Im"),
    (-0.5+4.0j,                "Q2 large Im"),
    # Q3: Re<0, Im<0
    (-1.0-1.0j,                "Q3 symmetric"),
    (-2.0-3.0j,                "Q3 general"),
    (-0.5-5.0j,                "Q3 large neg Im"),
    # Q4: Re>0, Im<0
    ( 1.0-1.0j,                "Q4 symmetric"),
    ( 3.0-0.5j,                "Q4 small neg Im"),
    ( 0.5-4.0j,                "Q4 large neg Im"),
    ( 2.0-2.0j,                "Q4 equal"),
    # Near real axis (Im → 0+)
    ( 1.0+1e-3j,               "near real axis pos Q1"),
    ( 3.0+1e-3j,               "near real axis pos Q1 large"),
    (-2.0+1e-3j,               "near real axis pos Q2"),
    # Near real axis (Im → 0-)
    ( 1.0-1e-3j,               "near real axis neg Q4"),
    ( 3.0-1e-3j,               "near real axis neg Q4 large"),
    (-2.0-1e-3j,               "near real axis neg Q3"),
    # Near origin
    ( 0.01+0.01j,              "near origin Q1"),
    ( 0.001+0j,                "near origin real"),
    ( 0+0.001j,                "near origin imag"),
    # Large |z| (asymptotic regime)
    (10.0+1.0j,                "large |z| Q1"),
    ( 1.0+10.0j,               "large imag Q1"),
    (15.0+0.1j,                "large real near real axis"),
    (-10.0+1.0j,               "large Q2"),
    (10.0-1.0j,                "large Q4"),
    (-5.0-5.0j,                "large Q3"),
    # Physics-relevant: Shkarofsky arguments psi-phi, -psi-phi, -phi
    # Typical: psi~1.5 (real), phi~1.6 (can be complex)
    ( 1.5+0j,                  "psi typical"),
    (-1.5+0j,                  "-psi typical"),
    ( 0-1.6j,                  "-phi imaginary"),
    ( 1.5-1.6j,                "psi-phi Q4"),
    (-1.5-1.6j,                "-(psi+phi) Q3"),
    ( 1.5+1.6j,                "psi-phi* Q1"),
    (-1.5+1.6j,                "-(psi+phi*) Q2"),
    # mu=51 (T=10 keV): psi up to ~5
    ( 4.0-2.0j,                "high mu Q4"),
    (-4.0-2.0j,                "high mu Q3"),
    # Harmonic resonance region: phi near real (alpha_s > 0, large)
    ( 2.0+0j,                  "phi real large"),
    ( 5.0+0j,                  "phi real very large"),
]
# fmt: on

_IDS = [desc for _, desc in _CASES]

_RTOL = 1e-10
_ATOL = 1e-12  # absolute floor for small values


@pytest.mark.parametrize("z, desc", _CASES, ids=_IDS)
def test_wofz_accuracy(z, desc):
    """wofz_jax must match scipy.special.wofz to _RTOL relative error."""
    z_arr = jnp.array(z, dtype=jnp.complex128)
    got = complex(wofz_jax(z_arr))
    expected = scipy_wofz(complex(z))

    err = abs(got - expected)
    tol = _ATOL + _RTOL * abs(expected)
    assert err <= tol, (
        f"[{desc}] z={z}\n"
        f"  got      = {got}\n"
        f"  expected = {expected}\n"
        f"  abs err  = {err:.3e}  (tol={tol:.3e})"
    )


def test_wofz_reflection_identity():
    """w(z) + w(-z) = 2 exp(-z²) must hold for all z."""
    test_z = [1 + 1j, 2 - 3j, -1 + 0.5j, 0 + 2j, 3 + 0j, 1.5 - 1.6j]
    for z in test_z:
        z_arr = jnp.array(z, dtype=jnp.complex128)
        w_pos = complex(wofz_jax(z_arr))
        w_neg = complex(wofz_jax(-z_arr))
        expected = 2 * np.exp(-(z**2))
        actual = w_pos + w_neg
        assert abs(actual - expected) < 1e-9, (
            f"z={z}: w(z)+w(-z)={actual!r}, 2exp(-z²)={expected!r}"
        )


def test_wofz_jit_compatible():
    """wofz_jax must compile and run under jax.jit."""
    f = jax.jit(wofz_jax)
    z = jnp.array(1.5 + 2.3j, dtype=jnp.complex128)
    got = complex(f(z))
    expected = scipy_wofz(1.5 + 2.3j)
    assert abs(got - expected) < 1e-10


def test_wofz_vmap_compatible():
    """wofz_jax must support batched evaluation via vmap."""
    raw = [complex(z) for z, _ in _CASES[:20]]
    zs = jnp.array(raw, dtype=jnp.complex128)
    got = np.array(jax.vmap(wofz_jax)(zs))
    expected = np.array([scipy_wofz(z) for z in raw])
    np.testing.assert_allclose(got, expected, rtol=_RTOL, atol=_ATOL)


def test_wofz_jvp():
    """The custom JVP w'(z) = 2i/√π - 2z·w(z) must be consistent with finite differences."""
    z0 = jnp.array(1.0 + 0.5j, dtype=jnp.complex128)
    dz = jnp.array(0.3 + 0.2j, dtype=jnp.complex128)
    _, dw = jax.jvp(wofz_jax, (z0,), (dz,))

    eps = 1e-6
    dw_fd = (wofz_jax(z0 + eps * dz) - wofz_jax(z0 - eps * dz)) / (2 * eps)

    assert abs(complex(dw) - complex(dw_fd)) < 1e-8, (
        f"JVP mismatch: analytic={complex(dw)!r}, fd={complex(dw_fd)!r}"
    )


def test_wofz_no_nan_or_inf():
    """wofz_jax must never return NaN or Inf for any of the test points."""
    for z, desc in _CASES:
        z_arr = jnp.array(z, dtype=jnp.complex128)
        w = complex(wofz_jax(z_arr))
        assert np.isfinite(w.real) and np.isfinite(w.imag), (
            f"[{desc}] z={z}: got non-finite result {w}"
        )


def test_plasma_dispersion_function_values():
    """Z(z) = i√π·w(z): spot-check against scipy."""
    from raytrax.math.faddeeva import plasma_dispersion_function

    test_pts = [0.5 + 0j, 1.0 + 1.0j, 0 + 2.0j, 1.5 - 1.6j]
    for z in test_pts:
        z_arr = jnp.array(z, dtype=jnp.complex128)
        got = complex(plasma_dispersion_function(z_arr))
        expected = 1j * np.sqrt(np.pi) * scipy_wofz(z)
        assert abs(got - expected) < 1e-10, (
            f"Z({z}): got {got!r}, expected {expected!r}"
        )


def test_plasma_dispersion_function_derivative():
    """Z'(z) = -2(1 + z·Z(z)) must match finite differences."""
    from raytrax.math.faddeeva import (
        plasma_dispersion_function,
        plasma_dispersion_function_derivative,
    )

    z = jnp.linspace(-5, 5, 1000, dtype=jnp.complex128)
    dx = z[1] - z[0]
    Z = plasma_dispersion_function(z)
    Z_prime = plasma_dispersion_function_derivative(z)
    Z_prime_fd = (Z[2:] - Z[:-2]) / (2 * dx)
    np.testing.assert_allclose(Z_prime[1:-1], Z_prime_fd, rtol=0, atol=1e-3)
