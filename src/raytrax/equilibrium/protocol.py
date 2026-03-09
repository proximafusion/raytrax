"""Protocol definition for VMEC equilibrium objects (WoutLike)."""

from typing import Protocol, runtime_checkable

import jax
import jaxtyping as jt


@runtime_checkable
class WoutLike(Protocol):
    """Protocol for objects that can be used as VmecWOut."""

    rmnc: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    zmns: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    xm: jt.Int[jax.Array, " n_fourier_coefficients"]
    xn: jt.Int[jax.Array, " n_fourier_coefficients"]
    gmnc: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    gmns: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    bsupumnc: jt.Float[jax.Array, "n_fourier_coefficients_nyquist n_surfaces"]
    bsupvmnc: jt.Float[jax.Array, "n_fourier_coefficients_nyquist n_surfaces"]
    xm_nyq: jt.Int[jax.Array, " n_fourier_coefficients_nyquist"]
    xn_nyq: jt.Int[jax.Array, " n_fourier_coefficients_nyquist"]
    ns: int
    nfp: int
    lasym: bool
