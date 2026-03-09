"""Linear-algebra helpers: Hermitian and anti-Hermitian matrix decomposition."""

import jax


def anti_hermitian_part(matrix: jax.Array) -> jax.Array:
    """Returns the anti-Hermitian part of a matrix."""
    return 0.5 * (matrix - matrix.conj().T)


def hermitian_part(matrix: jax.Array) -> jax.Array:
    """Returns the Hermitian part of a matrix."""
    return 0.5 * (matrix + matrix.conj().T)
