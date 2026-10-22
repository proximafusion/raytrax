r"""Import of GVEC equilibria onto the cylindrical grid.

GVEC describes the geometry by a map $(X^1, X^2, \zeta) \mapsto \boldsymbol{x}$
that is chosen per equilibrium (the "hmap"), so a cross-section
$\zeta = \text{const}$ is in general not a plane $\phi = \text{const}$. The
import therefore samples every $(\rho, \vartheta)$ line of the GVEC grid along
$\zeta$, inverts $\phi(\zeta)$ on each line to cut out the output planes
$\phi_k$, and then scatter-interpolates within each plane exactly like the
VMEC pipeline. The result has the same layout as for VMEC, so the solver
handles it unchanged.

Supported are equilibria that

- are periodic under a rotation by $2\pi/n_\text{fp}$ about the $Z$-axis,
- are stellarator symmetric under $(x, y, z) \mapsto (x, -y, -z)$,
- have a strictly monotonic $\phi(\zeta)$ on every $(\rho, \vartheta)$ line,
  i.e. every half-plane $\phi = \text{const}$ cuts each flux surface in a
  single closed curve. This excludes figure-8 and knotted geometries.

All three properties are checked numerically and violations raise
``NotImplementedError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np

from raytrax.equilibrium.interpolate import (
    CylindricalGridResolution,
    cartesian_to_cylindrical_vector,
    cylindrical_grid_from_toroidal_samples,
)

if TYPE_CHECKING:
    import gvec

# The Jacobian vanishes on the magnetic axis, where B is only defined as a limit,
# so the innermost surface is sampled slightly off-axis.
_RHO_AXIS = 1e-5
# Surfaces beyond the LCFS, constructed by linear extension from the axis
# (GVEC cannot be evaluated at rho > 1).
_RHO_EXTENSION = (1.1, 1.2)
_SYMMETRY_RTOL = 1e-6


@dataclass
class GvecGridResolution:
    """Grid resolution for GVEC-based equilibrium imports.

    Attributes:
        cylindrical: Output cylindrical grid shared with all other importers.
        n_rho: Radial sample points on the GVEC grid.
        n_theta: Poloidal sample points on the GVEC grid.
        n_zeta: Toroidal sample points per field period used to invert
            $\\phi(\\zeta)$ on each $(\\rho, \\vartheta)$ line.
    """

    cylindrical: CylindricalGridResolution = dataclass_field(
        default_factory=CylindricalGridResolution
    )
    n_rho: int = 40
    n_theta: int = 45
    n_zeta: int = 120


def _evaluate(
    state: gvec.State,
    quantity: str,
    rho: Sequence[float] | np.ndarray,
    theta: Sequence[float] | np.ndarray,
    zeta: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Evaluate a GVEC quantity on a tensor grid, as array (rho, theta, zeta[, xyz])."""
    dataset = state.evaluate(
        quantity,
        rho=np.asarray(rho, dtype=float),
        theta=np.asarray(theta, dtype=float) % (2 * np.pi),
        zeta=np.asarray(zeta, dtype=float) % (2 * np.pi),
    )
    dims: tuple[str, ...] = ("rad", "pol", "tor")
    data = dataset[quantity]
    if "xyz" in data.dims:
        dims = (*dims, "xyz")
    return np.asarray(data.transpose(*dims).values)


def _toroidal_sense(state: gvec.State) -> int:
    """Return the direction of the net rotation in $\\phi$ as $\\zeta$ advances.

    GVEC's $\\zeta$ commonly runs along $-\\phi$.
    """
    zeta = np.linspace(0, 2 * np.pi / state.nfp, 65)
    pos = _evaluate(state, "pos", rho=[0.5], theta=[0.0], zeta=zeta)[0, 0]
    phi = np.unwrap(np.arctan2(pos[:, 1], pos[:, 0]))
    return 1 if phi[-1] > phi[0] else -1


def _check_field_periodicity(state: gvec.State, sense: int) -> None:
    """Raise unless the geometry is periodic under a rotation by 2*pi/nfp about Z."""
    period = 2 * np.pi / state.nfp
    rho = [0.3, 1.0]
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    zeta = np.linspace(0, period, 7, endpoint=False)
    pos = _evaluate(state, "pos", rho, theta, zeta)
    pos_next = _evaluate(state, "pos", rho, theta, zeta + period)
    angle = sense * period
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    deviation = np.max(np.linalg.norm(pos_next - pos @ rotation.T, axis=-1))
    if deviation > _SYMMETRY_RTOL * np.max(np.abs(pos)):
        raise NotImplementedError(
            "GVEC import requires periodicity under a rotation by 2*pi/nfp about "
            f"the Z-axis; the geometry deviates by {deviation:.3e} m."
        )


def _check_stellarator_symmetry(state: gvec.State) -> None:
    """Raise unless the equilibrium is symmetric under (x, y, z) -> (x, -y, -z)."""
    rho = [0.3, 1.0]
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / state.nfp, 7, endpoint=False)
    mirror = np.diag([1.0, -1.0, -1.0])
    pos = _evaluate(state, "pos", rho, theta, zeta)
    pos_mirrored = _evaluate(state, "pos", rho, -theta, -zeta)
    B = _evaluate(state, "B", rho, theta, zeta)
    B_mirrored = _evaluate(state, "B", rho, -theta, -zeta)
    pos_deviation = np.max(np.linalg.norm(pos_mirrored - pos @ mirror, axis=-1))
    B_deviation = np.max(np.linalg.norm(B_mirrored + B @ mirror, axis=-1))
    if pos_deviation > _SYMMETRY_RTOL * np.max(
        np.abs(pos)
    ) or B_deviation > _SYMMETRY_RTOL * np.max(np.abs(B)):
        raise NotImplementedError(
            "GVEC import requires stellarator symmetry under (x, y, z) -> "
            f"(x, -y, -z); position deviates by {pos_deviation:.3e} m and B by "
            f"{B_deviation:.3e} T."
        )


def _cut_constant_phi_planes(
    phi_lines: np.ndarray, values: np.ndarray, phi_planes: np.ndarray
) -> np.ndarray:
    """Linearly interpolate values along each line to where it crosses each plane.

    Args:
        phi_lines: Strictly increasing $\\phi$ along the last axis, shape (..., n_s).
        values: Values on the lines, shape (..., n_s, n_values).
        phi_planes: Plane angles, shape (n_planes,).

    Returns:
        Values at the crossings, shape (..., n_planes, n_values).
    """
    n_s = phi_lines.shape[-1]
    index = np.sum(phi_lines[..., None, :] < phi_planes[:, None], axis=-1) - 1
    index = np.clip(index, 0, n_s - 2)
    phi_lo = np.take_along_axis(phi_lines, index, axis=-1)
    phi_hi = np.take_along_axis(phi_lines, index + 1, axis=-1)
    weight = ((phi_planes - phi_lo) / (phi_hi - phi_lo))[..., None]
    lo = np.take_along_axis(values, index[..., None], axis=-2)
    hi = np.take_along_axis(values, index[..., None] + 1, axis=-2)
    return (1 - weight) * lo + weight * hi


def cylindrical_grid_for_gvec_equilibrium(
    state: gvec.State,
    n_rho: int,
    n_theta: int,
    n_zeta: int,
    n_r: int,
    n_z: int,
    n_phi: int,
) -> jt.Float[jax.Array, "n_r n_phi n_z rphizrhoBcyl=7"]:
    """Interpolate a GVEC equilibrium onto the cylindrical grid.

    Args:
        state: GVEC State, see the module docstring for the supported class.
        n_rho: Radial sample points on the GVEC grid.
        n_theta: Poloidal sample points on the GVEC grid.
        n_zeta: Toroidal sample points per field period.
        n_r: Output grid points along R.
        n_z: Output grid points along Z.
        n_phi: Output planes in the fundamental domain phi in [0, pi/nfp].

    Returns:
        Array of shape (n_r, n_phi, n_z, 7) holding (R, phi, Z, rho, B_R,
        B_phi, B_Z), the same layout as
        :func:`~raytrax.equilibrium.interpolate.cylindrical_grid_for_equilibrium`.
        Points outside the sampled region are NaN.

    Raises:
        NotImplementedError: If the equilibrium is not in the supported class.
    """
    sense = _toroidal_sense(state)
    _check_field_periodicity(state, sense)
    _check_stellarator_symmetry(state)

    half_period = np.pi / state.nfp
    rho = np.linspace(0.0, 1.0, n_rho)
    rho_sampled = rho.copy()
    rho_sampled[0] = _RHO_AXIS
    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    # Half a field period of margin on both sides of the fundamental domain
    # covers cross-sections that are tilted against the phi planes.
    zeta = sense * np.linspace(
        -half_period, 2 * half_period, int(1.5 * n_zeta) + 1, endpoint=True
    )
    i_zeta0 = int(np.argmin(np.abs(zeta)))

    pos = _evaluate(state, "pos", rho_sampled, theta, zeta)
    B = _evaluate(state, "B", rho_sampled, theta, zeta)

    phi_lines = np.unwrap(np.arctan2(pos[..., 1], pos[..., 0]), axis=-1)
    phi_lines -= (
        2 * np.pi * np.round(phi_lines[..., i_zeta0 : i_zeta0 + 1] / (2 * np.pi))
    )
    if np.any(np.diff(phi_lines, axis=-1) <= 0):
        raise NotImplementedError(
            "GVEC import requires phi to increase monotonically along zeta on "
            "every (rho, theta) line, i.e. each half-plane phi = const must cut "
            "each flux surface in a single closed curve (excludes e.g. figure-8 "
            "geometries)."
        )
    if np.any(phi_lines[..., 0] > 0) or np.any(phi_lines[..., -1] < half_period):
        raise NotImplementedError(
            "GVEC import supports cross-sections tilted by less than half a "
            "field period against the phi = const planes."
        )

    # Interpolate cylindrical components, which vary only with the shaping,
    # rather than Cartesian ones, which also rotate with phi along the line.
    B_cyl_lines = np.asarray(
        cartesian_to_cylindrical_vector(jnp.asarray(B), jnp.asarray(phi_lines))
    )
    RZ_lines = np.stack([np.hypot(pos[..., 0], pos[..., 1]), pos[..., 2]], axis=-1)
    phi_planes = np.linspace(0.0, half_period, n_phi)
    cut = _cut_constant_phi_planes(
        phi_lines, np.concatenate([RZ_lines, B_cyl_lines], axis=-1), phi_planes
    )  # (n_rho, n_theta, n_phi, 5)
    R = cut[..., 0]
    Z = cut[..., 1]
    B_cyl = cut[..., 2:]

    # Extend beyond the LCFS along straight lines from the axis, holding B at
    # its LCFS value.
    R_ext = [R[0] + r * (R[-1] - R[0]) for r in _RHO_EXTENSION]
    Z_ext = [Z[0] + r * (Z[-1] - Z[0]) for r in _RHO_EXTENSION]
    R = np.concatenate([R, np.stack(R_ext)])
    Z = np.concatenate([Z, np.stack(Z_ext)])
    phi = np.broadcast_to(phi_planes, R.shape)
    B_cyl = np.concatenate([B_cyl, np.stack([B_cyl[-1]] * len(_RHO_EXTENSION))])
    rho_all = np.concatenate([rho, _RHO_EXTENSION])
    rho_grid = np.broadcast_to(rho_all[:, None, None, None], (*R.shape, 1))

    return cylindrical_grid_from_toroidal_samples(
        rphiz_toroidal=jnp.asarray(np.stack([R, phi, Z], axis=-1)),
        value_toroidal=jnp.asarray(np.concatenate([rho_grid, B_cyl], axis=-1)),
        n_r=n_r,
        n_z=n_z,
    )


def dvolume_drho_gvec(
    state: gvec.State,
    rho_1d: np.ndarray,
    n_theta: int = 60,
    n_zeta: int = 60,
) -> jt.Float[jax.Array, " n_rho"]:
    """Compute $dV/d\\rho$ by integrating the GVEC Jacobian over $(\\vartheta, \\zeta)$.

    $dV/d\\rho = n_\\text{fp} \\int_0^{2\\pi/n_\\text{fp}} \\int_0^{2\\pi}
    J(\\rho, \\vartheta, \\zeta)\\, d\\vartheta\\, d\\zeta$, evaluated with the
    rectangle rule, which is spectrally accurate for these periodic integrands.

    Args:
        state: GVEC State.
        rho_1d: Radial points in [0, 1].
        n_theta: Poloidal quadrature points.
        n_zeta: Toroidal quadrature points per field period.

    Returns:
        $dV/d\\rho$ at each radial point.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / state.nfp, n_zeta, endpoint=False)
    jacobian = _evaluate(state, "Jac", np.asarray(rho_1d), theta, zeta)
    return jnp.asarray(4 * np.pi**2 * np.mean(jacobian, axis=(1, 2)))
