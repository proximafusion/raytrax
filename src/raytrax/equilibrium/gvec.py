r"""Import GVEC equilibria into raytrax as a :class:`MagneticConfiguration`.

GVEC (Galerkin Variational Equilibrium Code) is an MHD equilibrium solver that
uses a different representation than VMEC: instead of Fourier coefficients it
stores a finite-element/Galerkin representation, exposed to Python via
``pygvec``.

The approach here mirrors the existing VMEC pipeline:

1. Sample positions :math:`\mathbf{x}` and the magnetic field
   :math:`\mathbf{B}` on an intermediate flux-coordinate grid
   :math:`(\rho, \vartheta, \zeta)` using ``state.evaluate()``.
2. Re-use the existing :func:`interpolate_toroidal_to_cylindrical_grid`
   scatter-interpolation to map those samples onto the regular cylindrical
   output grid :math:`(R, \phi, Z)` expected by the ray tracer.
3. Numerically integrate the Jacobian to obtain :math:`dV/d\rho`.

Coordinate conventions
-----------------------
GVEC uses :math:`(R, Z, \phi)` ordering, while raytrax uses :math:`(R, \phi, Z)`.
Additionally, the GVEC toroidal angle :math:`\zeta` runs in the *opposite*
direction to VMEC's :math:`\phi`:

.. math::
    \phi_{\text{VMEC}} = -\zeta_{\text{GVEC}}

This module handles the conversion transparently so callers always receive
arrays in the :math:`(R, \phi, Z)` / :math:`(B_R, B_\phi, B_Z)` convention
used throughout raytrax.

Usage example::

    import gvec
    import raytrax

    state = gvec.find_state("path/to/W7X/")
    mag_config = raytrax.MagneticConfiguration.from_gvec(state)
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from raytrax.equilibrium.interpolate import (
    CylindricalGridResolution,
    MagneticConfiguration,
    interpolate_toroidal_to_cylindrical_grid,
)

if TYPE_CHECKING:
    # ``gvec`` is an optional dependency; only imported for type hints.
    pass


@dataclass
class GvecGridResolution:
    """Grid resolution for GVEC-based equilibrium imports.

    Combines the shared :class:`CylindricalGridResolution` output grid with
    GVEC-specific intermediate flux-coordinate sampling parameters.

    The GVEC pipeline first evaluates the equilibrium on an intermediate
    curvilinear :math:`(\\rho, \\vartheta, \\zeta)` grid, then
    scatter-interpolates the result onto the cylindrical output grid.

    Attributes:
        cylindrical: Output cylindrical grid shared with all other importers.
        n_rho: Number of radial (flux-surface) points on the intermediate grid.
            Includes rho=0 (magnetic axis) and rho=rho_max.
        n_theta: Number of poloidal points on the intermediate grid.
        rho_max: Maximum normalized effective radius to sample. Values slightly
            above 1.0 allow extrapolation just beyond the last closed flux
            surface (LCFS), consistent with the VMEC importer.
    """

    cylindrical: CylindricalGridResolution = dataclass_field(
        default_factory=CylindricalGridResolution
    )
    n_rho: int = 40
    n_theta: int = 45
    rho_max: float = 1.2


def _nfp_from_state(state: Any) -> int:
    """Extract the number of field periods from a gvec.State object.

    GVEC exposes this as ``state.nfp`` or ``state.nfperiod`` depending on the
    version; we try both.
    """
    for attr in ("nfp", "nfperiod", "n_fp", "n_field_periods"):
        if hasattr(state, attr):
            return int(getattr(state, attr))
    raise AttributeError(
        "Cannot determine number of field periods from gvec.State. "
        "Expected attribute 'nfp' or 'nfperiod'. "
        "Please pass nfp explicitly via the nfp parameter."
    )


def _compute_dvolume_drho_gvec(
    state: Any,
    rho_1d: np.ndarray,
    n_theta: int,
    n_zeta: int,
    nfp: int,
) -> np.ndarray:
    r"""Compute :math:`dV/d\rho` from the GVEC Jacobian.

    Uses the identity

    .. math::
        \frac{dV}{d\rho} = \text{nfp}
            \int_0^{2\pi/\text{nfp}} d\zeta
            \int_0^{2\pi} d\vartheta \; \left|\sqrt{g}(\rho, \vartheta, \zeta)\right|

    where :math:`\sqrt{g}` is the Jacobian determinant returned by
    ``state.evaluate("sqrtg", ...)``.  The double integral is approximated
    with the trapezoidal rule on a uniform :math:`(\vartheta, \zeta)` grid.

    Args:
        state: A ``gvec.State`` object.
        rho_1d: 1-D array of normalized effective radii.
        n_theta: Number of poloidal quadrature points.
        n_zeta: Number of toroidal quadrature points (within one field period).
        nfp: Number of field periods.

    Returns:
        Array of shape ``(len(rho_1d),)`` with :math:`dV/d\rho` values in SI
        units (m^3).
    """
    theta_1d = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zeta_1d = np.linspace(0, 2 * np.pi / nfp, n_zeta, endpoint=False)

    # Build meshgrid: shape (n_rho, n_theta, n_zeta)
    rho_g, theta_g, zeta_g = np.meshgrid(rho_1d, theta_1d, zeta_1d, indexing="ij")

    # GVEC uses (rho, vartheta, zeta) — same order as our meshgrid.
    # state.evaluate returns the Jacobian determinant sqrt(g).
    sqrtg = state.evaluate(
        "sqrtg",
        rho_g.ravel(),
        theta_g.ravel(),
        zeta_g.ravel(),
    ).reshape(len(rho_1d), n_theta, n_zeta)

    # Integrate over (vartheta, zeta) with the trapezoidal rule.
    # Full-period prefactor: nfp field periods
    dtheta = 2 * np.pi / n_theta
    dzeta = (2 * np.pi / nfp) / n_zeta
    integral = np.sum(np.abs(sqrtg), axis=(1, 2)) * dtheta * dzeta
    return nfp * integral


def magnetic_configuration_from_gvec(
    state: Any,
    nfp: int | None = None,
    magnetic_field_scale: float = 1.0,
    grid: GvecGridResolution | None = None,
) -> MagneticConfiguration:
    r"""Create a :class:`MagneticConfiguration` from a GVEC equilibrium state.

    Requires the optional ``pygvec`` package to be installed::

        pip install raytrax[gvec]

    Args:
        state: A ``gvec.State`` object, obtained e.g. via ``gvec.find_state``
            or ``gvec.run``.
        nfp: Number of field periods.  If ``None`` (default), the value is
            read from ``state.nfp`` (or ``state.nfperiod`` for older versions).
            Pass this explicitly if your version of GVEC does not expose the
            attribute.
        magnetic_field_scale: Uniform scale factor applied to all magnetic
            field values.  Useful for scaling to a different reactor size
            without re-running the equilibrium.
        grid: Grid resolution settings.  Defaults to
            :class:`GvecGridResolution` with sensible values.

    Returns:
        A :class:`MagneticConfiguration` ready to be passed to
        :func:`raytrax.trace`.

    Notes:
        GVEC uses :math:`(R, Z, \phi)` coordinate ordering with the toroidal
        angle running in the **opposite** direction compared to VMEC.  This
        function applies the sign flip transparently.

        Stellarator symmetry is assumed (``lasym=False``).  Non-symmetric
        equilibria would require sampling the full :math:`[0, 2\pi]` toroidal
        range and are not yet supported.
    """
    try:
        import gvec as _gvec  # noqa: F401  (only needed to raise a helpful error)
    except ImportError as e:
        raise ImportError(
            "The 'pygvec' package is required to import GVEC equilibria. "
            "Install it with:  pip install raytrax[gvec]"
        ) from e

    if grid is None:
        grid = GvecGridResolution()

    if nfp is None:
        nfp = _nfp_from_state(state)

    cyl = grid.cylindrical
    n_rho = grid.n_rho
    n_theta = grid.n_theta
    n_phi = cyl.n_phi
    rho_max = grid.rho_max

    # ------------------------------------------------------------------
    # Step 1: Build the intermediate flux-coordinate sampling grid.
    # Stellarator symmetry: only need half a field period [0, pi/nfp].
    # ------------------------------------------------------------------
    rho_1d = np.linspace(0.0, rho_max, n_rho)
    theta_1d = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    
    # GVEC zeta runs opposite to VMEC phi.  The raytrax output grid
    # expects phi in [0, pi/nfp].
    # By stellarator symmetry, we can sample zeta in [0, -pi/nfp] to get
    # exactly the phi interval [0, pi/nfp] via phi = -zeta.
    phi_max = np.pi / nfp  # half-period (stellarator symmetry)
    zeta_1d = np.linspace(0.0, -phi_max, n_phi)

    rho_g, theta_g, zeta_g = np.meshgrid(rho_1d, theta_1d, zeta_1d, indexing="ij")
    shape = rho_g.shape  # (n_rho, n_theta, n_phi)

    rho_flat = rho_g.ravel()
    theta_flat = theta_g.ravel()
    zeta_flat = zeta_g.ravel()

    # ------------------------------------------------------------------
    # Step 2: Evaluate positions X = (R, Z, phi_gvec) on the grid.
    # GVEC coordinate order: (R, Z, phi) -> we reorder to (R, phi, Z).
    # Toroidal angle: phi_raytrax = -zeta_gvec (sign flip + periodicity).
    # ------------------------------------------------------------------
    # state.evaluate("X1") = R, state.evaluate("X2") = Z
    # state.evaluate("phi") or zeta_flat directly gives the toroidal angle.
    R_flat = np.array(state.evaluate("X1", rho_flat, theta_flat, zeta_flat))
    Z_flat = np.array(state.evaluate("X2", rho_flat, theta_flat, zeta_flat))
    # phi in raytrax convention = -zeta (GVEC angle is inverted vs VMEC)
    phi_flat = -zeta_flat  # sign flip

    R = R_flat.reshape(shape)
    Z = Z_flat.reshape(shape)
    phi = phi_flat.reshape(shape)

    # rphiz_toroidal: shape (n_rho, n_theta, n_phi, 3)
    rphiz_toroidal = jnp.stack(
        [jnp.array(R), jnp.array(phi), jnp.array(Z)], axis=-1
    )

    # ------------------------------------------------------------------
    # Step 3: Evaluate the magnetic field B = (B_R, B_Z, B_phi) in GVEC
    # convention and convert to cylindrical (B_R, B_phi, B_Z).
    # GVEC returns B components in (R, Z, phi) order; we reorder.
    # ------------------------------------------------------------------
    BR_flat = np.array(state.evaluate("B1", rho_flat, theta_flat, zeta_flat))
    BZ_flat = np.array(state.evaluate("B2", rho_flat, theta_flat, zeta_flat))
    Bphi_flat = np.array(state.evaluate("B3", rho_flat, theta_flat, zeta_flat))
    # Sign flip for the toroidal component due to angle inversion
    # B_phi_raytrax = -B_zeta_gvec
    Bphi_raytrax_flat = -Bphi_flat

    BR = jnp.array(BR_flat.reshape(shape))
    Bphi = jnp.array(Bphi_raytrax_flat.reshape(shape))
    BZ = jnp.array(BZ_flat.reshape(shape))

    # Bcyl_toroidal: shape (n_rho, n_theta, n_phi, 3)
    Bcyl_toroidal = jnp.stack([BR, Bphi, BZ], axis=-1) * magnetic_field_scale

    # ------------------------------------------------------------------
    # Step 4: rho on the toroidal grid (just the meshgrid rho values).
    # ------------------------------------------------------------------
    rho_toroidal = jnp.array(rho_g)[..., jnp.newaxis]  # (n_rho, n_theta, n_phi, 1)

    # ------------------------------------------------------------------
    # Step 5: Scatter-interpolate to the cylindrical output grid.
    # Reuse the same function as the VMEC importer.
    # ------------------------------------------------------------------
    rmin = float(jnp.min(rphiz_toroidal[..., 0]))
    rmax = float(jnp.max(rphiz_toroidal[..., 0]))
    zmin = float(jnp.min(rphiz_toroidal[..., 2]))
    zmax = float(jnp.max(rphiz_toroidal[..., 2]))

    rz_cylindrical = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(rmin, rmax, cyl.n_r),
            jnp.linspace(zmin, zmax, cyl.n_z),
            indexing="ij",
        ),
        axis=-1,
    )

    # Concatenate rho and B as values to interpolate together
    value_toroidal = jnp.concatenate([rho_toroidal, Bcyl_toroidal], axis=-1)

    rhoBcyl_cylindrical = interpolate_toroidal_to_cylindrical_grid(
        rphiz_toroidal=rphiz_toroidal,
        rz_cylindrical=rz_cylindrical,
        value_toroidal=value_toroidal,
    )

    # Build the full cylindrical rphiz grid
    rphiz_cylindrical = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(rmin, rmax, cyl.n_r),
            jnp.linspace(0.0, phi_max, n_phi),
            jnp.linspace(zmin, zmax, cyl.n_z),
            indexing="ij",
        ),
        axis=-1,
    )

    rphiz_out = rphiz_cylindrical
    rho_out = rhoBcyl_cylindrical[..., 0]
    B_out = rhoBcyl_cylindrical[..., 1:]

    # ------------------------------------------------------------------
    # Step 6: Compute dV/drho on a 1-D radial grid via Jacobian integral.
    # ------------------------------------------------------------------
    rho_1d_profile = np.linspace(0.0, 1.0, cyl.n_rho_profile)
    dv_drho = _compute_dvolume_drho_gvec(
        state=state,
        rho_1d=rho_1d_profile,
        n_theta=n_theta,
        n_zeta=n_phi,
        nfp=nfp,
    )

    return MagneticConfiguration(
        rphiz=rphiz_out,
        magnetic_field=B_out,
        rho=rho_out,
        nfp=nfp,
        is_stellarator_symmetric=True,
        rho_1d=jnp.array(rho_1d_profile),
        dvolume_drho=jnp.array(dv_drho),
    )
