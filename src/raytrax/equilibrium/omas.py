r"""Import IMAS/OMAS axisymmetric tokamak equilibria into raytrax.

OMAS (Ordered Multidimensional Array Structures) provides an API to read/write
ITER IMAS data dictionaries. This module extracts 2D axisymmetric equilibria
from an ``omas.ODS`` object and converts them into a :class:`MagneticConfiguration`
that can be used by the ray tracer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from raytrax.equilibrium.interpolate import MagneticConfiguration

if TYPE_CHECKING:
    pass


def magnetic_configuration_from_omas(
    ods: Any,
    time_index: int = 0,
    grid_index: int = 0,
    magnetic_field_scale: float = 1.0,
) -> MagneticConfiguration:
    r"""Create a :class:`MagneticConfiguration` from an OMAS data structure (ODS).

    Requires the optional ``omas`` package to be installed::

        pip install omas

    Args:
        ods: An ``omas.ODS`` object containing the ``equilibrium`` data structure.
        time_index: Time slice index to extract (default: 0).
        grid_index: Index of the 2D profile grid to use (default: 0). The grid
            must be a regular rectangular $(R, Z)$ mesh.
        magnetic_field_scale: Uniform scale factor applied to the magnetic field.

    Returns:
        A :class:`MagneticConfiguration` ready to be passed to
        :func:`raytrax.trace`.

    Raises:
        ValueError: If the necessary fields are missing or the grid is not
            a regular 2D rectangular grid.
    """
    import scipy.interpolate

    eq = ods["equilibrium"]
    time_slice = eq["time_slice"][time_index]

    prof2d = time_slice["profiles_2d"][grid_index]
    prof1d = time_slice["profiles_1d"]

    # 1. Extract the 1D R and Z grid coordinates.
    # IMAS allows grid.dim1 and grid.dim2 for structured rectangular grids.
    if "grid" in prof2d and "dim1" in prof2d["grid"] and "dim2" in prof2d["grid"]:
        R_1d = np.array(prof2d["grid"]["dim1"])
        Z_1d = np.array(prof2d["grid"]["dim2"])
    elif "r" in prof2d and "z" in prof2d:
        # Fallback to r/z fields
        R_grid = np.array(prof2d["r"])
        Z_grid = np.array(prof2d["z"])
        if R_grid.ndim == 2:
            R_1d = R_grid[:, 0]
            Z_1d = Z_grid[0, :]
        else:
            R_1d = R_grid
            Z_1d = Z_grid
    else:
        raise ValueError("Could not find rectangular R/Z grid in OMAS profiles_2d.")

    n_r = len(R_1d)
    n_z = len(Z_1d)

    # 2. Extract magnetic field components.
    # New IMAS standard uses b_field_r, b_field_tor, b_field_z.
    # Older OMAS conventions might use b_r, b_tor, b_z.
    def get_b(name_new: str, name_old: str) -> np.ndarray:
        if name_new in prof2d:
            return np.array(prof2d[name_new])
        elif name_old in prof2d:
            return np.array(prof2d[name_old])
        raise ValueError(
            f"Magnetic field component {name_new} not found in profiles_2d."
        )

    BR_2d = get_b("b_field_r", "b_r")
    Bphi_2d = get_b("b_field_tor", "b_tor")
    BZ_2d = get_b("b_field_z", "b_z")

    if BR_2d.shape != (n_r, n_z):
        # IMAS often stores as (dim1, dim2) == (R, Z), but if transposed:
        if BR_2d.shape == (n_z, n_r):
            BR_2d = BR_2d.T
            Bphi_2d = Bphi_2d.T
            BZ_2d = BZ_2d.T
        else:
            raise ValueError(
                f"Grid shape mismatch: R={n_r}, Z={n_z}, but B field has shape {BR_2d.shape}"
            )

    # 3. Construct 2D rho field mapping
    # Poloidal flux psi on the 2D grid
    psi_2d = np.array(prof2d["psi"])
    if psi_2d.shape == (n_z, n_r):
        psi_2d = psi_2d.T

    # 1D profiles map psi to rho (normalized toroidal flux radius)
    psi_1d = np.array(prof1d["psi"])
    rho_1d_prof = np.array(prof1d["rho_tor_norm"])

    # Strictly increasing psi is needed for interpolation.
    # Sometimes psi goes from center (negative/min) to edge (zero/max), sometimes opposite.
    # Interp1d requires monotonically increasing x.
    sort_idx = np.argsort(psi_1d)
    psi_1d_sorted = psi_1d[sort_idx]
    rho_1d_sorted = rho_1d_prof[sort_idx]

    # Map psi on the 2D grid to rho. Values outside LCFS (if psi goes beyond psi_boundary)
    # can be linearly extrapolated.
    rho_2d = scipy.interpolate.interp1d(
        psi_1d_sorted,
        rho_1d_sorted,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(psi_2d)

    # 4. Construct 1D dV/drho profile
    # IMAS provides volume enclosed by the flux surface.
    vol_1d = np.array(prof1d["volume"])[sort_idx]
    # dV/drho using central differences
    dv_drho = np.gradient(vol_1d, rho_1d_sorted)

    # Create an evenly spaced 1D grid for interpolation, as expected by raytrax
    n_rho_profile = len(rho_1d_prof)
    rho_1d_uniform = np.linspace(0.0, 1.0, n_rho_profile)
    dv_drho_uniform = scipy.interpolate.interp1d(
        rho_1d_sorted,
        dv_drho,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )(rho_1d_uniform)

    # 5. Build full 4D arrays for MagneticConfiguration
    # axis 1 is the toroidal angle phi (n_phi=1 for axisymmetric)
    R_mesh, Z_mesh = np.meshgrid(R_1d, Z_1d, indexing="ij")

    # Shape: (n_r, 1, n_z)
    R_3d = R_mesh[:, np.newaxis, :]
    Z_3d = Z_mesh[:, np.newaxis, :]
    phi_3d = np.zeros_like(R_3d)

    # rphiz: (n_r, 1, n_z, 3)
    rphiz = jnp.stack([R_3d, phi_3d, Z_3d], axis=-1)

    # B field: (n_r, 1, n_z, 3)
    BR_3d = BR_2d[:, np.newaxis, :] * magnetic_field_scale
    Bphi_3d = Bphi_2d[:, np.newaxis, :] * magnetic_field_scale
    BZ_3d = BZ_2d[:, np.newaxis, :] * magnetic_field_scale
    B_cyl = jnp.stack([BR_3d, Bphi_3d, BZ_3d], axis=-1)

    # rho: (n_r, 1, n_z)
    rho_3d = jnp.array(rho_2d[:, np.newaxis, :])

    return MagneticConfiguration(
        rphiz=jnp.array(rphiz),
        magnetic_field=jnp.array(B_cyl),
        rho=rho_3d,
        nfp=1,
        is_stellarator_symmetric=True,
        rho_1d=jnp.array(rho_1d_uniform),
        dvolume_drho=jnp.array(dv_drho_uniform),
        is_axisymmetric=True,
    )
