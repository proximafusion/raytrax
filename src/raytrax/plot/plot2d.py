"""Visualization functions for Raytrax."""

from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator

from raytrax.equilibrium.interpolate import MagneticConfiguration
from raytrax.types import BeamProfile, RadialProfiles

RZSlice = namedtuple("RZSlice", ["R", "Z", "B", "rho"])


def interpolate_rz_slice(
    magnetic_configuration: MagneticConfiguration,
    phi: float,
    n_r: int = 300,
    n_z: int = 300,
) -> RZSlice:
    """Interpolate magnetic field magnitude and rho on an R-Z plane at specified phi.

    Args:
        magnetic_configuration: The magnetic configuration object.
        phi: The toroidal angle at which to evaluate.
        n_r: Number of R grid points for interpolation.
        n_z: Number of Z grid points for interpolation.

    Returns:
        RZSlice namedtuple with fields (R, Z, B, rho) where R and Z are 2D meshgrid
        arrays, B is the interpolated magnetic field magnitude, and rho is the
        normalized effective radius.
    """
    rphiz = magnetic_configuration.rphiz
    B_mag = np.linalg.norm(np.array(magnetic_configuration.magnetic_field), axis=-1)
    rho_3d = np.array(magnetic_configuration.rho)

    # Extract 1D coordinate arrays
    r_vals = np.array(rphiz[:, 0, 0, 0])
    phi_vals = np.array(rphiz[0, :, 0, 1])
    z_vals = np.array(rphiz[0, 0, :, 2])

    # Create interpolators
    interp_B = RegularGridInterpolator((r_vals, phi_vals, z_vals), B_mag)
    interp_rho = RegularGridInterpolator((r_vals, phi_vals, z_vals), rho_3d)

    # Evaluate on a fine grid
    r_fine = np.linspace(r_vals.min(), r_vals.max(), n_r)
    z_fine = np.linspace(z_vals.min(), z_vals.max(), n_z)
    R, Z = np.meshgrid(r_fine, z_fine, indexing="ij")
    points = np.stack([R.ravel(), np.full(R.size, phi), Z.ravel()], axis=-1)
    B_slice = interp_B(points).reshape(R.shape)
    rho_slice = interp_rho(points).reshape(R.shape)

    # Mask B values where rho > 1 (outside last closed flux surface)
    B_slice = np.where(rho_slice <= 1.0, B_slice, np.nan)

    return RZSlice(R=R, Z=Z, B=B_slice, rho=rho_slice)


def plot_magnetic_field_rz(
    magnetic_configuration: MagneticConfiguration, phi: float, ax=None, **kwargs
):
    """Plot the magnetic field strength in the R-Z plane."""

    if ax is None:
        _, ax = plt.subplots()

    slice_data = interpolate_rz_slice(magnetic_configuration, phi)
    defaults = {
        "levels": 5,
        "cmap": "viridis",
    }

    CS = ax.contour(slice_data.R, slice_data.Z, slice_data.B, **(defaults | kwargs))
    ax.clabel(CS, inline=True, fontsize=8)

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")

    return ax


def plot_electron_density_rz(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
    phi: float,
    ax=None,
    **kwargs,
):
    """Plot the electron density in the R-Z plane.

    Args:
        magnetic_configuration: The magnetic configuration object.
        radial_profiles: Radial profiles containing electron density vs rho.
        phi: The toroidal angle at which to evaluate.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        **kwargs: Additional arguments passed to contourf (e.g., levels, cmap).

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Get the R-Z slice data
    slice_data = interpolate_rz_slice(magnetic_configuration, phi)

    # Interpolate electron density profile onto the R-Z grid
    # electron_density is in units of 10^20 m^-3
    # Use right=0.0 so density smoothly goes to zero at rho=1
    ne_interp = np.interp(
        slice_data.rho.ravel(),
        np.array(radial_profiles.rho),
        np.array(radial_profiles.electron_density),
        left=0.0,
        right=0.0,
    ).reshape(slice_data.rho.shape)

    ne_interp = np.where(slice_data.rho <= 1, ne_interp, np.nan)

    defaults = {
        "levels": 20,
        "cmap": "plasma",
    }

    cont = ax.contourf(slice_data.R, slice_data.Z, ne_interp, **(defaults | kwargs))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cont, cax=cax, label="$n_e$ [$10^{20}$ m$^{-3}$]")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")

    return ax


def plot_effective_radius_rz(
    magnetic_configuration: MagneticConfiguration, phi: float, ax=None, **kwargs
):
    """Plot the normalized effective radius rho in the R-Z plane."""

    if ax is None:
        _, ax = plt.subplots()

    slice_data = interpolate_rz_slice(magnetic_configuration, phi)
    defaults = {
        "levels": np.arange(0.1, 1.1, 0.1),
        "colors": "0.7",
        "linewidths": 1,
    }

    ax.contour(slice_data.R, slice_data.Z, slice_data.rho, **(defaults | kwargs))
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")

    return ax


def plot_beamtrace_rz(
    beam_trace: BeamProfile,
    phi: float,
    ax=None,
    add_colorbar: bool = True,
    **kwargs,
):
    """Plot the beam trace in the R-Z plane, coloured by linear power density.

    Args:
        beam_trace: The traced BeamProfile.
        phi: Toroidal angle (unused; kept for API consistency).
        ax: Matplotlib axes. If None, a new figure is created.
        add_colorbar: If True (default), attach a matched-height colorbar.
        **kwargs: Passed to LineCollection (e.g. ``lw``, ``label``).
            ``color`` is ignored since colouring is driven by the power density.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    Z = np.array(beam_trace.position[:, 2])
    x = np.array(beam_trace.position[:, 0])
    y = np.array(beam_trace.position[:, 1])
    R = np.sqrt(x**2 + y**2)
    p = np.array(beam_trace.linear_power_density) / 1e6  # MW/m

    # Build per-segment colour values (midpoint average of adjacent points)
    points = np.stack([R, Z], axis=1).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    p_seg = 0.5 * (p[:-1] + p[1:])

    kwargs.pop("color", None)  # colour is set by cmap
    lc = LineCollection(segments, cmap="plasma", **kwargs)  # type: ignore[arg-type]
    lc.set_array(p_seg)
    ax.add_collection(lc)
    ax.autoscale()

    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(lc, cax=cax, label="Linear power density [MW/m]")

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")

    return ax
