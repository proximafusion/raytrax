"""1D plotting functions for radial profiles."""

import matplotlib.pyplot as plt
import numpy as np

from raytrax.types import BeamProfile, RadialProfiles, RadialProfile


def plot_radial_electron_density(radial_profiles: RadialProfiles, ax=None):
    """Plot the electron density as function of rho.

    Args:
        radial_profiles: The input RadialProfiles.
        ax: Optional matplotlib axes. If None, creates new figure with two subplots.

    Returns:
        Matplotlib axes (single axis or tuple of two axes).
    """
    rho = np.array(radial_profiles.rho)
    ne = np.array(radial_profiles.electron_density)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(rho, ne)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Electron Density [10$^{20} m$^{-3}$]")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    return ax


def plot_radial_electron_temperature(radial_profiles: RadialProfiles, ax=None):
    """Plot the electron temperature as function of rho.

    Args:
        radial_profiles: The input RadialProfiles.
        ax: Optional matplotlib axes. If None, creates new figure.

    Returns:
        Matplotlib axis.
    """
    rho = np.array(radial_profiles.rho)
    te = np.array(radial_profiles.electron_temperature)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(rho, te)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Electron Temperature [keV]")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    return ax


def plot_linear_power_density(beam_profile: BeamProfile, ax=None):
    """Plot the linear power density as function of arc length.

    Args:
        beam_profile: The traced BeamProfile.
        ax: Optional matplotlib axes. If None, creates new figure.

    Returns:
        Matplotlib axis.
    """
    s = np.array(beam_profile.arc_length)
    p = np.array(beam_profile.linear_power_density)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(s, p / 1e6)  # Convert to MW/m
    ax.set_xlabel("Arc length [m]")
    ax.set_ylabel("Linear Power Density [MW/m]")
    ax.grid(True, alpha=0.3)

    return ax


def plot_radial_power_density(radial_profile: RadialProfile, ax=None):
    """Plot the radial power deposition profile.

    Args:
        radial_profile: The output RadialProfile with volumetric power density.
        ax: Optional matplotlib axis. If None, creates new figure.

    Returns:
        Matplotlib axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    rho = np.array(radial_profile.rho)
    power = np.array(radial_profile.volumetric_power_density)

    ax.plot(rho, power / 1e6)  # Convert to MW/m³
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Volumetric Power Density [MW/m$^3$]")
    ax.set_xlim(0, 1)

    return ax
