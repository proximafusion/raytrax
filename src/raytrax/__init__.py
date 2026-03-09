"""Main module for Raytrax."""

import warnings

import jax

if not getattr(jax.config, "jax_enable_x64", False):
    warnings.warn(
        "raytrax requires 64-bit precision. "
        "Call jax.config.update('jax_enable_x64', True) before importing JAX or raytrax.",
        stacklevel=2,
    )

from .api import trace as trace
from .equilibrium.interpolate import MagneticConfiguration as MagneticConfiguration
from .tracer.buffers import Interpolators as Interpolators
from .types import Beam as Beam
from .types import RadialProfiles as RadialProfiles
from .types import TracerSettings as TracerSettings

__all__ = [
    "Beam",
    "Interpolators",
    "MagneticConfiguration",
    "RadialProfiles",
    "TracerSettings",
    "trace",
]
