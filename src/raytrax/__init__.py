"""Main module for Raytrax."""

import jax

jax.config.update("jax_enable_x64", True)

from .api import trace
from .types import Beam, Interpolators, MagneticConfiguration, RadialProfiles
