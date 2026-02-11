"""Main module for Raytrax."""

import jax

jax.config.update("jax_enable_x64", True)

from .api import trace, get_interpolator_for_equilibrium
from .types import Beam, MagneticConfiguration, RadialProfiles
