"""W7-X example equilibrium and beam utilities."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vmecpp
from vmecpp import VmecWOut

from raytrax.equilibrium.interpolate import MagneticConfiguration


@dataclass(frozen=True)
class AntennaPosition:
    """Cylindrical position of a W7-X ECRH antenna.

    Attributes:
        r: Major radius in metres.
        phi_deg: Toroidal angle in degrees.
        z: Vertical position in metres.
    """

    r: float
    phi_deg: float
    z: float

    @property
    def cartesian(self) -> tuple[float, float, float]:
        """Cartesian (x, y, z) position in metres."""
        phi_rad = np.deg2rad(self.phi_deg)
        return (
            float(self.r * np.cos(phi_rad)),
            float(self.r * np.sin(phi_rad)),
            float(self.z),
        )


class PortA:
    """W7-X Port A — vertically stacked ECRH launcher configuration.

    Three gyrotron beams are stacked vertically at the same toroidal position::

        from raytrax.examples.w7x import PortA
        position = jnp.array(PortA.D1.cartesian)
    """

    C1 = AntennaPosition(r=6.58, phi_deg=0.094, z=+0.28)  # top beam
    D1 = AntennaPosition(r=6.595, phi_deg=0.094, z=0.0)  # middle beam
    E1 = AntennaPosition(r=6.58, phi_deg=0.094, z=-0.28)  # bottom beam


_DATA_DIR = Path(__file__).parent.parent / "equilibrium" / "data"


def get_w7x_equilibrium() -> VmecWOut:
    """Get the W7-X equilibrium using vmecpp.

    Runs VMEC++ on the bundled W7-X input file and caches the result as JSON.

    Returns:
        VmecWOut: The W7-X equilibrium
    """
    input_path = _DATA_DIR / "w7x.json"
    wout_json_path = _DATA_DIR / "w7x_wout.json"

    if wout_json_path.exists():
        with open(wout_json_path, "r") as f:
            json_data = f.read()
        return VmecWOut.model_validate_json(json_data)

    print("Creating W7-X equilibrium using vmecpp...")
    vmec_input = vmecpp.VmecInput.from_file(input_path)
    # Scale phiedge so the on-axis field reaches ~2.52 T, placing the
    # 140 GHz 2nd-harmonic EC resonance just inside the magnetic axis.
    # The bundled w7x.json uses phiedge=-1.74 Wb (B_axis≈2.374 T); the
    # correct value is phiedge_orig * (2.52 / 2.374) ≈ -1.847 Wb.
    vmec_input.phiedge = -1.847
    vmec_output = vmecpp.run(vmec_input)
    wout = vmec_output.wout

    json_data = wout.model_dump_json()
    with open(wout_json_path, "w") as f:
        f.write(json_data)

    print(f"Saved W7-X equilibrium to cache: {wout_json_path}")
    return wout


def get_w7x_magnetic_configuration(
    magnetic_field_scale: float = 1.0,
) -> MagneticConfiguration:
    """Get the W7-X magnetic configuration on a cylindrical interpolation grid.

    Loads the bundled W7-X vmecpp equilibrium and builds a
    :class:`~raytrax.MagneticConfiguration` ready to pass to
    :func:`~raytrax.trace`.

    Args:
        magnetic_field_scale: Uniform scale factor applied to the magnetic
            field magnitude (default 1.0).  Use this to match a target on-axis
            field strength, e.g. ``b_scale = B0_target / b0_native``.

    Returns:
        MagneticConfiguration: W7-X magnetic configuration on a cylindrical grid
    """
    wout = get_w7x_equilibrium()
    return MagneticConfiguration.from_vmec_wout(
        wout, magnetic_field_scale=magnetic_field_scale
    )


def w7x_aiming_angles_to_direction(
    theta_pol_deg: float,
    theta_tor_deg: float,
    antenna_phi_deg: float,
) -> tuple[float, float, float]:
    """Convert W7-X aiming angles to a Cartesian unit direction vector.

    The W7-X ECRH system specifies beam direction via a poloidal angle
    (elevation from the horizontal plane) and a toroidal angle (in-plane
    rotation), defined relative to the antenna azimuthal position.

    Args:
        theta_pol_deg: Poloidal aiming angle in degrees (elevation above the
            horizontal plane, positive towards the top of the machine).
        theta_tor_deg: Toroidal aiming angle in degrees (in-plane rotation,
            positive in the direction of increasing phi).
        antenna_phi_deg: Azimuthal angle of the antenna in degrees (the phi
            coordinate of the antenna in cylindrical coordinates).

    Returns:
        Normalised Cartesian direction vector ``(dx, dy, dz)``.
    """
    alpha = np.deg2rad(theta_pol_deg)
    beta = np.deg2rad(theta_tor_deg)
    phi = np.deg2rad(antenna_phi_deg)

    d_r = -np.cos(alpha) * np.cos(beta)
    d_phi = np.cos(alpha) * np.sin(beta)
    d_z = np.sin(alpha)

    d_x = d_r * np.cos(phi) - d_phi * np.sin(phi)
    d_y = d_r * np.sin(phi) + d_phi * np.cos(phi)

    norm = np.sqrt(d_x**2 + d_y**2 + d_z**2)
    return (float(d_x / norm), float(d_y / norm), float(d_z / norm))
