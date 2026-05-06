"""Functional wrapper for TRAVIS ECRH code with dataclass containers."""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jax import Array
from jaxtyping import Float


@dataclass
class TravisECRHInput:
    """Input parameters for TRAVIS ECRH calculation."""

    antenna_position_cyl: Float[Array, "3"]
    """Antenna position (R_m, phi_deg, Z_m) in cylindrical coordinates."""

    target_position: Float[Array, "3"]
    """Target position in coordinates specified by target_coords_type."""

    frequency_ghz: float
    """Beam frequency in GHz."""

    power_mw: float
    """Input power in MW."""

    equilibrium_file: str
    """Path to VMEC wout equilibrium file."""

    target_coords_type: str = "cyl"
    """Target coordinate system: 'cart' (X,Y,Z), 'cyl' (R,phi,Z), or 'W7X-angles' (theta_pol,theta_tor,0)."""

    rho_grid: Float[Array, " n"] | None = None
    """Normalized flux coordinate grid for plasma profiles."""

    electron_density_1e20: Float[Array, " n"] | None = None
    """Electron density in 10^20 m^-3."""

    electron_temperature_keV: Float[Array, " n"] | None = None
    """Electron temperature in keV."""

    max_length_m: float = 30.0
    """Maximum ray path length in meters."""

    max_steps: int = 5000
    """Maximum number of ray tracing steps."""

    dielectric_tracing: str = "cold"
    """Dielectric tensor model for ray tracing: 'cold' or 'weakly_relativistic'."""

    hamiltonian: str = "West"
    """Hamiltonian formulation: 'West' or 'Tokman'."""

    b0_normalization: float | None = None
    """If set, normalize B field to this value [T] on the magnetic axis at phi=0.
    Corresponds to TRAVIS's 'B0_normalization_type at angle on magn.axis'."""

    ne_parm: tuple[float, float, float, float, float] = (0.0, 1.0, 2.0, 0.0, 0.0)
    """TRAVIS Ne-parm parameters (a, p, q, h, w) for analytic profile."""

    te_parm: tuple[float, float, float, float, float] = (0.0, 1.0, 2.0, 0.0, 0.0)
    """TRAVIS Te-parm parameters (a, p, q, h, w) for analytic profile."""


@dataclass
class TravisECRHOutput:
    """Output from TRAVIS ECRH calculation."""

    position_m: Float[Array, "n_points 3"]
    """Ray trajectory positions in Cartesian coordinates (X, Y, Z) in meters."""

    refractive_index: Float[Array, "n_points 3"]
    """Refractive index vector components (Nx, Ny, Nz) along trajectory."""

    arc_length_m: Float[Array, " n_points"]
    """Arc length along ray in meters."""

    optical_depth: Float[Array, " n_points"]
    """Optical depth (tau) along trajectory."""

    rho: Float[Array, " n_points"]
    """Normalized flux coordinate along trajectory."""

    absorption_m_inv: Float[Array, " n_points"]
    """Absorption coefficient in m^-1."""

    linear_power_density_w_per_m: Float[Array, " n_points"]
    """Power deposition per unit length in W/m."""

    electron_density_1e20: Float[Array, " n_points"]
    """Electron density in 10^20 m^-3 along trajectory."""

    electron_temperature_keV: Float[Array, " n_points"]
    """Electron temperature in keV along trajectory."""

    magnetic_field_magnitude_T: Float[Array, " n_points"]
    """Magnetic field magnitude in Tesla along trajectory."""

    rho_profile: Float[Array, " n_rho"]
    """Radial grid for integrated power deposition profile."""

    power_density_w_per_m3: Float[Array, " n_rho"]
    """Power density profile in W/m^3."""

    total_absorbed_power_mw: float
    """Total absorbed power in MW."""

    success: bool
    """Whether TRAVIS execution completed successfully."""


def run_travis(
    travis_executable: Path, params: TravisECRHInput, output_dir: Path | None = None
) -> TravisECRHOutput:
    """Run TRAVIS with given parameters and return results."""

    if output_dir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            return _run_travis_internal(travis_executable, params, output_dir)
    else:
        return _run_travis_internal(travis_executable, params, output_dir)


def _run_travis_internal(
    travis_executable: Path, params: TravisECRHInput, output_dir: Path
) -> TravisECRHOutput:
    """Execute TRAVIS workflow: write inputs, run, parse outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_travis_input_files(params, output_dir)
    _execute_travis(travis_executable, output_dir)
    return _parse_travis_output(output_dir)


def _write_travis_input_files(params: TravisECRHInput, output_dir: Path) -> None:
    """Write TRAVIS input files in the .data format."""
    input_file = output_dir / "travis_input.data"

    if not params.equilibrium_file:
        raise ValueError("Equilibrium file is required")

    # Use equilibrium path as-is - TRAVIS will resolve it
    equilibrium_str = str(params.equilibrium_file)

    # Always use analytic plasma profiles
    ne_central = (
        params.electron_density_1e20[0]
        if params.electron_density_1e20 is not None
        else 1.0
    )
    te_central = (
        params.electron_temperature_keV[0]
        if params.electron_temperature_keV is not None
        else 3.0
    )

    ne_parm_str = " ".join(str(x) for x in params.ne_parm)
    te_parm_str = " ".join(str(x) for x in params.te_parm)

    plasma_section = f"""File_with_plasma_profiles nofile analytic
Central_Ne_[1e20/m^3] {ne_central}
Ne-parm {ne_parm_str}
Central_Te_[keV] {te_central}
Te-parm {te_parm_str}"""

    # Coordinate system mapping
    coord_type_map = {
        "cart": "cartesian",
        "cyl": "cylindrical",
        "W7X-angles": "W7X aiming angles",
        "ITER-angles": "ITER aiming angles",
    }
    target_coords_str = coord_type_map.get(
        params.target_coords_type, params.target_coords_type
    )

    # Write input file using multiline f-string
    input_content = f"""***This_is_input_file_for_TRAVIS_ECRH_code***
TempDirectory ./
Vessel_Configuration nofile
Mirror_Configuration nofile
Magnetic_Configuration {equilibrium_str}
Use_EFIT_file_directly 0
B0_normalization_type {"at angle on magn.axis" if params.b0_normalization is not None else "do not scale"}
B0_normalization_value_and_B_direction {params.b0_normalization if params.b0_normalization is not None else 1.0} 0
Angle_for_B0_[degree] 0
Flux_surface_label toroidal_rho
Use_and_save_mesh 1 0
Stellarator_symmetry 1
Accuracy_[m] 0.001
gridStep_[m] 0.022
gridStep_[degree] 2
Bmn_truncation_level 2e-05
Plasma_size(rmax/a)_and_edge_width(dr/a) 1 0
{plasma_section}
Central_Zeff 1.5
Zeff-parm 0 1 1 0 0
Type_of_distribution_function Maxwell
File_with_distribution_function nofile
Adjoint_approach_model lmfp (with trapped particles)
Adjoint_approach_DKES_data nofile
Adjoint_approach_collision_operator momentum_conservation
#_of_points_in_deposition_profile 100
***Beams_data_below_this_line***
Number_of_beams 1
******Single ray*****
Beam_id 1
Beam_name single_ray
Heating_Scenario O
Frequency_[GHz] {params.frequency_ghz}
Input_power_[MW] {params.power_mw}
Antenna_position {params.antenna_position_cyl[0]} {params.antenna_position_cyl[1]} {params.antenna_position_cyl[2]} 0
antenna_position_in_cartesian_coordinates 0
Target_position {params.target_position[0]} {params.target_position[1]} {params.target_position[2]}
Target_coordinates {target_coords_str}
Beam_radii_[m]_and_astigmatism_axis[deg] 0.02 0.02 0
Beam_focal_lengths_[m]_and_QOemul_flag 10 10 0
Number_of_concentric_circles_about_the_central_ray 0
Number_of_rays_in_each_circle 0
Stop_tracing_if_no_more_power 1
max_path_of_beam_[m] {params.max_length_m}
max_RK_iterations {params.max_steps}
min_RK_stepsize_[wave_length] 1e-05
max_RK_stepsize_[wave_length] 10
"""

    input_content += f"""RK_accuracy 1e-05
Dielectric_tensor_summation_limit_[0_for_auto] 0
Max_power_of_larmor_expansion_and_grid_parms 1 7 700
Dielectric_tensor_model_for_tracing {params.dielectric_tracing}
Hamiltonian_for_tracing {params.hamiltonian}
Number_of_passes_and_reflection_coefficients 1 1 1
"""

    with open(input_file, "w") as f:
        f.write(input_content)


def _write_plasma_profile_file(params: TravisECRHInput, filepath: Path) -> None:
    """Write plasma profiles in TRAVIS format."""
    if (
        params.rho_grid is None
        or params.electron_density_1e20 is None
        or params.electron_temperature_keV is None
    ):
        raise ValueError("Plasma profiles are required")

    profile_lines = [
        f"   {float(rho):18.15f}     {float(ne) * 1e20:18.12e}          {float(te):18.12f}                  1.5"
        for rho, ne, te in zip(
            params.rho_grid,
            params.electron_density_1e20,
            params.electron_temperature_keV,
        )
    ]

    profile_content = f"""CC                r/a               ne,m^3               Te,keV             Zeff
                                                                                numberOfPoints {len(params.rho_grid)}
{chr(10).join(profile_lines)}
"""

    with open(filepath, "w") as f:
        f.write(profile_content)


def _execute_travis(travis_executable: Path, output_dir: Path) -> None:
    """Execute TRAVIS."""
    input_file = output_dir / "travis_input.data"

    result = subprocess.run(
        [str(travis_executable), str(input_file)],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )

    status_file = output_dir / "run.status"
    if status_file.exists():
        with open(status_file, "r") as f:
            status_line = f.readline().strip()
            exit_code = int(status_line.split()[0])
            if exit_code != 0:
                raise RuntimeError(
                    f"TRAVIS execution failed with exit code {exit_code}"
                )
    elif result.returncode != 0:
        raise RuntimeError(f"TRAVIS execution failed: {result.stderr}")


def _parse_travis_output(output_dir: Path) -> TravisECRHOutput:
    """Parse TRAVIS output files."""
    beamtrace_file = output_dir / "beamtrace_1"
    if not beamtrace_file.exists():
        raise RuntimeError("TRAVIS output file beamtrace_1 not found")

    traj_data = _parse_beamtrace(beamtrace_file)

    profile_file = output_dir / "Pabs_Icd_profiles_1"
    if not profile_file.exists():
        raise RuntimeError("TRAVIS output file Pabs_Icd_profiles_1 not found")

    profile_data = _parse_radial_profile(profile_file)

    return TravisECRHOutput(
        position_m=traj_data["position_m"],
        refractive_index=traj_data["refractive_index"],
        arc_length_m=traj_data["arc_length_m"],
        optical_depth=traj_data["optical_depth"],
        rho=traj_data["rho"],
        absorption_m_inv=traj_data["absorption_m_inv"],
        linear_power_density_w_per_m=traj_data["linear_power_density_w_per_m"],
        electron_density_1e20=traj_data["electron_density_1e20"],
        electron_temperature_keV=traj_data["electron_temperature_keV"],
        magnetic_field_magnitude_T=traj_data["magnetic_field_magnitude_T"],
        rho_profile=profile_data["rho"],
        power_density_w_per_m3=profile_data["power_density_w_per_m3"],
        total_absorbed_power_mw=profile_data["total_absorbed_power_mw"],
        success=True,
    )


def _parse_beamtrace(filepath: Path) -> dict:
    """Parse TRAVIS beamtrace output file.

    Columns: Nray, path, X, Y, Z, Nx, Ny, Nz, rho, ne, Te, |B|,
             Nper, Npar, Nperc, Nparc, damp0, damp, tau0, tau, ...
    """
    data = []
    with open(filepath, "r") as f:
        f.readline()

        for line in f:
            values = line.split()
            if len(values) > 50:
                data.append([float(v) for v in values])

    data = np.array(data)

    return {
        "position_m": data[:, 2:5],
        "refractive_index": data[:, 5:8],
        "arc_length_m": data[:, 1],
        "rho": data[:, 8],
        "optical_depth": data[:, 18],
        "absorption_m_inv": data[:, 16],
        "linear_power_density_w_per_m": (data[:, 23] + data[:, 24]) * 1e6,  # MW/m → W/m
        "electron_density_1e20": data[:, 9] / 1e20,  # Column 10: ne in m^-3
        "electron_temperature_keV": data[:, 10],  # Column 11: Te in keV
        "magnetic_field_magnitude_T": data[
            :, 11
        ],  # Column 12: |B| (needs denormalization)
    }


def _parse_radial_profile(filepath: Path) -> dict:
    """Parse TRAVIS radial profile output file.

    Columns: reff/a, dP_p/dV, dP_t/dV, P_p, P_t, dP/dV, Pabs, ...
    """
    data = []
    with open(filepath, "r") as f:
        f.readline()
        f.readline()

        for line in f:
            values = line.split()
            if len(values) >= 6:
                data.append([float(v) for v in values])

    data = np.array(data)

    return {
        "rho": data[:, 0],
        "power_density_w_per_m3": data[:, 5] * 1e6,
        "total_absorbed_power_mw": data[-1, 6] if len(data) > 0 else 0.0,
    }


def save_reference_data(output: TravisECRHOutput, filepath: Path) -> None:
    """Save TRAVIS output to JSON for use as reference data."""
    data = {
        "position_m": np.asarray(output.position_m).tolist(),
        "refractive_index": np.asarray(output.refractive_index).tolist(),
        "arc_length_m": np.asarray(output.arc_length_m).tolist(),
        "optical_depth": np.asarray(output.optical_depth).tolist(),
        "rho": np.asarray(output.rho).tolist(),
        "absorption_m_inv": np.asarray(output.absorption_m_inv).tolist(),
        "linear_power_density_w_per_m": np.asarray(
            output.linear_power_density_w_per_m
        ).tolist(),
        "electron_density_1e20": np.asarray(output.electron_density_1e20).tolist(),
        "electron_temperature_keV": np.asarray(
            output.electron_temperature_keV
        ).tolist(),
        "magnetic_field_magnitude_T": np.asarray(
            output.magnetic_field_magnitude_T
        ).tolist(),
        "rho_profile": np.asarray(output.rho_profile).tolist(),
        "power_density_w_per_m3": np.asarray(output.power_density_w_per_m3).tolist(),
        "total_absorbed_power_mw": output.total_absorbed_power_mw,
        "success": output.success,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_reference_data(filepath: Path) -> TravisECRHOutput:
    """Load TRAVIS reference data from JSON."""
    import jax.numpy as jnp

    with open(filepath, "r") as f:
        data = json.load(f)

    return TravisECRHOutput(
        position_m=jnp.array(data["position_m"]),
        refractive_index=jnp.array(data["refractive_index"]),
        arc_length_m=jnp.array(data["arc_length_m"]),
        optical_depth=jnp.array(data["optical_depth"]),
        rho=jnp.array(data["rho"]),
        absorption_m_inv=jnp.array(data["absorption_m_inv"]),
        linear_power_density_w_per_m=jnp.array(data["linear_power_density_w_per_m"]),
        electron_density_1e20=jnp.array(data["electron_density_1e20"]),
        electron_temperature_keV=jnp.array(data["electron_temperature_keV"]),
        magnetic_field_magnitude_T=jnp.array(data["magnetic_field_magnitude_T"]),
        rho_profile=jnp.array(data["rho_profile"]),
        power_density_w_per_m3=jnp.array(data["power_density_w_per_m3"]),
        total_absorbed_power_mw=data["total_absorbed_power_mw"],
        success=data["success"],
    )
