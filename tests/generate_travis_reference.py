"""Script to generate TRAVIS reference data for integration testing."""

import argparse
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from travis_wrapper import TravisECRHInput, run_travis, save_reference_data


def create_w7x_test_case(equilibrium_file: str) -> TravisECRHInput:
    """Create W7-X test case matching raytrax test parameters."""
    
    antenna_position_cyl = jnp.array([6.509, -6.564, 0.38])
    target_position = jnp.array([15.7, 19.7, 0.0])
    frequency_ghz = 140.0
    
    n_rho = 51
    rho_grid = jnp.linspace(0, 1, n_rho)
    ne0_1e20 = 1.0
    electron_density_1e20 = ne0_1e20 * (1 - rho_grid**2)
    te0_keV = 3.0
    electron_temperature_keV = te0_keV * (1 - rho_grid**2)
    
    return TravisECRHInput(
        antenna_position_cyl=antenna_position_cyl,
        target_position=target_position,
        target_coords_type="W7X-angles",
        frequency_ghz=frequency_ghz,
        power_mw=1.0,
        equilibrium_file=equilibrium_file,
        rho_grid=rho_grid,
        electron_density_1e20=electron_density_1e20,
        electron_temperature_keV=electron_temperature_keV,
        max_steps=10000,
        max_length_m=20.0,
        hamiltonian="West",
        dielectric_tracing="cold",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate TRAVIS reference data")
    parser.add_argument("--travis-exe", type=Path, required=True, help="Path to TRAVIS executable")
    parser.add_argument("--equilibrium", type=Path, required=True,
                       help="Path to VMEC wout file")
    parser.add_argument("--output", type=Path, default=Path("data/travis_reference_w7x.json"), 
                       help="Output JSON file")
    parser.add_argument("--output-dir", type=Path, help="Directory for TRAVIS output files")
    
    args = parser.parse_args()
    
    print("Setting up W7-X test case...")
    input_params = create_w7x_test_case(str(args.equilibrium))
    
    print("Running TRAVIS...")
    output = run_travis(args.travis_exe, input_params, output_dir=args.output_dir)
    
    print(f"\nTRAVIS simulation complete:")
    print(f"  Success: {output.success}")
    print(f"  Ray points: {len(output.arc_length_m)}")
    print(f"  Total path length: {output.arc_length_m[-1]:.3f} m")
    print(f"  Final optical depth: {output.optical_depth[-1]:.3f}")
    print(f"  Total absorbed power: {output.total_absorbed_power_mw:.3f} MW")
    
    print(f"\nSaving reference data to {args.output}...")
    save_reference_data(output, args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()
