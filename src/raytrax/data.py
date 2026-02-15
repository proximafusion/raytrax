"""Utilities for loading example equilibrium data."""

from pathlib import Path

import vmecpp
from vmecpp import VmecWOut


def get_w7x_wout() -> VmecWOut:
    """Get the W7-X equilibrium using vmecpp.

    Runs VMEC++ on the bundled W7-X input file and caches the result as JSON.

    Returns:
        VmecWOut: The W7-X equilibrium
    """
    data_dir = Path(__file__).parent / "data"
    input_path = data_dir / "w7x.json"
    wout_json_path = data_dir / "w7x_wout.json"

    # Return cached result if available
    if wout_json_path.exists():
        print(f"Loading W7-X equilibrium from cache: {wout_json_path}")
        with open(wout_json_path, "r") as f:
            json_data = f.read()
        return VmecWOut.model_validate_json(json_data)

    print("Creating W7-X equilibrium using vmecpp...")
    vmec_input = vmecpp.VmecInput.from_file(input_path)
    vmec_output = vmecpp.run(vmec_input)
    wout = vmec_output.wout

    json_data = wout.model_dump_json()
    with open(wout_json_path, "w") as f:
        f.write(json_data)

    print(f"Saved W7-X equilibrium to cache: {wout_json_path}")
    return wout
