"""Utilities for downloading example or test data."""

import urllib.request
from pathlib import Path

import vmecpp
from vmecpp import VmecWOut


def download_file(url: str, dest_path: Path):
    """Download a file if it doesn't exist."""
    if not dest_path.exists():
        print(f"Downloading {url} to {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    else:
        print(f"File already exists at {dest_path}.")


def equilibrium_from_vmec_input(input_path: Path):
    """Run VMEC++ from a given input file and return the equilibrium."""
    vmec_input = vmecpp.VmecInput.from_file(input_path)
    print("Running VMEC++...")
    vmec_output = vmecpp.run(vmec_input)
    print("VMEC++ run complete.")
    return vmec_output.wout


def get_w7x_wout():
    """Function to get the W7-X equilibrium using vmecpp.

    This function does the following:
    1. Downloads the W7-X equilibrium file if needed
    2. Runs vmecpp if needed and creates a JSON cache using model_dump_json
    3. Loads the JSON cache using model_validate_json if it exists

    Returns:
        VmecWOut: The W7-X equilibrium
    """
    # URLs and file paths
    W7X_JSON = "https://github.com/proximafusion/vmecpp/raw/main/examples/data/w7x.json"
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    input_path = data_dir / "w7x.json"
    wout_json_path = data_dir / "w7x_wout.json"

    # Download the input file if needed
    download_file(W7X_JSON, input_path)

    # If the wout JSON exists, load & return it
    if wout_json_path.exists():
        print(f"Loading W7-X equilibrium from cache: {wout_json_path}")
        with open(wout_json_path, "r") as f:
            json_data = f.read()
        return VmecWOut.model_validate_json(json_data)

    print("Creating W7-X equilibrium using vmecpp...")
    wout = equilibrium_from_vmec_input(input_path)

    json_data = wout.model_dump_json()
    with open(wout_json_path, "w") as f:
        f.write(json_data)

    print(f"Saved W7-X equilibrium to cache: {wout_json_path}")
    return wout
