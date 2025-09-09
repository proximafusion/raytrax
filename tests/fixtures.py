import urllib.request
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def download_file(url: str, dest_path: Path):
    """Download a file if it doesn't exist."""
    if not dest_path.exists():
        print(f"Downloading {url} to {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    else:
        print(f"File already exists at {dest_path}.")


@dataclass
class TestWout:
    rmnc: jax.Array
    zmns: jax.Array
    xm: jax.Array
    xn: jax.Array
    bsupumnc: jax.Array
    bsupvmnc: jax.Array
    xm_nyq: jax.Array
    xn_nyq: jax.Array
    ns: int
    nfp: int
    lasym: bool = False


@pytest.fixture
def torus_wout():
    """Fixture for a torus shaped Wout-like object."""
    n_surfaces = 5
    major_radius = 2.0
    minor_radius = 0.5
    rmnc = np.zeros((2, n_surfaces))
    rmnc[0] = major_radius
    rmnc[1] = np.sqrt(np.linspace(0, 1, n_surfaces)) * minor_radius

    xm = np.array([0, 1])
    xn = np.array([0, 0])
    zmns = np.zeros((2, n_surfaces))
    zmns[1] = np.sqrt(np.linspace(0, 1, n_surfaces)) * minor_radius

    xm_nyq = np.array([0, 1])
    xn_nyq = np.array([0, 0])
    bsupumnc = np.zeros((2, n_surfaces))
    bsupvmnc = np.zeros((2, n_surfaces))
    bsupvmnc[0] = 0.7
    return TestWout(
        rmnc=jnp.array(rmnc),
        zmns=jnp.array(zmns),
        xm=jnp.array(xm),
        xn=jnp.array(xn),
        bsupumnc=jnp.array(bsupumnc),
        bsupvmnc=jnp.array(bsupvmnc),
        xm_nyq=jnp.array(xm_nyq),
        xn_nyq=jnp.array(xn_nyq),
        ns=5,
        nfp=1,
        lasym=False,
    )


def equilibrium_from_vmec_input(input_path: Path):
    """Run VMEC++ from a given input file and return the equilibrium."""
    try:
        import vmecpp
    except ImportError:
        pytest.skip("vmecpp not installed, skipping test")

    vmec_input = vmecpp.VmecInput.from_file(input_path)
    print("Running VMEC++...")
    vmec_output = vmecpp.run(vmec_input)
    print("VMEC++ run complete.")
    return vmec_output.wout


@pytest.fixture
def w7x_wout():
    """Fixture for the W7-X equilibrium.

    This fixture:
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

    try:
        from vmecpp import VmecWOut
    except ImportError:
        pytest.skip("vmecpp not installed, skipping test")

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
