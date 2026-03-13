"""Test that importing raytrax does not initialize the JAX XLA backend.

Similar regressions have happened in `optimistix`
https://github.com/patrick-kidger/optimistix/pull/186 so this test both
guards against regressions in our dependencies and makes sure we don't
accidentally introduce this problem ourselves."""

import subprocess
import sys


def test_import_does_not_initialize_jax_backend():
    script = """
import jax
jax.config.update('jax_enable_x64', True)

import raytrax  # noqa: F401

import jax._src.xla_bridge as xb
initialized = bool(xb._backends)
if initialized:
    raise SystemExit(f"JAX backend was initialized on import: {list(xb._backends.keys())}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"JAX backend was initialized during `import raytrax`.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
