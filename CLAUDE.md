## Project Overview

Raytrax is an ECRH (Electron Cyclotron Resonance Heating) ray tracer for fusion plasmas, built on JAX.
It traces microwave beams through magnetized plasma, computing absorption and power deposition profiles.
Key features: JIT compilation via JAX, automatic differentiation for gradient-based beam optimization.

## Commands

### Build & Install
Ensure you are working in a virtualenv before pip-installing anything.

```bash
pip install -e ".[test]"          # dev install with test deps
```

### Lint & Format
```bash
ruff check .                      # lint
ruff format --check .             # format check
ruff format .                     # auto-format
ruff check --fix .                # auto-fix lint issues
mypy --install-types --non-interactive --ignore-missing-imports src/raytrax
```

Pre-commit hooks run ruff check + ruff format automatically.

### Tests
```bash
pytest                            # all unit tests
pytest tests/solver_test.py       # single test file
pytest -k "test_name"             # single test by name
pytest integration_tests/ -m integration -v -s   # integration tests (slower, needs vmecpp)
```

Test files use the `*_test.py` naming convention. Fixtures for magnetic configurations (W7-X, tokamak) are in `tests/fixtures.py` and re-exported via `tests/conftest.py`.

### Key Patterns
- **Source layout**: `src/raytrax/` is the package root, with subpackages `tracer/` (ODE solver, ray/buffer types), `physics/` (Hamiltonian, absorption, dielectric tensor, etc.), `equilibrium/` (VMEC Fourier evaluation, cylindrical interpolation, `WoutLike` protocol), `math/` (Bessel, Faddeeva, Shkarofsky), `plot/`, `examples/`.
- **64-bit precision required**: JAX must have `jax_enable_x64 = True` before import.
- **Differentiability**: `trace(..., trim=False)` keeps fixed-size padded arrays for use with `jax.grad`/`jax.jacfwd`. The `trim=True` default slices to valid entries but breaks AD.
- **Stellarator symmetry**: The solver maps toroidal angles to the fundamental domain `[0, π/nfp]` and applies symmetry transforms for B-field evaluation.
