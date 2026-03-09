---
icon: lucide/settings-2
---

# Solver Settings & Performance Tuning

## Solver Settings

The ODE solver used for ray tracing has four tunable parameters, exposed via [`TracerSettings`][raytrax.types.TracerSettings]. The defaults are chosen to work well for typical ECRH scenarios in metre-scale devices:

```python
import raytrax

settings = raytrax.TracerSettings(
    relative_tolerance=1e-4,   # PID controller rtol
    absolute_tolerance=1e-6,   # PID controller atol
    max_step_size=0.05,        # metres
    max_arc_length=20.0,       # metres
)

result = raytrax.trace(mag_conf, profiles, beam, settings=settings)
```

Because all four fields are ordinary JAX values (not static compile-time constants), **changing them does not trigger a recompilation** (see below).

### When to Adjust the Settings

**`max_step_size`** is the most impactful parameter. The adaptive controller ([`Tsit5`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Tsit5) + [`PIDController`](https://docs.kidger.site/diffrax/api/stepsize_controller/#diffrax.PIDController)) will take steps smaller than this when needed, but never larger. Reducing it forces finer integration near caustics or in steep-gradient regions — at the cost of more steps.

**`relative_tolerance` / `absolute_tolerance`** control the error estimate the PID controller tries to satisfy per step. Tightening them (e.g. `rtol=1e-6`) improves accuracy but increases the number of steps. The defaults are sufficient for power deposition profiles; gradient-based optimisation may benefit from tighter tolerances.

**`max_arc_length`** is a safety cutoff. Rays that haven't terminated (by leaving the plasma or being absorbed) are stopped at this arc length. 20 m is generous for any current fusion device; you would only need to lower it to speed up debugging of misconfigured inputs, or raise it for very large devices.

## JAX Compilation

Raytrax is built on [JAX](https://docs.jax.dev), which compiles Python functions to optimised machine code the first time they are called. On subsequent calls with inputs of the same types and shapes, the cached compiled function is reused — making repeated calls (e.g. inside an optimisation loop) very fast.

The first call to [`trace`][raytrax.api.trace] is therefore noticeably slower than all subsequent ones. This is expected and is the reason Raytrax is designed to be used as a library inside a program, not as a command-line tool.

### What Triggers a Recompilation

JAX recompiles whenever the *structure* of the inputs changes. For Raytrax, this means:

| Change | Recompiles? |
|---|---|
| Beam position, direction, frequency, power | No |
| Profile values (same `rho` grid) | No |
| Magnetic field values (same grid shape) | No |
| [`TracerSettings`][raytrax.types.TracerSettings] tolerances or step sizes | No |
| `rho` grid size (e.g. 40 → 80 points) | **Yes** |
| Magnetic field grid shape | **Yes** |
| Wave mode (`"X"` ↔ `"O"`) | **Yes** |
| Switching between tokamak and stellarator | **Yes** |
| Number of field periods (`nfp`) | No |

In practice, your grid shapes and device type are fixed for a given study, so you typically pay the compilation cost once per Python session.

### Pre-warming the Cache

If you want to avoid the compilation delay inside a time-sensitive loop, call `trace` once with representative inputs before your main computation:

```python
import raytrax

# One-time compilation warm-up
_ = raytrax.trace(mag_conf, profiles, beam)

# Subsequent calls use the cached compiled function
for beam in beam_sweep:
    result = raytrax.trace(mag_conf, profiles, beam)
```

### Persistent Compilation Cache

Pre-warming only helps within a single Python session. JAX also supports a **persistent on-disk cache** that stores compiled XLA artifacts between runs, so even the very first call in a new session is fast:

```python
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

import raytrax  # import after setting the cache dir
```

Alternatively, set the environment variable before starting Python:

```bash
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
```

The cache is keyed on the compiled computation *and* the JAX/XLA version, so it is invalidated automatically when you upgrade JAX. It is safe to share the cache directory across multiple scripts that use Raytrax with the same input shapes.

!!! info
    The cache directory must be set **before** the first JAX operation (and before importing raytrax). Setting it afterwards has no effect on already-compiled functions.
