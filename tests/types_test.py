"""Tests for SafetensorsMixin serialization and RadialProfiles methods."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from raytrax.types import RadialProfiles, SafetensorsMixin


@dataclass
class SimpleConfig(SafetensorsMixin):
    """Test dataclass with mixed field types."""

    array_field: jnp.ndarray
    int_field: int
    bool_field: bool


def test_safetensors_roundtrip():
    """Test save/load roundtrip preserves data."""
    config = SimpleConfig(
        array_field=jnp.array([1.0, 2.0, 3.0]),
        int_field=42,
        bool_field=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.safetensors"
        config.save(str(path))
        loaded = SimpleConfig.load(str(path))

    assert jnp.allclose(config.array_field, loaded.array_field)
    assert config.int_field == loaded.int_field
    assert config.bool_field == loaded.bool_field


def test_safetensors_with_nan():
    """Test that NaN values are preserved."""
    config = SimpleConfig(
        array_field=jnp.array([1.0, jnp.nan, 3.0]),
        int_field=7,
        bool_field=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.safetensors"
        config.save(str(path))
        loaded = SimpleConfig.load(str(path))

    assert jnp.allclose(config.array_field, loaded.array_field, equal_nan=True)
    assert config.int_field == loaded.int_field
    assert config.bool_field == loaded.bool_field


def test_safetensors_multidimensional_arrays():
    """Test that multidimensional arrays work."""
    config = SimpleConfig(
        array_field=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        int_field=100,
        bool_field=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.safetensors"
        config.save(str(path))
        loaded = SimpleConfig.load(str(path))

    assert jnp.allclose(config.array_field, loaded.array_field)
    assert loaded.array_field.shape == (2, 2)


# ---------------------------------------------------------------------------
# RadialProfiles.with_zero_density_at_boundary
# ---------------------------------------------------------------------------


def _make_profiles(ne_edge: float = 0.5, n: int = 200) -> RadialProfiles:
    rho = jnp.linspace(0.0, 1.0, n)
    ne = ne_edge + (1.0 - ne_edge) * (1.0 - rho**2)  # ne(0)=1, ne(1)=ne_edge
    Te = 2.0 * (1.0 - rho**1.5)
    return RadialProfiles(rho=rho, electron_density=ne, electron_temperature=Te)


def test_with_zero_density_at_boundary_zero_at_lcfs():
    """Tapered profile must be exactly zero at rho=1."""
    p = _make_profiles().with_zero_density_at_boundary(0.1)
    np.testing.assert_allclose(float(p.electron_density[-1]), 0.0, atol=1e-12)


def test_with_zero_density_at_boundary_interior_unchanged():
    """Taper must not modify ne deep inside (rho <= 1 - boundary_layer_width)."""
    orig = _make_profiles()
    tapered = orig.with_zero_density_at_boundary(0.1)
    # rho <= 0.9: weight == 1
    mask = np.array(orig.rho) <= 0.9
    np.testing.assert_allclose(
        np.array(tapered.electron_density)[mask],
        np.array(orig.electron_density)[mask],
        rtol=1e-6,
    )


def test_with_zero_density_at_boundary_monotone_in_taper_region():
    """ne must be non-increasing over the taper region."""
    p = _make_profiles().with_zero_density_at_boundary(0.1)
    rho = np.array(p.rho)
    ne = np.array(p.electron_density)
    taper_mask = rho >= 0.9
    diffs = np.diff(ne[taper_mask])
    assert np.all(diffs <= 1e-10), "ne must be non-increasing over the taper region"


def test_with_zero_density_at_boundary_preserves_temperature():
    """Temperature must be unchanged by with_zero_density_at_boundary."""
    orig = _make_profiles()
    tapered = orig.with_zero_density_at_boundary(0.1)
    np.testing.assert_array_equal(
        np.array(tapered.electron_temperature), np.array(orig.electron_temperature)
    )


def test_with_zero_density_at_boundary_returns_new_object():
    """with_zero_density_at_boundary must return a new object, not modify in place."""
    orig = _make_profiles()
    ne_orig_edge = float(orig.electron_density[-1])
    tapered = orig.with_zero_density_at_boundary(0.1)
    assert float(orig.electron_density[-1]) == ne_orig_edge  # original unchanged
    assert float(tapered.electron_density[-1]) == 0.0


def test_with_zero_density_at_boundary_invalid_width_raises():
    """Out-of-range boundary_layer_width must raise ValueError."""
    p = _make_profiles()
    with pytest.raises(ValueError):
        p.with_zero_density_at_boundary(0.0)
    with pytest.raises(ValueError):
        p.with_zero_density_at_boundary(1.5)  # > rho_max=1
    with pytest.raises(ValueError):
        p.with_zero_density_at_boundary(-0.1)
