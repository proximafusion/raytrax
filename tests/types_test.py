"""Tests for SafetensorsMixin serialization."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import pytest

from raytrax.types import SafetensorsMixin


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
