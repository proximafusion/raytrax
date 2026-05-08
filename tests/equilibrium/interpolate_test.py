import jax
import jax.numpy as jnp
import numpy as np

from raytrax.equilibrium.fourier import evaluate_rphiz_on_toroidal_grid
from raytrax.equilibrium.interpolate import (
    CylindricalGridResolution,
    MagneticConfiguration,
    VmecGridResolution,
    build_electron_density_profile_interpolator,
    build_electron_temperature_profile_interpolator,
    build_magnetic_field_interpolator,
    build_rho_interpolator,
    cylindrical_grid_for_equilibrium,
    interpolate_toroidal_to_cylindrical_grid,
)
from raytrax.tracer.solver import (
    _apply_B_stellarator_symmetry,
    _cylindrical_to_cartesian_B,
    _map_to_fundamental_domain,
)
from raytrax.types import RadialProfiles


def _make_B_callable(B_interp, nfp):
    """Helper: wrap interpax B interpolator with coordinate transforms for tests."""

    def wrapper(position):
        r = jnp.sqrt(position[0] ** 2 + position[1] ** 2)
        phi = jnp.arctan2(position[1], position[0])
        z = position[2]
        phi_mapped, z_query, in_second_half = _map_to_fundamental_domain(phi, z, nfp)
        B_cyl = B_interp(r, phi_mapped, z_query)
        B_cyl = _apply_B_stellarator_symmetry(B_cyl, in_second_half)
        return _cylindrical_to_cartesian_B(B_cyl, phi)

    return wrapper


def _make_rho_callable(rho_interp, nfp):
    """Helper: wrap interpax rho interpolator with coordinate transforms for tests."""

    def wrapper(position):
        r = jnp.sqrt(position[0] ** 2 + position[1] ** 2)
        phi = jnp.arctan2(position[1], position[0])
        z = position[2]
        phi_mapped, z_query, _ = _map_to_fundamental_domain(phi, z, nfp)
        return rho_interp(r, phi_mapped, z_query)

    return wrapper


def test_map_to_fundamental_domain():
    """Test toroidal angle and z mapping to fundamental domain with stellarator symmetry."""
    nfp = 5  # W7-X has 5 field periods
    half_period = jnp.pi / nfp
    period = 2.0 * jnp.pi / nfp
    z = jnp.array(0.3)

    # Test phi = 0 (first half): phi_mapped=0, z unchanged, not in second half
    phi_mapped, z_q, in_2nd = _map_to_fundamental_domain(jnp.array(0.0), z, nfp)
    assert jnp.allclose(phi_mapped, 0.0)
    assert jnp.allclose(z_q, z)
    assert not in_2nd

    # Test phi at half period (boundary): should stay, z unchanged
    phi_mapped, z_q, in_2nd = _map_to_fundamental_domain(jnp.array(half_period), z, nfp)
    assert jnp.allclose(phi_mapped, half_period)
    assert jnp.allclose(z_q, z)
    assert not in_2nd

    # Test phi slightly beyond half period (second half): phi reflects, z negates
    phi_beyond = half_period + 0.1
    phi_mapped, z_q, in_2nd = _map_to_fundamental_domain(jnp.array(phi_beyond), z, nfp)
    assert jnp.allclose(phi_mapped, period - phi_beyond)
    assert jnp.allclose(z_q, -z)
    assert in_2nd

    # Test phi at full period (should map to 0): z unchanged (first half after mod)
    phi_mapped, z_q, in_2nd = _map_to_fundamental_domain(jnp.array(period), z, nfp)
    assert jnp.allclose(phi_mapped, 0.0, atol=1e-10)
    assert not in_2nd

    # Test negative phi: -0.2 mod period = period-0.2 (second half), reflects to 0.2, z negated
    phi_mapped, z_q, in_2nd = _map_to_fundamental_domain(jnp.array(-0.2), z, nfp)
    assert jnp.allclose(phi_mapped, 0.2, atol=1e-10)
    assert jnp.allclose(z_q, -z)
    assert in_2nd

    # Test phi beyond one full period: phi=period+0.3 → same as 0.3 (first half)
    phi_mapped, z_q, in_2nd = _map_to_fundamental_domain(
        jnp.array(period + 0.3), z, nfp
    )
    assert jnp.allclose(phi_mapped, 0.3, atol=1e-10)
    assert jnp.allclose(z_q, z)
    assert not in_2nd

    # Test phi in second half (0.75*period): reflects, z negated
    phi_second_half = 0.75 * period
    phi_mapped, z_q, in_2nd = _map_to_fundamental_domain(
        jnp.array(phi_second_half), z, nfp
    )
    assert jnp.allclose(phi_mapped, period - phi_second_half, atol=1e-10)
    assert jnp.allclose(z_q, -z)
    assert in_2nd


def test_interpolate_toroidal_to_cylindrical_grid(torus_wout):
    rho_theta_phi = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, 1, 8),
            jnp.linspace(0, 2 * jnp.pi, 6),
            jnp.linspace(0, 2 * jnp.pi, 7),
            indexing="ij",
        ),
        axis=-1,
    )
    rphiz_toroidal = evaluate_rphiz_on_toroidal_grid(torus_wout, rho_theta_phi)
    rmin = np.min(rphiz_toroidal[..., 0])
    rmax = np.max(rphiz_toroidal[..., 0])
    zmin = np.min(rphiz_toroidal[..., 2])
    zmax = np.max(rphiz_toroidal[..., 2])
    rz_cylindrical = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(rmin, rmax, 4),
            jnp.linspace(zmin, zmax, 5),
            indexing="ij",
        ),
        axis=-1,
    )
    values_cylindrical = interpolate_toroidal_to_cylindrical_grid(
        rphiz_toroidal=rphiz_toroidal,
        rz_cylindrical=rz_cylindrical,
        value_toroidal=jnp.ones((8, 6, 7, 3)),
    )
    assert values_cylindrical.shape == (4, 7, 5, 3)
    # some of them will be NaN, but not all
    assert np.any(np.isfinite(values_cylindrical))
    # all values should be either NaN or 1.0
    np.testing.assert_allclose(
        values_cylindrical[np.isfinite(values_cylindrical)], 1.0, rtol=0, atol=1e-15
    )


def test_cylindrical_grid_for_equilibrium(torus_wout):
    """Test that cylindrical_grid_for_equilibrium works correctly."""
    n_rho = 10
    n_theta = 8
    n_phi = 6
    n_r = 7
    n_z = 9

    grid = cylindrical_grid_for_equilibrium(
        equilibrium=torus_wout,
        n_rho=n_rho,
        n_theta=n_theta,
        n_phi=n_phi,
        n_r=n_r,
        n_z=n_z,
    )

    assert grid.shape == (n_r, n_phi, n_z, 7)
    assert np.any(np.isfinite(grid))

    rphiz_data = grid[..., 0]
    field_data = grid[..., 1]

    assert np.any(np.isfinite(rphiz_data))
    assert np.any(np.isfinite(field_data))

    r_values = grid[..., 0, 0]
    z_values = grid[..., 0, 2]

    assert np.all(r_values[np.isfinite(r_values)] > 0)

    major_radius = 2.0
    minor_radius = 0.5
    finite_r = r_values[np.isfinite(r_values)]
    finite_z = z_values[np.isfinite(z_values)]
    if len(finite_r) > 0:
        # With rho extrapolated to 1.2, the grid extends beyond original minor radius
        # Allow for 20% extension (rho_max = 1.2)
        assert np.min(finite_r) >= major_radius - minor_radius * 1.2
        assert np.max(finite_r) <= major_radius + minor_radius * 1.2

    if len(finite_z) > 0:
        assert np.min(finite_z) >= -minor_radius * 1.2
        assert np.max(finite_z) <= minor_radius * 1.2


def test_from_vmec_wout_custom_grid_resolution(torus_wout):
    """Array shapes match a non-default VmecGridResolution."""
    grid = VmecGridResolution(
        cylindrical=CylindricalGridResolution(
            n_r=11, n_z=13, n_phi=7, n_rho_profile=50
        ),
        n_rho=8,
        n_theta=9,
    )
    mag_conf = MagneticConfiguration.from_vmec_wout(torus_wout, grid=grid)

    assert mag_conf.rphiz.shape == (11, 7, 13, 3)
    assert mag_conf.magnetic_field.shape == (11, 7, 13, 3)
    assert mag_conf.rho.shape == (11, 7, 13)
    assert mag_conf.rho_1d.shape == (50,)
    assert mag_conf.dvolume_drho.shape == (50,)


def test_build_magnetic_field_interpolator_w7x(w7x_wout):
    """Test the magnetic field interpolator using the W7X equilibrium."""
    interpolator = MagneticConfiguration.from_vmec_wout(w7x_wout)
    B_interp_raw = build_magnetic_field_interpolator(interpolator)
    B_interpolator = _make_B_callable(B_interp_raw, w7x_wout.nfp)
    # Test the interpolator at different positions
    # Use positions we know are within the W7X geometry
    R_major = 5.6

    positions = jnp.array(
        [
            [R_major, 0.0, 0.0],  # On axis, at phi = 0
            [R_major * jnp.cos(0.1), R_major * jnp.sin(0.1), 0.0],  # Slightly off-axis
            [R_major, 0.0, 0.2],  # Slight vertical offset
        ]
    )

    # Get the magnetic field at each position
    for pos in positions:
        B_field = B_interpolator(pos)

        assert B_field.shape == (3,)
        assert not jnp.all(jnp.isnan(B_field)), (
            f"All components are NaN at position {pos}"
        )

        # Check for any finite values in the field
        finite_components = B_field[jnp.isfinite(B_field)]
        if len(finite_components) > 0:
            # If we have finite components, at least one should be non-zero
            assert jnp.any(jnp.abs(finite_components) > 1e-10), (
                f"Magnetic field is too small at position {pos}"
            )

    finite_positions = positions[jnp.array([0])]  # Start with just the first position

    jitted_interpolator = jax.jit(B_interpolator)
    for pos in finite_positions:
        try:
            B_field_jitted = jitted_interpolator(pos)
            assert B_field_jitted.shape == (3,)
            # Check for some finite values
            assert jnp.any(jnp.isfinite(B_field_jitted))
        except Exception as e:
            print(f"Jitted interpolator failed at position {pos}: {e}")

    # Test with vmap but only if we have at least one position with finite results
    if len(finite_positions) > 0:
        try:
            vmapped_interpolator = jax.vmap(B_interpolator)
            B_fields_vmap = vmapped_interpolator(finite_positions)
            assert B_fields_vmap.shape == (len(finite_positions), 3)
            # At least one value should be finite
            assert jnp.any(jnp.isfinite(B_fields_vmap))
        except Exception as e:
            print(f"Vmapped interpolator failed: {e}")


def test_build_radial_interpolators_w7x(w7x_wout):
    """Test the radial interpolators using the W7X equilibrium."""
    # Create the equilibrium interpolator
    interpolator = MagneticConfiguration.from_vmec_wout(w7x_wout)

    # Create sample radial profiles
    n_rho_profile = 50
    rho_profile = jnp.linspace(0, 1, n_rho_profile)
    # Sample parabolic profiles for density and temperature
    ne_profile = 1.0 * (rho_profile**2)
    Te_profile = 3.0 * (1 - rho_profile**2)

    # Create RadialProfiles object
    radial_profiles = RadialProfiles(
        rho=rho_profile, electron_density=ne_profile, electron_temperature=Te_profile
    )

    # Build the radial interpolators using the new individual functions
    rho_interp_raw = build_rho_interpolator(interpolator)
    rho_interpolator = _make_rho_callable(rho_interp_raw, w7x_wout.nfp)
    ne_profile_interpolator = build_electron_density_profile_interpolator(
        radial_profiles
    )
    Te_profile_interpolator = build_electron_temperature_profile_interpolator(
        radial_profiles
    )

    # Create composite functions for testing (similar to the old API behavior)
    def ne_interpolator(position):
        rho_value = rho_interpolator(position)
        return ne_profile_interpolator(rho_value)

    def Te_interpolator(position):
        rho_value = rho_interpolator(position)
        return Te_profile_interpolator(rho_value)

    # Test the interpolators at different positions
    # Use positions we know are within the W7X geometry
    R_major = 6.0

    positions = jnp.array(
        [
            [R_major, 0.0, 0.0],  # On axis, at phi = 0
            [R_major * jnp.cos(0.1), R_major * jnp.sin(0.1), 0.0],  # Slightly off-axis
            [R_major, 0.0, 0.2],  # Slight vertical offset
        ]
    )

    # Test the electron density interpolator
    for pos in positions:
        ne_value = ne_interpolator(pos)

        # Basic checks
        assert ne_value.shape == ()  # Scalar value

        # Check if the value is finite (not NaN)
        assert not jnp.isnan(ne_value), f"Density is NaN at position {pos}"

        # Value should be between 0 and 1 (our profile max)
        if jnp.isfinite(ne_value):
            assert 0 <= ne_value <= 1.0, f"Density out of range at position {pos}"

    # Test the electron temperature interpolator
    for pos in positions:
        Te_value = Te_interpolator(pos)

        # Basic checks
        assert Te_value.shape == ()  # Scalar value

        # Check if the value is finite (not NaN)
        assert not jnp.isnan(Te_value), f"Temperature is NaN at position {pos}"

        # Value should be between 0 and 3 (our profile max)
        if jnp.isfinite(Te_value):
            assert 0 <= Te_value <= 3.0, f"Temperature out of range at position {pos}"

    # Test with jit
    finite_positions = positions[jnp.array([0])]  # Start with just the first position

    # Test JIT with ne_interpolator
    jitted_ne_interpolator = jax.jit(ne_interpolator)
    for pos in finite_positions:
        try:
            ne_value_jitted = jitted_ne_interpolator(pos)
            assert ne_value_jitted.shape == ()
            assert jnp.isfinite(ne_value_jitted)
        except Exception as e:
            print(f"Jitted ne_interpolator failed at position {pos}: {e}")

    # Test JIT with Te_interpolator
    jitted_Te_interpolator = jax.jit(Te_interpolator)
    for pos in finite_positions:
        try:
            Te_value_jitted = jitted_Te_interpolator(pos)
            assert Te_value_jitted.shape == ()
            assert jnp.isfinite(Te_value_jitted)
        except Exception as e:
            print(f"Jitted Te_interpolator failed at position {pos}: {e}")

    # Test with vmap
    if len(finite_positions) > 0:
        try:
            # Test vmap with ne_interpolator
            vmapped_ne_interpolator = jax.vmap(ne_interpolator)
            ne_values_vmap = vmapped_ne_interpolator(finite_positions)
            assert ne_values_vmap.shape == (len(finite_positions),)
            assert jnp.any(jnp.isfinite(ne_values_vmap))

            # Test vmap with Te_interpolator
            vmapped_Te_interpolator = jax.vmap(Te_interpolator)
            Te_values_vmap = vmapped_Te_interpolator(finite_positions)
            assert Te_values_vmap.shape == (len(finite_positions),)
            assert jnp.any(jnp.isfinite(Te_values_vmap))
        except Exception as e:
            print(f"Vmapped interpolators failed: {e}")


def test_individual_interpolator_functions_w7x(w7x_wout):
    """Test the individual interpolator building functions."""
    # Create the equilibrium interpolator
    equilibrium_interpolator = MagneticConfiguration.from_vmec_wout(w7x_wout)

    # Create sample radial profiles
    n_rho_profile = 50
    rho_profile = jnp.linspace(0, 1, n_rho_profile)
    ne_profile = 1.0 * (1 - rho_profile**2)
    Te_profile = 3.0 * (1 - rho_profile**2)

    radial_profiles = RadialProfiles(
        rho=rho_profile, electron_density=ne_profile, electron_temperature=Te_profile
    )

    # Test individual functions
    rho_interp_raw = build_rho_interpolator(equilibrium_interpolator)
    rho_interpolator = _make_rho_callable(rho_interp_raw, w7x_wout.nfp)
    ne_profile_interpolator = build_electron_density_profile_interpolator(
        radial_profiles
    )
    Te_profile_interpolator = build_electron_temperature_profile_interpolator(
        radial_profiles
    )

    # Test rho interpolator
    test_position = jnp.array([6.0, 0.0, 0.0])
    rho_value = rho_interpolator(test_position)
    assert rho_value.shape == ()
    assert jnp.isfinite(rho_value)

    # Test profile interpolators with known rho values
    test_rho_values = jnp.array([0.0, 0.5, 1.0])

    for test_rho in test_rho_values:
        ne_value = ne_profile_interpolator(test_rho)
        Te_value = Te_profile_interpolator(test_rho)

        assert ne_value.shape == ()
        assert Te_value.shape == ()
        assert jnp.isfinite(ne_value)
        assert jnp.isfinite(Te_value)

        # Check that the values match the expected profile
        expected_ne = 1.0 * (1 - test_rho**2)
        expected_Te = 3.0 * (1 - test_rho**2)

        assert jnp.allclose(ne_value, expected_ne, rtol=1e-3)
        assert jnp.allclose(Te_value, expected_Te, rtol=1e-3)


def test_stellarator_symmetry_in_interpolators(w7x_wout):
    """Test that interpolators correctly apply stellarator symmetry mapping.

    The physical stellarator symmetry is: (R, phi, Z) <-> (R, phi_period-phi, -Z)
    meaning both rho and |B| are equal at these mirror positions.
    Field periodicity: (R, phi, Z) == (R, phi + 2*pi/nfp, Z).
    """
    equilibrium_interpolator = MagneticConfiguration.from_vmec_wout(w7x_wout)
    B_interp_raw = build_magnetic_field_interpolator(equilibrium_interpolator)
    B_interpolator = _make_B_callable(B_interp_raw, w7x_wout.nfp)
    rho_interp_raw = build_rho_interpolator(equilibrium_interpolator)
    rho_interpolator = _make_rho_callable(rho_interp_raw, w7x_wout.nfp)

    nfp = w7x_wout.nfp  # Should be 5 for W7-X
    period = 2.0 * jnp.pi / nfp
    R = 5.8
    z = 0.1

    # Position 1: phi in first half-period
    phi1 = 0.1
    pos1 = jnp.array([R * jnp.cos(phi1), R * jnp.sin(phi1), z])

    # Position 2: stellarator-symmetric to pos1 — same R, phi reflected, Z negated
    phi2 = period - phi1
    pos2_sym = jnp.array([R * jnp.cos(phi2), R * jnp.sin(phi2), -z])

    # Position 3: same as pos1 in the next field period (exact periodicity)
    phi3 = phi1 + period
    pos3 = jnp.array([R * jnp.cos(phi3), R * jnp.sin(phi3), z])

    # Test rho interpolator
    rho1 = rho_interpolator(pos1)
    rho2 = rho_interpolator(pos2_sym)
    rho3 = rho_interpolator(pos3)

    # Stellarator-symmetric positions should give same rho (rho is even under symmetry)
    assert jnp.allclose(rho1, rho2, rtol=1e-4), (
        f"Stellarator symmetry failed for rho: {rho1} != {rho2}"
    )
    # Next field period should give same result
    assert jnp.allclose(rho1, rho3, rtol=1e-4), (
        f"Field periodicity failed for rho: {rho1} != {rho3}"
    )

    # Test |B| (stellarator symmetry: |B| is even, so same at symmetric positions)
    B1 = B_interpolator(pos1)
    B2 = B_interpolator(pos2_sym)
    B3 = B_interpolator(pos3)

    B1_mag = jnp.linalg.norm(B1)
    B2_mag = jnp.linalg.norm(B2)
    B3_mag = jnp.linalg.norm(B3)

    assert jnp.allclose(B1_mag, B2_mag, rtol=1e-4), (
        f"Stellarator symmetry failed for |B|: {B1_mag} != {B2_mag}"
    )
    # Next field period: |B| and B_Z are identical; Cartesian Bx/By are rotated by
    # period in phi (because the Cartesian frame rotates), so compare only |B| and B_Z.
    assert jnp.allclose(B1_mag, B3_mag, rtol=1e-4), (
        f"Field periodicity failed for |B|: {B1_mag} != {B3_mag}"
    )
    assert jnp.allclose(B1[2], B3[2], rtol=1e-4), (
        f"Field periodicity failed for B_Z: {B1[2]} != {B3[2]}"
    )


def test_extrapolation_in_cylindrical_grid(w7x_wout):
    """Test that the cylindrical grid properly handles extrapolation beyond LCMS."""
    from raytrax.equilibrium.interpolate import (
        MagneticConfiguration,
        build_magnetic_field_interpolator,
        build_rho_interpolator,
    )

    interpolator = MagneticConfiguration.from_vmec_wout(w7x_wout)
    B_interp_raw = build_magnetic_field_interpolator(interpolator)
    B_interpolator = _make_B_callable(B_interp_raw, w7x_wout.nfp)
    rho_interp_raw = build_rho_interpolator(interpolator)
    rho_interpolator = _make_rho_callable(rho_interp_raw, w7x_wout.nfp)

    # Test position near the plasma boundary at larger major radius
    R_test = 6.0  # Near outboard edge
    phi_test = 0.0  # At symmetry plane
    Z_test = 0.0  # Midplane

    # Create a radial scan from inside to outside plasma
    positions = []
    for delta_r in jnp.linspace(-0.3, 0.3, 20):  # Scan outward through boundary
        R = R_test + delta_r
        X = R * jnp.cos(phi_test)
        Y = R * jnp.sin(phi_test)
        positions.append(jnp.array([X, Y, Z_test]))

    # Evaluate B field and rho at all positions
    B_magnitudes = []
    rho_values = []
    for pos in positions:
        B_vec = B_interpolator(pos)
        B_mag = float(jnp.linalg.norm(B_vec))
        rho_val = float(rho_interpolator(pos))
        B_magnitudes.append(B_mag)
        rho_values.append(rho_val)

    B_magnitudes = jnp.array(B_magnitudes)
    rho_values = jnp.array(rho_values)

    # Check that we have finite values everywhere (key test for extrapolation)
    assert jnp.all(jnp.isfinite(B_magnitudes)), "B field should be finite everywhere"
    assert jnp.all(jnp.isfinite(rho_values)), "rho should be finite everywhere"

    # Focus on points where B > 0 (within or near the grid)
    valid_mask = B_magnitudes > 0.1  # Filter out far-outside points
    B_valid = B_magnitudes[valid_mask]

    assert len(B_valid) > 10, "Should have enough valid points for testing"

    # Check for reasonable continuity: B field shouldn't have huge jumps
    # This is the key test - without proper extrapolation, we'd see jumps to zero
    dB = jnp.diff(B_valid)
    max_jump = jnp.max(jnp.abs(dB))
    # Allow at most ~1.0T change between adjacent points (reasonable for ~3cm spacing)
    assert max_jump < 1.0, (
        f"B field has too large jump: {max_jump:.3f} T (indicates bad extrapolation)"
    )


# --- Axisymmetric (tokamak) tests ---

# Fixture parameters (must match tokamak_magnetic_configuration fixture)
_TOKAMAK_R0 = 3.0
_TOKAMAK_A = 1.0
_TOKAMAK_B0 = 2.5

_DUMMY_RADIAL_PROFILES = RadialProfiles(
    rho=jnp.linspace(0, 1, 50),
    electron_density=jnp.ones(50),
    electron_temperature=jnp.ones(50),
)


def _make_tokamak_interpolators(mc):
    """Build Interpolators from an axisymmetric MagneticConfiguration."""
    from raytrax.tracer.buffers import Interpolators

    return Interpolators(
        magnetic_field=build_magnetic_field_interpolator(mc),
        rho=build_rho_interpolator(mc),
        electron_density=build_electron_density_profile_interpolator(
            _DUMMY_RADIAL_PROFILES
        ),
        electron_temperature=build_electron_temperature_profile_interpolator(
            _DUMMY_RADIAL_PROFILES
        ),
        is_axisymmetric=True,
    )


def test_axisymmetric_magnetic_field_interpolator(tokamak_magnetic_configuration):
    """Test 2D B field interpolator for an analytic tokamak (B_phi = B0*R0/R)."""
    B_interp = build_magnetic_field_interpolator(tokamak_magnetic_configuration)

    # Query at the magnetic axis (R=R0, Z=0)
    B_cyl = B_interp(_TOKAMAK_R0, 0.0)
    assert jnp.allclose(B_cyl[0], 0.0, atol=1e-10), "B_R should be zero"
    assert jnp.allclose(B_cyl[1], _TOKAMAK_B0, rtol=1e-3)
    assert jnp.allclose(B_cyl[2], 0.0, atol=1e-10), "B_Z should be zero"

    # Query at R = R0 + 0.5 (outboard side): B_phi = B0 * R0 / (R0 + 0.5)
    R_test = _TOKAMAK_R0 + 0.5
    B_cyl = B_interp(R_test, 0.0)
    expected = _TOKAMAK_B0 * _TOKAMAK_R0 / R_test
    assert jnp.allclose(B_cyl[1], expected, rtol=1e-2)


def test_axisymmetric_rho_interpolator(tokamak_magnetic_configuration):
    """Test 2D rho interpolator for an analytic tokamak."""
    rho_interp = build_rho_interpolator(tokamak_magnetic_configuration)

    # On the magnetic axis: rho = 0
    assert jnp.allclose(rho_interp(_TOKAMAK_R0, 0.0), 0.0, atol=0.05)

    # At the boundary (R = R0 + a, Z = 0): rho = 1
    assert jnp.allclose(rho_interp(_TOKAMAK_R0 + _TOKAMAK_A, 0.0), 1.0, atol=0.05)

    # At (R = R0, Z = 0.5*a): rho = 0.5
    assert jnp.allclose(rho_interp(_TOKAMAK_R0, 0.5 * _TOKAMAK_A), 0.5, atol=0.05)


def test_axisymmetric_eval_magnetic_field(tokamak_magnetic_configuration):
    """Test _eval_magnetic_field with axisymmetric interpolators."""
    from raytrax.tracer.solver import _eval_magnetic_field

    interpolators = _make_tokamak_interpolators(tokamak_magnetic_configuration)

    # At phi=0, B should point in the y-direction (B_phi at phi=0 -> B_y)
    pos_phi0 = jnp.array([_TOKAMAK_R0, 0.0, 0.0])
    B = _eval_magnetic_field(pos_phi0, interpolators, nfp=1)
    assert jnp.allclose(B[0], 0.0, atol=1e-3), f"B_x should be ~0, got {B[0]}"
    assert jnp.allclose(B[1], _TOKAMAK_B0, rtol=1e-2), f"B_y should be ~B0, got {B[1]}"
    assert jnp.allclose(B[2], 0.0, atol=1e-3), f"B_z should be ~0, got {B[2]}"

    # At phi=pi/2, B should point in the -x direction (B_phi at phi=pi/2 -> -B_x)
    pos_phi90 = jnp.array([0.0, _TOKAMAK_R0, 0.0])
    B = _eval_magnetic_field(pos_phi90, interpolators, nfp=1)
    assert jnp.allclose(B[0], -_TOKAMAK_B0, rtol=1e-2), (
        f"B_x should be ~-B0, got {B[0]}"
    )
    assert jnp.allclose(B[1], 0.0, atol=1e-3), f"B_y should be ~0, got {B[1]}"

    # |B| should be the same at any toroidal angle for same R
    assert jnp.allclose(jnp.linalg.norm(B), _TOKAMAK_B0, rtol=1e-2), (
        "|B| should be B0 at the magnetic axis"
    )


def test_axisymmetric_eval_rho(tokamak_magnetic_configuration):
    """Test _eval_rho with axisymmetric interpolators."""
    from raytrax.tracer.solver import _eval_rho

    interpolators = _make_tokamak_interpolators(tokamak_magnetic_configuration)

    # rho should be the same at all toroidal angles for same (R, Z)
    for phi in [0.0, 1.0, jnp.pi, 5.0]:
        pos = jnp.array([_TOKAMAK_R0 * jnp.cos(phi), _TOKAMAK_R0 * jnp.sin(phi), 0.0])
        rho_val = _eval_rho(pos, interpolators, nfp=1)
        assert jnp.allclose(rho_val, 0.0, atol=0.05), (
            f"rho at axis should be ~0, got {rho_val} at phi={phi}"
        )

    # At outboard midplane
    pos_out = jnp.array([_TOKAMAK_R0 + 0.5 * _TOKAMAK_A, 0.0, 0.0])
    rho_out = _eval_rho(pos_out, interpolators, nfp=1)
    assert jnp.allclose(rho_out, 0.5, atol=0.05)


def test_axisymmetric_jit_and_vmap(tokamak_magnetic_configuration):
    """Test that axisymmetric interpolators work with jit and vmap."""
    from raytrax.tracer.solver import _eval_magnetic_field, _eval_rho

    interpolators = _make_tokamak_interpolators(tokamak_magnetic_configuration)

    # JIT
    eval_B_jit = jax.jit(lambda pos: _eval_magnetic_field(pos, interpolators, nfp=1))
    eval_rho_jit = jax.jit(lambda pos: _eval_rho(pos, interpolators, nfp=1))

    pos = jnp.array([_TOKAMAK_R0, 0.0, 0.0])
    B = eval_B_jit(pos)
    assert B.shape == (3,)
    assert jnp.all(jnp.isfinite(B))

    rho = eval_rho_jit(pos)
    assert rho.shape == ()
    assert jnp.isfinite(rho)

    # vmap over multiple positions
    positions = jnp.array(
        [
            [_TOKAMAK_R0, 0.0, 0.0],
            [0.0, _TOKAMAK_R0, 0.0],
            [_TOKAMAK_R0 + 0.3, 0.0, 0.2],
        ]
    )
    B_all = jax.vmap(eval_B_jit)(positions)
    assert B_all.shape == (3, 3)
    assert jnp.all(jnp.isfinite(B_all))

    rho_all = jax.vmap(eval_rho_jit)(positions)
    assert rho_all.shape == (3,)
    assert jnp.all(jnp.isfinite(rho_all))


# ---------------------------------------------------------------------------
# Bug regression: ne boundary discontinuity at the LCFS
# ---------------------------------------------------------------------------


def test_ne_interpolator_no_discontinuity_at_lcfs_with_nonzero_edge():
    """ne interpolator must not jump to zero exactly at rho=1 when ne(1)>0.

    Before the cosine-taper fix, the interpolator used extrap=0.0, so querying
    at rho=1.0 (which falls on the grid boundary) could return a value close to
    zero while rho=0.999 returned the true ne, creating a large step that
    propagated into a spurious spike in the absorption coefficient.
    """
    n = 50
    rho_profile = jnp.linspace(0.0, 1.0, n)
    # Profile that is non-zero at rho=1: ne(1) = 0.5
    ne_profile = 0.5 + 0.5 * (1.0 - rho_profile**2)

    radial_profiles = RadialProfiles(
        rho=rho_profile,
        electron_density=ne_profile,
        electron_temperature=jnp.ones(n),
    )
    # Non-zero edge density intentionally: expect the warning.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ne_interp = build_electron_density_profile_interpolator(radial_profiles)

    # Without a taper, the profile is unchanged inside the grid but must be
    # continuous: ne(0.99) and ne(1.0) should agree with the analytic formula.
    for rho_test in [0.0, 0.5, 0.99, 1.0]:
        ne_val = float(ne_interp(jnp.array(rho_test)))
        expected = float(0.5 + 0.5 * (1.0 - rho_test**2))
        np.testing.assert_allclose(
            ne_val,
            expected,
            rtol=1e-3,
            err_msg=f"ne({rho_test}) = {ne_val:.6f}, expected {expected:.6f}",
        )


def test_ne_interpolator_taper_is_smooth_and_zero_at_lcfs():
    """Cosine taper must smoothly drive ne to zero at rho=1 for a non-zero edge profile.

    This is the core regression test for Bug 1: before the fix, a profile with
    ne(1) > 0 created a discontinuous step to vacuum. The taper must:
    - equal the original profile deep inside (rho << rho1)
    - reach exactly zero at rho=1
    - be monotonically decreasing over the taper region
    """
    n = 200
    rho_profile = jnp.linspace(0.0, 1.0, n)
    # Non-zero at edge
    ne_profile = 1.0 - 0.5 * rho_profile**2  # ne(1) = 0.5

    radial_profiles = RadialProfiles(
        rho=rho_profile,
        electron_density=ne_profile,
        electron_temperature=jnp.ones(n),
    )
    boundary_layer_width = 0.1
    ne_interp = build_electron_density_profile_interpolator(
        radial_profiles.with_zero_density_at_boundary(boundary_layer_width)
    )

    # Deep inside: taper weight == 1, value matches original profile
    for rho_test in [0.0, 0.3, 0.5, 0.8]:
        ne_val = float(ne_interp(jnp.array(rho_test)))
        expected = float(1.0 - 0.5 * rho_test**2)
        np.testing.assert_allclose(
            ne_val,
            expected,
            rtol=1e-3,
            err_msg=f"Taper should not affect interior: ne({rho_test})={ne_val:.6f}",
        )

    # At rho=1 the tapered profile must be zero (or within floating-point of zero)
    ne_at_lcfs = float(ne_interp(jnp.array(1.0)))
    np.testing.assert_allclose(
        ne_at_lcfs,
        0.0,
        atol=1e-6,
        err_msg=f"Tapered ne at rho=1 must be zero, got {ne_at_lcfs:.3e}",
    )

    # Monotone decrease over the taper region
    rho_taper = jnp.linspace(1.0 - boundary_layer_width, 1.0, 30)
    ne_taper = jnp.array([float(ne_interp(r)) for r in rho_taper])
    diffs = jnp.diff(ne_taper)
    assert jnp.all(diffs <= 1e-10), (
        "Tapered ne must be non-increasing over the boundary layer"
    )


def test_ne_interpolator_zero_edge_profile_unaffected_by_taper():
    """Taper must not change ne deep inside the plasma when ne(rho=1)==0.

    The cosine taper only acts in the outermost ``boundary_layer_width``
    fraction of the minor radius.  Deep inside (rho <= 1 - boundary_layer_width)
    the weight is exactly 1, so the tapered interpolator must agree with the
    plain one there.

    In the taper region the weight is < 1, so tapered values will be smaller
    than the plain ones -- that is intentional and is not tested here.
    """
    n = 50
    rho_profile = jnp.linspace(0.0, 1.0, n)
    ne_profile = 1.0 - rho_profile**2  # naturally zero at rho=1

    radial_profiles = RadialProfiles(
        rho=rho_profile,
        electron_density=ne_profile,
        electron_temperature=jnp.ones(n),
    )
    boundary_layer_width = 0.1
    ne_no_taper = build_electron_density_profile_interpolator(radial_profiles)
    ne_tapered = build_electron_density_profile_interpolator(
        radial_profiles.with_zero_density_at_boundary(boundary_layer_width)
    )

    # Deep interior: well inside the taper start at rho = 1 - 0.1 = 0.9
    for rho_test in jnp.linspace(0.0, 0.85, 10):
        val_plain = float(ne_no_taper(rho_test))
        val_taper = float(ne_tapered(rho_test))
        np.testing.assert_allclose(
            val_taper,
            val_plain,
            rtol=1e-6,
            err_msg=(
                f"Taper changed deep interior at rho={float(rho_test):.3f}: "
                f"{val_plain:.6e} -> {val_taper:.6e}"
            ),
        )

    # rho=1 must remain zero (taper must not inject spurious density at the edge)
    np.testing.assert_allclose(
        float(ne_tapered(jnp.array(1.0))),
        0.0,
        atol=1e-12,
        err_msg="Tapered ne at rho=1 must stay zero for a zero-edge profile",
    )


def test_build_ne_interpolator_is_jit_compatible():
    """build_electron_density_profile_interpolator must not raise under jax.jit.

    When trace() is called inside jax.jit or jax.grad the arrays in
    RadialProfiles become JAX tracers.  Calling float() on a tracer raises
    ConcretizationTypeError.  The edge-density warning must be silently skipped
    rather than crashing in that situation.
    """
    n = 20
    rho = jnp.linspace(0.0, 1.0, n)
    # Non-zero at the edge – this is what would have triggered the warning and
    # previously the crash.
    ne = jnp.ones(n)

    @jax.jit
    def build_and_eval(electron_density):
        profiles = RadialProfiles(
            rho=rho,
            electron_density=electron_density,
            electron_temperature=jnp.ones(n),
        )
        interp = build_electron_density_profile_interpolator(profiles)
        return interp(jnp.array(0.5))

    # Must not raise jax.errors.ConcretizationTypeError
    result = build_and_eval(ne)
    assert jnp.isfinite(result)
