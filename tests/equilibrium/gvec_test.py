"""Tests for the GVEC equilibrium import.

The unit tests use an analytic stand-in for ``gvec.State`` whose cross-sections
zeta = const are tilted against the planes phi = const, so they exercise the
general geometry without requiring GVEC. The tests at the end compare real GVEC
equilibria of W7-X against VMEC and are skipped unless ``gvec`` and the data in
the ``W7X/`` directory at the repository root are available.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from raytrax.api import trace
from raytrax.equilibrium.gvec import (
    GvecGridResolution,
    cylindrical_grid_for_gvec_equilibrium,
    dvolume_drho_gvec,
)
from raytrax.equilibrium.interpolate import (
    CylindricalGridResolution,
    MagneticConfiguration,
    build_magnetic_field_interpolator,
    build_rho_interpolator,
)
from raytrax.tracer.solver import (
    _apply_B_stellarator_symmetry,
    _cylindrical_to_cartesian_B,
    _map_to_fundamental_domain,
)
from raytrax.types import Beam, RadialProfiles, TracerSettings

jax.config.update("jax_enable_x64", True)


class _DataArray:
    """The part of ``xarray.DataArray`` used by the importer."""

    def __init__(self, values, dims):
        self.values = values
        self.dims = dims

    def transpose(self, *dims):
        return _DataArray(
            np.transpose(self.values, [self.dims.index(d) for d in dims]), dims
        )


class _TiltedTorusState:
    """Analytic stand-in for ``gvec.State``.

    Circular cross-sections of radius rho * a(zeta) around a circular axis of
    radius R0. The cross-section plane at zeta is spanned by e_R and
    cos(alpha) e_Z + sin(alpha) e_phi, i.e. tilted by alpha(zeta) against the
    plane phi = const. Like GVEC, phi runs along -zeta. B is set to zero at
    rho = 0, where the Jacobian vanishes, so that using the axis value fails.

    Args:
        nfp: Reported number of field periods.
        geometry_nfp: Periodicity of the geometry (defaults to ``nfp``).
        tilt: Mean tilt angle alpha in radians.
        tilt_modulation: Relative toroidal modulation of the tilt angle.
        shaping: Relative toroidal modulation of the minor radius.
        wobble: Amplitude of phi(zeta) = -zeta + wobble * sin(nfp zeta) on the axis.
        asymmetry: Amplitude of a stellarator-asymmetric shaping term.
    """

    R0 = 5.5
    a0 = 0.5
    B0 = 2.5

    def __init__(
        self,
        nfp=5,
        geometry_nfp=None,
        tilt=0.3,
        tilt_modulation=0.5,
        shaping=0.2,
        wobble=0.0,
        asymmetry=0.0,
    ):
        self.nfp = nfp
        self.m = geometry_nfp or nfp
        self.tilt = tilt
        self.tilt_modulation = tilt_modulation
        self.shaping = shaping
        self.wobble = wobble
        self.asymmetry = asymmetry

    def position(self, rho, theta, zeta):
        m = self.m
        phi = -zeta + self.wobble * np.sin(m * zeta)
        a = self.a0 * (
            1 + self.shaping * np.cos(m * zeta) + self.asymmetry * np.sin(m * zeta)
        )
        alpha = self.tilt * (1 + self.tilt_modulation * np.cos(m * zeta))
        e_R = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1)
        e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
        e_Z = np.array([0.0, 0.0, 1.0])
        e_2 = np.cos(alpha)[..., None] * e_Z + np.sin(alpha)[..., None] * e_phi
        r1 = (rho * a * np.cos(theta))[..., None]
        r2 = (rho * a * np.sin(theta))[..., None]
        return self.R0 * e_R + r1 * e_R + r2 * e_2

    def field(self, x):
        """Analytic stellarator-symmetric field B(x) in Cartesian components."""
        R = np.hypot(x[..., 0], x[..., 1])
        phi = np.arctan2(x[..., 1], x[..., 0])
        Z = x[..., 2]
        B_R = 0.3 * Z
        B_phi = self.B0 * self.R0 / R
        B_Z = 0.2 + 0.1 * (R - self.R0)
        return np.stack(
            [
                B_R * np.cos(phi) - B_phi * np.sin(phi),
                B_R * np.sin(phi) + B_phi * np.cos(phi),
                B_Z,
            ],
            axis=-1,
        )

    def evaluate(self, *quantities, rho, theta, zeta):
        r, t, z = np.meshgrid(rho, theta, zeta, indexing="ij")
        out = {}
        for quantity in quantities:
            if quantity == "pos":
                out["pos"] = _DataArray(
                    np.moveaxis(self.position(r, t, z), -1, 0),
                    ("xyz", "rad", "pol", "tor"),
                )
            elif quantity == "B":
                B = self.field(self.position(r, t, z))
                B[r == 0] = 0.0
                out["B"] = _DataArray(B, ("rad", "pol", "tor", "xyz"))
            elif quantity == "Jac":
                h = 1e-6
                columns = [
                    (self.position(r + h, t, z) - self.position(r - h, t, z)) / (2 * h),
                    (self.position(r, t + h, z) - self.position(r, t - h, z)) / (2 * h),
                    (self.position(r, t, z + h) - self.position(r, t, z - h)) / (2 * h),
                ]
                jac = np.abs(np.linalg.det(np.stack(columns, axis=-1)))
                out["Jac"] = _DataArray(jac, ("rad", "pol", "tor"))
        return out


_GRID = GvecGridResolution(
    cylindrical=CylindricalGridResolution(n_r=70, n_z=70, n_phi=24, n_rho_profile=50),
    n_rho=30,
    n_theta=48,
    n_zeta=96,
)


@pytest.fixture(scope="module", params=[1, 2, 3, 5], ids=lambda n: f"nfp{n}")
def tilted_state(request):
    return _TiltedTorusState(nfp=request.param)


@pytest.fixture(scope="module")
def tilted_config(tilted_state):
    return MagneticConfiguration.from_gvec(tilted_state, grid=_GRID)


def _evaluate_on_config(config, positions):
    """Evaluate rho and Cartesian B of a configuration at Cartesian positions."""
    rho_interp = build_rho_interpolator(config)
    B_interp = build_magnetic_field_interpolator(config)

    def evaluate(position):
        r = jnp.hypot(position[0], position[1])
        phi = jnp.arctan2(position[1], position[0])
        phi_mapped, z_query, second_half = _map_to_fundamental_domain(
            phi, position[2], config.nfp
        )
        B_cyl = _apply_B_stellarator_symmetry(
            B_interp(r, phi_mapped, z_query), second_half
        )
        return rho_interp(r, phi_mapped, z_query), _cylindrical_to_cartesian_B(
            B_cyl, phi
        )

    rho, B = jax.vmap(evaluate)(jnp.asarray(positions))
    return np.asarray(rho), np.asarray(B)


def _random_samples(state, rho_values, n=200, seed=0):
    rng = np.random.default_rng(seed)
    rho = rng.choice(rho_values, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    zeta = rng.uniform(0, 2 * np.pi, n)
    return rho, state.position(rho, theta, zeta)


def test_fixture_cross_sections_are_tilted(tilted_state):
    """The cross-section zeta = 0 spans a range of phi, unlike for VMEC."""
    x = tilted_state.position(1.0, np.linspace(0, 2 * np.pi, 64), 0.0)
    phi = np.arctan2(x[:, 1], x[:, 0])
    assert np.ptp(phi) > np.radians(3)


def test_config_metadata(tilted_state, tilted_config):
    assert tilted_config.nfp == tilted_state.nfp
    assert tilted_config.is_stellarator_symmetric
    assert not tilted_config.is_axisymmetric
    assert tilted_config.rphiz.shape == (70, 24, 70, 3)
    np.testing.assert_allclose(
        tilted_config.rphiz[0, :, 0, 1], np.linspace(0, np.pi / tilted_state.nfp, 24)
    )


def test_rho_matches_geometry(tilted_state, tilted_config):
    rho_true, x = _random_samples(tilted_state, [0.2, 0.4, 0.6, 0.8])
    rho, _ = _evaluate_on_config(tilted_config, x)
    np.testing.assert_allclose(rho, rho_true, atol=0.01)


def test_magnetic_field_matches_analytic_field(tilted_state, tilted_config):
    _, x = _random_samples(tilted_state, [0.0, 0.05, 0.3, 0.6, 0.9])
    _, B = _evaluate_on_config(tilted_config, x)
    B_true = tilted_state.field(x)
    error = np.linalg.norm(B - B_true, axis=-1) / np.linalg.norm(B_true, axis=-1)
    assert np.max(error) < 1e-3


def test_magnetic_field_on_axis_is_not_zero(tilted_state, tilted_config):
    """The import does not rely on B at rho = 0, where the Jacobian vanishes."""
    x = tilted_state.position(np.zeros(8), np.zeros(8), np.linspace(0, 2 * np.pi, 8))
    _, B = _evaluate_on_config(tilted_config, x)
    np.testing.assert_allclose(
        np.linalg.norm(B, axis=-1),
        np.linalg.norm(tilted_state.field(x), axis=-1),
        rtol=1e-3,
    )


def test_rho_extends_beyond_lcfs(tilted_state, tilted_config):
    x = tilted_state.position(np.full(16, 1.1), np.linspace(0, 2 * np.pi, 16), 0.3)
    rho, _ = _evaluate_on_config(tilted_config, x)
    assert np.all(rho > 1.0)


@pytest.mark.parametrize("tilt", [0.0, 0.3])
def test_dvolume_drho_tilted_circular_torus(tilt):
    """Circular cross-sections tilted by a constant alpha.

    The Jacobian is rho a^2 (R0 + rho a cos(theta)) cos(alpha), so
    dV/drho = 4 pi^2 a^2 R0 rho cos(alpha).
    """
    state = _TiltedTorusState(tilt=tilt, tilt_modulation=0.0, shaping=0.0)
    rho = np.linspace(0.0, 1.0, 11)
    dv = np.asarray(dvolume_drho_gvec(state, rho))
    expected = 4 * np.pi**2 * state.a0**2 * state.R0 * rho * np.cos(tilt)
    np.testing.assert_allclose(dv, expected, rtol=1e-6, atol=1e-9)


def test_rejects_wrong_periodicity():
    with pytest.raises(NotImplementedError, match="periodicity"):
        cylindrical_grid_for_gvec_equilibrium(
            _TiltedTorusState(nfp=5, geometry_nfp=3),
            n_rho=5,
            n_theta=8,
            n_zeta=16,
            n_r=10,
            n_z=10,
            n_phi=4,
        )


def test_rejects_stellarator_asymmetric_equilibrium():
    with pytest.raises(NotImplementedError, match="stellarator symmetry"):
        cylindrical_grid_for_gvec_equilibrium(
            _TiltedTorusState(asymmetry=0.1),
            n_rho=5,
            n_theta=8,
            n_zeta=16,
            n_r=10,
            n_z=10,
            n_phi=4,
        )


def test_rejects_non_monotonic_phi():
    """phi(zeta) turning back, as for figure-8 geometries."""
    with pytest.raises(NotImplementedError, match="monotonically"):
        cylindrical_grid_for_gvec_equilibrium(
            _TiltedTorusState(wobble=0.4),
            n_rho=5,
            n_theta=8,
            n_zeta=32,
            n_r=10,
            n_z=10,
            n_phi=4,
        )


# ---------------------------------------------------------------------------
# Real GVEC equilibria of W7-X, compared against the VMEC equilibrium.
# W7X/ holds the hmap=1 state and wout; W7X/gframe-tilted/ (optional) holds the
# same equilibrium with tilted cross-sections, generated by
# scripts/make_tilted_gframe_w7x.py.
# ---------------------------------------------------------------------------

_W7X_DIR = Path(__file__).parents[2] / "W7X"


def _w7x_gvec_state(subdir=""):
    gvec = pytest.importorskip("gvec")
    state_dir = _W7X_DIR / subdir
    if not any(state_dir.glob("*_State_final.dat")):
        pytest.skip(f"No GVEC state in {state_dir}")
    return gvec.find_state(state_dir)


@pytest.fixture(scope="module")
def w7x_vmec_config():
    wout_path = next(_W7X_DIR.glob("wout_*.nc"), None)
    if wout_path is None:
        pytest.skip(f"No wout in {_W7X_DIR}")
    from vmecpp import VmecWOut

    wout = VmecWOut.from_wout_file(str(wout_path))
    return MagneticConfiguration.from_vmec_wout(wout)


@pytest.mark.parametrize("subdir", ["", "gframe-tilted"])
def test_w7x_gvec_matches_vmec(subdir, w7x_vmec_config):
    state = _w7x_gvec_state(subdir)
    config = MagneticConfiguration.from_gvec(state)
    assert config.nfp == 5

    rng = np.random.default_rng(1)
    rho_true = np.array([0.1, 0.3, 0.6, 0.9])
    theta = np.sort(rng.uniform(0, 2 * np.pi, 20))
    zeta = np.sort(rng.uniform(0, 2 * np.pi, 20))
    dataset = state.evaluate("pos", "B", rho=rho_true, theta=theta, zeta=zeta)
    x = np.moveaxis(dataset["pos"].values, 0, -1).reshape(-1, 3)
    absB_true = np.linalg.norm(dataset["B"].values, axis=-1).reshape(-1)
    rho_expected = np.repeat(rho_true, theta.size * zeta.size)

    rho_gvec, B_gvec = _evaluate_on_config(config, x)
    rho_vmec, B_vmec = _evaluate_on_config(w7x_vmec_config, x)
    absB_gvec = np.linalg.norm(B_gvec, axis=-1)

    assert np.max(np.abs(rho_gvec - rho_expected)) < 0.01
    assert np.max(np.abs(absB_gvec - absB_true) / absB_true) < 5e-3
    assert np.max(np.abs(rho_gvec - rho_vmec)) < 0.015
    np.testing.assert_allclose(absB_gvec, np.linalg.norm(B_vmec, axis=-1), rtol=5e-3)


def test_w7x_gvec_volume_is_independent_of_hmap():
    rho = np.linspace(0.0, 1.0, 50)
    dv = np.asarray(dvolume_drho_gvec(_w7x_gvec_state(), rho))
    dv_tilted = np.asarray(dvolume_drho_gvec(_w7x_gvec_state("gframe-tilted"), rho))
    np.testing.assert_allclose(dv_tilted, dv, rtol=1e-4, atol=1e-6)


def test_w7x_gvec_volume_matches_vmec(w7x_vmec_config):
    """dV/drho agrees with VMEC where VMEC resolves it (rho >= 0.15)."""
    rho = np.asarray(w7x_vmec_config.rho_1d)
    dv = np.asarray(dvolume_drho_gvec(_w7x_gvec_state(), rho))
    outer = rho >= 0.15
    np.testing.assert_allclose(
        dv[outer], np.asarray(w7x_vmec_config.dvolume_drho)[outer], rtol=1e-3
    )


def test_w7x_gvec_volume_is_linear_near_axis():
    rho = np.array([0.01, 0.02, 0.05])
    dv = np.asarray(dvolume_drho_gvec(_w7x_gvec_state(), rho))
    np.testing.assert_allclose(dv / rho, dv[0] / rho[0], rtol=1e-2)


def _w7x_ecrh_trace(config):
    """Trace a W7-X ECRH beam (140 GHz, O-mode) with B scaled to 2.52 T on axis."""
    rho = np.linspace(0, 1, 101)
    profiles = RadialProfiles(
        rho=jnp.asarray(rho),
        electron_density=jnp.asarray(0.75 * (1 - rho**2) ** 2),
        electron_temperature=jnp.asarray(5.0 * (1 - rho**2) ** 2 + 0.05),
    )
    r, phi, z = 6.50866, np.deg2rad(-6.56378), -0.38
    alpha, beta = np.deg2rad(15.7), np.deg2rad(19.7001)
    d_r = -np.cos(alpha) * np.cos(beta)
    d_phi = np.cos(alpha) * np.sin(beta)
    direction = np.array(
        [
            d_r * np.cos(phi) - d_phi * np.sin(phi),
            d_r * np.sin(phi) + d_phi * np.cos(phi),
            np.sin(alpha),
        ]
    )
    beam = Beam(
        position=jnp.array([r * np.cos(phi), r * np.sin(phi), z]),
        direction=jnp.asarray(direction / np.linalg.norm(direction)),
        frequency=jnp.array(140e9),
        mode="O",
        power=1e6,
    )
    return trace(
        config, profiles, beam, settings=TracerSettings(relative_tolerance=1e-6)
    )


@pytest.fixture(scope="module")
def w7x_field_scale():
    state = _w7x_gvec_state()
    B_axis = state.evaluate("B", rho=[0.0], theta=[0.0], zeta=[0.0])["B"].values
    return 2.52076 / float(np.linalg.norm(B_axis))


@pytest.mark.parametrize("subdir", ["", "gframe-tilted"])
def test_w7x_gvec_trace_matches_vmec(subdir, w7x_field_scale):
    from vmecpp import VmecWOut

    wout = VmecWOut.from_wout_file(str(next(_W7X_DIR.glob("wout_*.nc"))))
    vmec = _w7x_ecrh_trace(
        MagneticConfiguration.from_vmec_wout(wout, magnetic_field_scale=w7x_field_scale)
    )
    gvec_result = _w7x_ecrh_trace(
        MagneticConfiguration.from_gvec(
            _w7x_gvec_state(subdir), magnetic_field_scale=w7x_field_scale
        )
    )

    assert float(vmec.absorbed_power_fraction) > 0.5
    np.testing.assert_allclose(gvec_result.optical_depth, vmec.optical_depth, rtol=5e-3)
    np.testing.assert_allclose(
        gvec_result.absorbed_power_fraction, vmec.absorbed_power_fraction, rtol=2e-3
    )
    np.testing.assert_allclose(
        gvec_result.deposition_rho_mean, vmec.deposition_rho_mean, atol=3e-3
    )
    s_vmec = np.asarray(vmec.beam_profile.arc_length)
    s_gvec = np.asarray(gvec_result.beam_profile.arc_length)
    x_gvec = np.asarray(gvec_result.beam_profile.position)
    common = s_vmec <= s_gvec.max()
    x_gvec_at_vmec = np.stack(
        [np.interp(s_vmec[common], s_gvec, x_gvec[:, i]) for i in range(3)], axis=-1
    )
    deviation = np.linalg.norm(
        x_gvec_at_vmec - np.asarray(vmec.beam_profile.position)[common], axis=-1
    )
    assert deviation.max() < 1e-3
