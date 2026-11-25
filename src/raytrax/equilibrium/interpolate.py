r"""Utilities for interpolating flux surface quantities on a cylindrical grid.

Since ray tracing is computed in cartesian space (which can easily be transformed
to/from cylindrical coordinates), the effective minor radius $\rho$ and the magnetic
field $B$ need to be extrapolated from toroidal coordinates ($\rho$, $\theta$, $\phi$)
to cylindrical coordinates ($r$, $\phi$, $z$).
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv

import interpax
import jax
import jax.numpy as jnp
import jaxtyping as jt
import matplotlib.path
import numpy as np
import optimistix as optx
from beartype import beartype as typechecker
from scipy.interpolate import griddata

from raytrax.equilibrium.protocol import WoutLike
from raytrax.types import RadialProfiles, SafetensorsMixin

from .fourier import dvolume_drho as compute_dvolume_drho
from .fourier import evaluate_rphiz_on_toroidal_grid, extrapolate_coefficients


@dataclass
class CylindricalGridResolution:
    """Output cylindrical grid resolution, shared by all equilibrium importers.

    Defines the $(R, \\phi, Z)$ cylindrical grid on which the magnetic
    configuration is stored and the ray tracer interpolates. For axisymmetric
    (tokamak) configurations ``n_phi`` is ignored.

    Attributes:
        n_r: Number of grid points along the major radius $R$.
        n_z: Number of grid points along the vertical coordinate $Z$.
        n_phi: Number of toroidal planes $\\phi$. Ignored for axisymmetric
            configurations.
        n_rho_profile: Number of radial points for the 1-D $dV/d\\rho$ profile.
    """

    n_r: int = 95
    n_z: int = 105
    n_phi: int = 50
    n_rho_profile: int = 200


@dataclass
class VmecGridResolution:
    """Grid resolution for VMEC-based equilibrium imports.

    Combines the shared `CylindricalGridResolution` output grid with the
    VMEC-specific sampling that sets its extent.

    The VMEC import solves for the flux coordinates of each cylindrical grid node
    with Newton's method and evaluates the Fourier series there, so the values at
    the nodes do not depend on ``n_rho`` and ``n_theta``. These only sample the
    flux surfaces up to $\\rho = 1.2$ to find the $R$ and $Z$ extent of the grid.

    Attributes:
        cylindrical: Output cylindrical grid shared with all other importers.
        n_rho: Number of flux surfaces sampled to find the grid extent.
        n_theta: Number of poloidal points per flux surface sampled to find the
            grid extent.
    """

    cylindrical: CylindricalGridResolution = dataclass_field(
        default_factory=CylindricalGridResolution
    )
    n_rho: int = 40
    n_theta: int = 45


@jax.tree_util.register_dataclass
@dataclass
class MagneticConfiguration(SafetensorsMixin):
    r"""Magnetic configuration and geometry on a cylindrical grid.

    Contains the magnetic field $\boldsymbol{B}$ and normalized effective radius $\rho$ on a
    3D cylindrical grid ($R$, $\phi$, $Z$), along with volume information for
    computing deposition profiles.
    """

    rphiz: jt.Float[jax.Array, "npoints 3"]
    r"""The ($R$, $\phi$, $Z$) coordinates of the points on the interpolation grid.

    Must be a tensor grid that is uniformly spaced along each axis, with at least 4
    nodes per axis: the field and $\rho$ are interpolated using only the first and
    last node of each axis. For a stellarator the $\phi$ axis spans the fundamental
    domain $[0, \pi/n_{fp}]$ and the $Z$ grid must be symmetric about $Z = 0$.
    """

    magnetic_field: jt.Float[jax.Array, "npoints 3"]
    r"""The magnetic field $(B_R, B_\phi, B_Z)$ in cylindrical components at each grid point."""

    rho: jt.Float[jax.Array, " npoints"]
    """The normalized effective minor radius at each point on the interpolation grid."""

    nfp: int = dataclass_field(metadata={"static": True})
    """Number of field periods (toroidal periodicity)."""

    is_stellarator_symmetric: bool = dataclass_field(metadata={"static": True})
    """Whether the configuration has stellarator symmetry."""

    rho_1d: jt.Float[jax.Array, " nrho_1d"]
    """1D radial grid for the volume derivative and the power deposition profile.

    Must be uniformly spaced: the deposited power is binned into bins of equal
    width centred on these points.
    """

    dvolume_drho: jt.Float[jax.Array, " nrho_1d"]
    r"""Volume derivative $dV/d\rho$ on the 1D radial grid."""

    is_axisymmetric: bool = dataclass_field(default=False, metadata={"static": True})
    """Whether the configuration is axisymmetric (tokamak) or 3D (stellarator)."""

    @classmethod
    def from_vmec_wout(
        cls,
        equilibrium: WoutLike,
        magnetic_field_scale: float = 1.0,
        grid: VmecGridResolution | None = None,
    ) -> "MagneticConfiguration":
        """Create a MagneticConfiguration from a VMEC++ equilibrium.

        Optionally, apply a uniform scale factor to the magnetic field magnitude
        with respect to the equilibrium data.

        Args:
            equilibrium: an MHD equilibrium compatible with `vmecpp.VmecWOut`
            magnetic_field_scale: Factor to multiply all magnetic field values by.
            grid: Grid resolution settings. Defaults to
                :class:`VmecGridResolution` with its default values.

        Returns:
            A MagneticConfiguration object containing interpolation data.
        """
        if grid is None:
            grid = VmecGridResolution()

        interpolated_array = cylindrical_grid_for_equilibrium(
            equilibrium=equilibrium,
            n_rho=grid.n_rho,
            n_theta=grid.n_theta,
            n_phi=grid.cylindrical.n_phi,
            n_r=grid.cylindrical.n_r,
            n_z=grid.cylindrical.n_z,
        )
        rphiz = interpolated_array[..., :3]
        rho = interpolated_array[..., 3]
        magnetic_field = interpolated_array[..., 4:] * magnetic_field_scale

        # Compute volume derivative on 1D radial grid
        rho_1d = jnp.linspace(0, 1, grid.cylindrical.n_rho_profile)
        dv_drho = compute_dvolume_drho(equilibrium, rho_1d)

        return cls(
            rphiz=rphiz,
            magnetic_field=magnetic_field,
            rho=rho,
            nfp=equilibrium.nfp,
            is_stellarator_symmetric=not equilibrium.lasym,
            rho_1d=rho_1d,
            dvolume_drho=dv_drho,
        )

    def to_pyvista_grid(self) -> "pv.StructuredGrid":
        """Convert to a PyVista StructuredGrid in Cartesian coordinates.

        Point arrays written:

        - ``rho``: normalised effective minor radius (NaN outside LCFS)
        - ``B``: magnetic field vector :math:`(B_R, B_\\phi, B_Z)` in T
        - ``absB``: magnetic field magnitude :math:`|\\boldsymbol{B}|` in T

        Requires the ``pyvista`` package.

        Returns:
            A :class:`pyvista.StructuredGrid` that can be visualised or saved
            directly via ``.save("output.vts")``.
        """
        import pyvista as pv

        rphiz = np.array(self.rphiz)
        R, phi, Z = rphiz[..., 0], rphiz[..., 1], rphiz[..., 2]

        B = np.array(self.magnetic_field)
        rho = np.array(self.rho)

        grid = pv.StructuredGrid(R * np.cos(phi), R * np.sin(phi), Z)
        grid["rho"] = rho.flatten(order="F")  # NaN outside LCFS
        grid["B"] = B.reshape(-1, 3, order="F")
        grid["absB"] = np.linalg.norm(B, axis=-1).flatten(order="F")

        return grid


@jt.jaxtyped(typechecker=typechecker)
def interpolate_toroidal_to_cylindrical_grid(
    rphiz_toroidal: jt.Float[jax.Array, "n_rho n_theta n_phi rphiz=3"],
    rz_cylindrical: jt.Float[jax.Array, "n_r n_z rz=2"],
    value_toroidal: jt.Float[jax.Array, "n_rho n_theta n_phi n_values"],
) -> jt.Float[jax.Array, "n_r n_phi n_z n_values"]:
    """Interpolate toroidal coordinates to a cylindrical grid.

    Args:
    - rphiz_toroidal: An array of cylindrical coordinates (the last dimension
        are the three coordinates R, phi, Z) on the toroidal grid.
    - rz_cylindrical: The cylindrical coordinates (R, Z) on the cylindrical grid.
        Note that phi does not need to be provided because we use the same phi
        grid as for the toroidal grid, so no interpolation in phi is needed.
    - value_toroidal: The values of the quantities to be interpolated on the
        toroidal grid nodes.

    Returns:
    The values of the quantities interpolated to the cylindrical grid as a JAX
    array.
    """
    n_r, n_z, _ = rz_cylindrical.shape
    n_values = value_toroidal.shape[-1]
    values = []
    phis = rphiz_toroidal[0, 0, :, 1]
    for i_phi, _ in enumerate(phis):
        # this uses plain numpy, because griddata is anyway not JAX compatible
        value = griddata(
            np.array(rphiz_toroidal[:, :, i_phi, ::2]).reshape(-1, 2),
            np.array(value_toroidal[:, :, i_phi]).reshape(-1, n_values),
            np.array(rz_cylindrical).reshape(-1, 2),
            method="linear",
        ).reshape(n_r, n_z, n_values)
        values.append(value)
    return jnp.stack(values, axis=1)


_N_SEED_RHO = 10
_N_SEED_THETA = 10
"""Coarse (rho, theta) grid per plane whose nearest point seeds each node's Newton solve."""

# Best-so-far: a node that does not converge keeps its smallest-residual iterate.
_NEWTON: optx.AbstractRootFinder = optx.BestSoFarRootFinder(
    optx.Newton(rtol=1e-12, atol=1e-12)
)


@jt.jaxtyped(typechecker=typechecker)
def cylindrical_grid_for_equilibrium(
    equilibrium: WoutLike, n_rho: int, n_theta: int, n_phi: int, n_r: int, n_z: int
) -> jt.Float[jax.Array, "n_r n_phi n_z rphizrhoBxyz=7"]:
    """Compute the effective minor radius and the magnetic field on a cylindrical grid.

    Args:
        equilibrium: The MHD equilibrium.
        n_rho: Number of radial points.
        n_theta: Number of poloidal points.
        n_phi: Number of toroidal points (equal in both grids!).
        n_r: Number of radial points in the cylindrical grid.
        n_z: Number of vertical points in the cylindrical grid.

    Returns:
        A JAX array of values on the cylindrical grid with shape (n_r, n_phi, n_z, 7).
        The last dimension is a vector with the following components:
        (R, phi, Z, rho, B_R, B_phi, B_Z) where the magnetic field components
        are in cylindrical coordinates.
    """
    if equilibrium.lasym:
        raise NotImplementedError(
            "Non stellarator symmetric equilibria are not supported yet."
        )
    # Each node's flux coordinates are found by Newton on (R, Z)(rho, theta) in the
    # node's phi plane, so rho and B are exact at the nodes rather than interpolated
    # across a triangulation of a toroidal grid.
    phis = jnp.linspace(0, jnp.pi / equilibrium.nfp, n_phi)
    rho_max = 1.2  # extend beyond the LCMS to allow extrapolation
    rho_theta_phi = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, rho_max, n_rho),
            jnp.linspace(0, 2 * jnp.pi, n_theta),
            phis,
            indexing="ij",
        ),
        axis=-1,
    )
    rphiz = evaluate_rphiz_on_toroidal_grid(equilibrium, rho_theta_phi)
    r_grid = jnp.linspace(rphiz[..., 0].min(), rphiz[..., 0].max(), n_r)
    # Symmetric about Z = 0, so that a point and its stellarator-symmetric mirror
    # (phi, Z) -> (-phi, -Z) interpolate from mirrored nodes and agree; otherwise rho
    # and B jump on the planes phi = 0 and phi = pi / nfp.
    z_max = jnp.abs(rphiz[..., 2]).max()
    z_grid = jnp.linspace(-z_max, z_max, n_z)
    rphiz_cylindrical = jnp.stack(
        jnp.meshgrid(r_grid, phis, z_grid, indexing="ij"), axis=-1
    )
    ns = equilibrium.ns
    ds = 1 / (ns - 1)
    xn = jnp.asarray(equilibrium.xn)[:, None]
    # R, Z and their phi derivatives: d/dphi exp(-i n phi) = -i n exp(-i n phi).
    # Newton only needs the first two, so it gets its own, cheaper spline.
    geometry = _ThetaHarmonics(
        knots=jnp.sqrt(jnp.linspace(0, 1, ns)),
        coefficients=jnp.stack(
            [
                jnp.asarray(equilibrium.rmnc),
                jnp.asarray(equilibrium.zmns),
                -1j * xn * jnp.asarray(equilibrium.rmnc),
                -1j * xn * jnp.asarray(equilibrium.zmns),
            ]
        ),
        xm=jnp.asarray(equilibrium.xm),
        xn=jnp.asarray(equilibrium.xn),
        m_max=int(np.max(equilibrium.xm)),
    )
    field = _ThetaHarmonics(
        knots=jnp.sqrt(jnp.linspace(ds / 2, 1 - ds / 2, ns - 1)),
        coefficients=jnp.stack(
            [
                jnp.asarray(equilibrium.bsupumnc)[:, 1:],
                jnp.asarray(equilibrium.bsupvmnc)[:, 1:],
            ]
        ),
        xm=jnp.asarray(equilibrium.xm_nyq),
        xn=jnp.asarray(equilibrium.xn_nyq),
        m_max=int(np.max(equilibrium.xm_nyq)),
    )
    seed_rho_theta_phi = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, rho_max, _N_SEED_RHO),
            jnp.linspace(0, 2 * jnp.pi, _N_SEED_THETA),
            phis,
            indexing="ij",
        ),
        axis=-1,
    )
    seed_rphiz = evaluate_rphiz_on_toroidal_grid(equilibrium, seed_rho_theta_phi)
    nodes = np.moveaxis(np.asarray(rphiz_cylindrical), 1, 0).reshape(n_phi, -1, 3)
    # Newton only runs on nodes inside a finely sampled ring a bit beyond rho_max;
    # the rest of the box is outside the domain. Each plane is padded to the same
    # node count so the solve compiles once.
    ring = np.asarray(
        evaluate_rphiz_on_toroidal_grid(
            equilibrium,
            jnp.stack(
                jnp.meshgrid(
                    jnp.array([1.05 * rho_max]),
                    jnp.linspace(0, 2 * jnp.pi, 256),
                    phis,
                    indexing="ij",
                ),
                axis=-1,
            ),
        )[0]
    )
    solve = np.stack(
        [
            matplotlib.path.Path(ring[:, p, ::2]).contains_points(nodes[p, :, ::2])
            for p in range(n_phi)
        ]
    )
    selected = np.argsort(~solve, axis=1, kind="stable")[:, : solve.sum(1).max()]
    # Samples off the axis seed Newton: (rho, theta) is singular there.
    solved = _solve_planes(
        geometry,
        field,
        rho_max,
        _NEWTON,
        (
            phis,
            jnp.asarray(np.take_along_axis(nodes, selected[..., None], axis=1)),
            jnp.moveaxis(seed_rho_theta_phi[1:], 2, 0).reshape(n_phi, -1, 3),
            jnp.moveaxis(seed_rphiz[1:], 2, 0).reshape(n_phi, -1, 3),
        ),
    )
    values = np.full((n_phi, n_r * n_z, 4), np.nan)
    planes = np.arange(n_phi)[:, None]
    values[planes, selected] = np.where(
        np.take_along_axis(solve, selected, axis=1)[..., None], solved, np.nan
    )
    values_rphiz = np.moveaxis(values.reshape(n_phi, n_r, n_z, 4), 0, 1)
    return jnp.concatenate([rphiz_cylindrical, jnp.asarray(values_rphiz)], axis=-1)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["knots", "coefficients", "xm", "xn"],
    meta_fields=["m_max"],
)
@dataclass(frozen=True)
class _ThetaHarmonics:
    """Fourier series sum_mn c_mn(rho) exp(i (m theta - n phi)) per quantity."""

    knots: jax.Array
    coefficients: jax.Array
    xm: jax.Array
    xn: jax.Array
    m_max: int

    def in_plane(self, phi: jax.Array) -> interpax.Interpolator1D:
        """Spline in rho of a_m with series = sum_m a_m(rho) exp(i m theta) at phi."""
        phased = self.coefficients * jnp.exp(-1j * self.xn * phi)[:, None]
        a_m = jax.vmap(
            partial(
                jax.ops.segment_sum,
                segment_ids=self.xm,
                num_segments=self.m_max + 1,
            )
        )(phased)
        return interpax.Interpolator1D(
            self.knots, jnp.moveaxis(a_m, -1, 0), extrap=True
        )


@partial(jax.jit, static_argnames=("rho_max", "root_finder"))
def _solve_planes(
    geometry: _ThetaHarmonics,
    field: _ThetaHarmonics,
    rho_max: float,
    root_finder: optx.AbstractRootFinder,
    planes: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> jax.Array:
    # Sequential over planes: the nearest-seed search is nodes x seeds per plane.
    return jax.lax.map(
        partial(_plane_values, geometry, field, rho_max, root_finder), planes
    )


def _plane_values(
    geometry: _ThetaHarmonics,
    field: _ThetaHarmonics,
    rho_max: float,
    root_finder: optx.AbstractRootFinder,
    plane: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> jax.Array:
    """(rho, B_R, B_phi, B_Z) at the nodes of one phi plane; NaN beyond rho_max."""
    phi, nodes, sample_rho_theta_phi, sample_rphiz = plane
    distance = jnp.linalg.norm(
        nodes[:, None, ::2] - sample_rphiz[None, :, ::2], axis=-1
    )
    guess = sample_rho_theta_phi[jnp.argmin(distance, axis=1)]
    geometry_in_plane = geometry.in_plane(phi)
    rz = interpax.Interpolator1D(
        geometry_in_plane.x, geometry_in_plane.f[:, :2], extrap=True
    )
    return jax.vmap(
        partial(
            _node_values,
            rz,
            geometry_in_plane,
            field.in_plane(phi),
            rho_max,
            root_finder,
        )
    )(nodes[:, ::2], guess[:, :2])


def _node_values(
    rz_harmonics: interpax.Interpolator1D,
    geometry: interpax.Interpolator1D,
    field: interpax.Interpolator1D,
    rho_max: float,
    root_finder: optx.AbstractRootFinder,
    rz: jax.Array,
    guess: jax.Array,
) -> jax.Array:
    # Solve for rho (cos theta, sin theta), which unlike (rho, theta) is a smooth
    # chart through the axis, so a Newton step across it needs no special case.
    solution: optx.Solution = optx.root_find(
        _rz_residual,
        root_finder,
        guess[0] * jnp.stack([jnp.cos(guess[1]), jnp.sin(guess[1])]),
        args=(rz_harmonics, rz),
        max_steps=10,
        throw=False,
    )
    rho = jnp.linalg.norm(solution.value)
    theta = jnp.arctan2(solution.value[1], solution.value[0])
    # B is held at its LCMS value beyond rho = 1.
    series, series_theta = _sum_harmonics(geometry, jnp.minimum(rho, 1.0), theta)
    b_theta, b_phi = _sum_harmonics(field, jnp.minimum(rho, 1.0), theta)[0].real
    b_cylindrical = jnp.stack(
        [
            b_theta * series_theta[0].real + b_phi * series[2].real,
            b_phi * series[0].real,
            b_theta * series_theta[1].imag + b_phi * series[3].imag,
        ]
    )
    # An unconverged node keeps its best iterate: NaN would read downstream as
    # vacuum (rho = 1.1, B = 0), a hole in the plasma.
    return jnp.where(rho <= rho_max, jnp.append(rho, b_cylindrical), jnp.nan)


def _rz_residual(
    y: jax.Array, args: tuple[interpax.Interpolator1D, jax.Array]
) -> jax.Array:
    rz_harmonics, rz = args
    a_m = extrapolate_coefficients(rz_harmonics, jnp.linalg.norm(y)[None])[0]
    series = a_m @ jnp.exp(1j * jnp.arange(a_m.shape[-1]) * jnp.arctan2(y[1], y[0]))
    return jnp.stack([series[0].real, series[1].imag]) - rz


def _sum_harmonics(
    harmonics: interpax.Interpolator1D, rho: jax.Array, theta: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """sum_m a_m exp(i m theta) and its theta derivative, per quantity."""
    a_m = extrapolate_coefficients(harmonics, rho[None])[0]
    m = jnp.arange(a_m.shape[-1])
    phase = jnp.exp(1j * m * theta)
    return a_m @ phase, a_m @ (1j * m * phase)


def _catmull_rom_weights(u: jax.Array) -> jax.Array:
    """Weights of the nodes -1, 0, 1, 2 at fractional position u in [0, 1] of a cell."""
    u2, u3 = u * u, u * u * u
    return jnp.stack(
        [
            (-u3 + 2 * u2 - u) / 2,
            (3 * u3 - 5 * u2 + 2) / 2,
            (-3 * u3 + 4 * u2 + u) / 2,
            (u3 - u2) / 2,
        ]
    )


def _block_weights(w: jax.Array, shift: jax.Array, clamp: bool) -> jax.Array:
    """Weights of a 4-node block that starts `shift` nodes after the stencil.

    In the first cell of an axis (shift = -1) the stencil node before the grid is
    missing, in the last cell (shift = +1) the one after it. With `clamp` the missing
    node takes the value of the edge node, otherwise its weight is dropped here and the
    caller supplies the node.
    """
    if clamp:
        left = jnp.stack([w[0] + w[1], w[2], w[3], jnp.zeros_like(w[0])])
        right = jnp.stack([jnp.zeros_like(w[0]), w[0], w[1], w[2] + w[3]])
    else:
        left = jnp.stack([w[1], w[2], w[3], jnp.zeros_like(w[0])])
        right = jnp.stack([jnp.zeros_like(w[0]), w[0], w[1], w[2]])
    return jnp.where(shift < 0, left, jnp.where(shift > 0, right, w))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GridInterpolator:
    r"""Catmull-Rom interpolation on a uniform grid, with continuous first derivatives.

    The ray equations contain the gradients of $\boldsymbol{B}$ and $\rho$, so the
    derivatives of a trajectory with respect to its inputs need their second
    derivatives. Interpolants with continuous first derivatives keep these finite,
    which makes gradients through the ray tracer correct. Each value depends on the
    4 nearest nodes along each axis. At the outer nodes of the $R$ and $Z$ axes the
    missing neighbour takes the value of the edge node.

    The first `lower.size` axes of `f` are the grid axes, any further axes are
    vector components. Grid values that are NaN and query points outside the
    grid both evaluate to `fill`. Construction does no work, so building one
    inside `jax.jit` on every call is free and gradients reach `f` directly.
    """

    lower: jt.Float[jax.Array, " ndim"]
    """Coordinates of the first grid node along each axis."""

    upper: jt.Float[jax.Array, " ndim"]
    """Coordinates of the last grid node along each axis."""

    f: jt.Float[jax.Array, "..."]
    """Values on the grid, at least 4 nodes along each grid axis."""

    mirror_sign: jt.Float[jax.Array, "..."]
    r"""Sign of each component at the stellarator-mirrored point $(-\phi, -Z)$."""

    fill: float = dataclass_field(metadata={"static": True})
    """Value for NaN grid nodes and for points outside the grid."""

    square: bool = dataclass_field(default=False, metadata={"static": True})
    r"""Interpolate $f^2$ and return its square root.

    $\rho$ has a cone at the magnetic axis, which a cubic resolves poorly, but
    $\rho^2$ is smooth there.
    """

    stellarator_mirror: bool = dataclass_field(default=False, metadata={"static": True})
    r"""The axes are $(R, \phi, Z)$ on the fundamental domain $[0, \pi/n_{fp}]$.

    Beyond $\phi = 0$ and $\phi = \pi/n_{fp}$ the stencil reads the
    stellarator-mirrored plane, with $Z$ reversed and components multiplied by
    `mirror_sign`, so the first derivatives are continuous across these planes. The
    $Z$ grid must be symmetric about $Z = 0$.
    """

    def _nodes(self, block: jax.Array) -> jax.Array:
        block = jnp.nan_to_num(block, nan=self.fill)
        return block * block if self.square else block

    # Jitted so that a call outside jit compiles once; inside jit it is inlined.
    @jax.jit
    def __call__(self, *x: jt.Float[jax.Array, ""]) -> jax.Array:
        """Interpolate at the point with coordinates `x`, one per grid axis."""
        ndim = len(x)
        shape = self.f.shape[:ndim]
        components = self.f.shape[ndim:]
        n = jnp.array(shape)
        t = (jnp.stack(x) - self.lower) / (self.upper - self.lower) * (n - 1)
        index = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, n - 2)
        w = _catmull_rom_weights(t - index)
        # The block always lies inside the grid; in the first and last cell of an
        # axis it is shifted by one node against the stencil.
        start = jnp.clip(index - 1, 0, n - 4)
        shift = index - 1 - start
        mirrored_axis = 1 if self.stellarator_mirror else None
        weights = [
            _block_weights(w[:, i], shift[i], clamp=i != mirrored_axis)
            for i in range(ndim)
        ]
        block = jax.lax.dynamic_slice(
            self.f, (*start, *[0] * len(components)), (4,) * ndim + components
        )
        value = self._nodes(block)
        for weight in weights:
            value = jnp.tensordot(weight, value, axes=(0, 0))
        if self.stellarator_mirror:
            # Only the first and last phi cell need the mirrored plane. Reading it
            # in every cell would make tracing about a third slower.
            value = value + jax.lax.cond(
                shift[1] != 0,
                self._mirror_plane,
                lambda *_: jnp.zeros(components),
                start,
                shift[1],
                w[:, 1],
                weights[0],
                weights[2],
            )
        if self.square:
            value = jnp.sqrt(jnp.maximum(value, 1e-12))
        inside = jnp.all((t >= 0) & (t <= n - 1))
        return jnp.where(inside, value, self.fill)

    def _mirror_plane(
        self,
        start: jax.Array,
        shift_phi: jax.Array,
        w_phi: jax.Array,
        weights_r: jax.Array,
        weights_z: jax.Array,
    ) -> jax.Array:
        """Contribution of the phi plane beyond the edge of the fundamental domain.

        That plane is the stellarator mirror of the plane next to the edge, read
        with Z reversed. `w_phi` are the stencil weights along phi, of which the
        missing plane has the first (at phi = 0) or the last (at phi = pi/nfp).
        """
        n_phi, n_z = self.f.shape[1], self.f.shape[2]
        components = self.f.shape[3:]
        plane = jax.lax.dynamic_slice(
            self.f,
            (start[0], jnp.where(shift_phi < 0, 1, n_phi - 2), n_z - 4 - start[2])
            + (0,) * len(components),
            (4, 1, 4) + components,
        )[:, 0]
        plane = self._nodes(jnp.flip(plane, 1) * self.mirror_sign)
        weight = jnp.where(shift_phi < 0, w_phi[0], w_phi[3])
        return weight * jnp.tensordot(
            weights_z, jnp.tensordot(weights_r, plane, axes=(0, 0)), axes=(0, 0)
        )


def build_magnetic_field_interpolator(
    magnetic_configuration: MagneticConfiguration,
) -> GridInterpolator:
    """Build a magnetic field interpolator from the magnetic configuration.

    For stellarators, interpolates (B_R, B_phi, B_Z) in cylindrical coordinates
    (R, phi, Z) on the fundamental domain [0, pi/nfp]. For axisymmetric
    equilibria, interpolates (B_R, B_phi, B_Z) as a function of (R, Z).
    """
    # Stellarator symmetry: B_R(-phi, -Z) = -B_R(phi, Z), B_phi and B_Z are even.
    return _build_grid_interpolator(
        magnetic_configuration,
        magnetic_configuration.magnetic_field,
        fill=0.0,
        mirror_sign=jnp.array([-1.0, 1.0, 1.0]),
    )


def build_rho_interpolator(
    magnetic_configuration: MagneticConfiguration,
) -> GridInterpolator:
    """Build rho interpolator for the given magnetic configuration.

    For stellarators, interpolates rho in cylindrical coordinates (R, phi, Z) on
    the fundamental domain [0, pi/nfp]. For axisymmetric equilibria, interpolates
    rho as a function of (R, Z). The interpolant is that of rho^2, which is smooth
    through the magnetic axis.
    """
    return _build_grid_interpolator(
        magnetic_configuration,
        magnetic_configuration.rho,
        fill=1.1,
        mirror_sign=jnp.array(1.0),
        square=True,
    )


def _build_grid_interpolator(
    magnetic_configuration: MagneticConfiguration,
    f: jax.Array,
    fill: float,
    mirror_sign: jax.Array,
    square: bool = False,
) -> GridInterpolator:
    rphiz = magnetic_configuration.rphiz
    if magnetic_configuration.is_axisymmetric:
        return GridInterpolator(
            lower=rphiz[0, 0, 0, ::2],
            upper=rphiz[-1, 0, -1, ::2],
            f=f[:, 0],
            mirror_sign=mirror_sign,
            fill=fill,
            square=square,
        )
    if not magnetic_configuration.is_stellarator_symmetric:
        raise NotImplementedError(
            "Non stellarator-symmetric equilibria not yet supported"
        )
    return GridInterpolator(
        lower=rphiz[0, 0, 0],
        upper=rphiz[-1, -1, -1],
        f=f,
        mirror_sign=mirror_sign,
        fill=fill,
        square=square,
        stellarator_mirror=True,
    )


_VACUUM_RAMP_WIDTH = 0.02  # width in rho of the density ramp beyond the profile grid
_VACUUM_RAMP_POINTS = 9  # number of knots resolving the ramp


def build_electron_density_profile_interpolator(
    radial_profiles: RadialProfiles,
) -> interpax.Interpolator1D:
    r"""Build electron density profile interpolator.

    Beyond the last grid point $\rho_\mathrm{edge}$ (normally the LCFS), the
    density decreases from $n_e(\rho_\mathrm{edge})$ to zero over a thin layer of
    width $\Delta\rho = 0.02$, so that rays are refracted at the plasma–vacuum
    interface for profiles with a finite edge density. The ramp is sampled from a
    cubic Hermite polynomial that starts with the slope of the last profile
    interval and ends with zero slope. The slope is clipped so that the ramp only
    decreases. Profile and ramp are interpolated with a shape-preserving cubic
    spline: the density has a continuous gradient and does not overshoot the data
    between knots, so it cannot turn negative at the end of the ramp. Profiles need
    not be monotonic; at a local maximum or minimum of the data the gradient is
    zero at that knot.

    Args:
        radial_profiles: The radial profiles.

    Returns:
        An interpax.Interpolator1D that maps rho to electron density.
    """
    rho = radial_profiles.rho
    ne = radial_profiles.electron_density
    s = jnp.linspace(0.0, 1.0, _VACUUM_RAMP_POINTS + 1)[1:]
    rho_ramp = rho[-1] + _VACUUM_RAMP_WIDTH * s
    edge_slope = (ne[-1] - ne[-2]) / (rho[-1] - rho[-2])
    # Slope in units of s; between -3 ne_edge and 0 the ramp decreases monotonically.
    m = jnp.clip(edge_slope * _VACUUM_RAMP_WIDTH, -3.0 * ne[-1], 0.0)
    ne_ramp = ne[-1] * (1.0 - s**2 * (3.0 - 2.0 * s)) + m * s * (1.0 - s) ** 2
    dne_ramp = (
        ne[-1] * 6.0 * s * (s - 1.0) + m * (1.0 - s) * (1.0 - 3.0 * s)
    ) / _VACUUM_RAMP_WIDTH

    # Knot derivatives: shape-preserving estimates inside the profile, the exact ramp
    # derivatives beyond it, and the ramp's initial slope at the edge, so that the
    # spline joins profile and ramp without bending the last profile interval.
    dne = interpax.approx_df(rho, ne, "monotonic-0", 0)
    dne = dne.at[-1].set(m / _VACUUM_RAMP_WIDTH)
    return interpax.Interpolator1D(
        x=jnp.concatenate([rho, rho_ramp]),
        f=jnp.concatenate([ne, ne_ramp]),
        method="monotonic-0",
        extrap=0.0,
        fx=jnp.concatenate([dne, dne_ramp]),
    )


def build_electron_temperature_profile_interpolator(
    radial_profiles: RadialProfiles,
) -> interpax.Interpolator1D:
    """Build electron temperature profile interpolator.

    Interpolates with a shape-preserving cubic spline, like the density: the
    temperature has a continuous gradient and does not overshoot the data between
    knots.

    Args:
        radial_profiles: The radial profiles.

    Returns:
        An interpax.Interpolator1D that maps rho to electron temperature.
    """
    return interpax.Interpolator1D(
        x=radial_profiles.rho,
        f=radial_profiles.electron_temperature,
        method="monotonic-0",
        extrap=0.0,
    )


def build_radial_interpolators(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
) -> tuple[
    GridInterpolator,
    interpax.Interpolator1D,
    interpax.Interpolator1D,
]:
    """Build radial interpolators for the given magnetic configuration and radial profiles.

    This is a convenience function that combines the individual interpolator builders.
    For cleaner code, consider using the individual functions directly.

    Args:
        magnetic_configuration: The magnetic configuration.
        radial_profiles: The radial profiles.

    Returns:
        A tuple of three functions:
        - rho_interpolator: maps position to radial coordinate (rho)
        - electron_density_profile_interpolator: maps rho to electron density
        - electron_temperature_profile_interpolator: maps rho to electron temperature
    """
    rho_interpolator = build_rho_interpolator(magnetic_configuration)
    ne_profile_interpolator = build_electron_density_profile_interpolator(
        radial_profiles
    )
    Te_profile_interpolator = build_electron_temperature_profile_interpolator(
        radial_profiles
    )

    return rho_interpolator, ne_profile_interpolator, Te_profile_interpolator
