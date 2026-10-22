"""Re-solve a GVEC equilibrium with cross-sections tilted against the phi = const planes.

Builds a G-frame (``which_hmap=21``) whose origin is the magnetic axis of the input
equilibrium and whose cutting planes are perpendicular to that axis, cuts the input
LCFS with these planes, and re-solves with the input's profiles and flux. The result
describes the same equilibrium in a frame where the cross-sections zeta = const are
not planes phi = const, which ``tests/equilibrium/gvec_test.py`` uses to test the
GVEC import.

For W7-X the planes are tilted by up to ~18 degrees. The solve takes ~30 min and
peaks at ~7 GB of memory, so run it with a memory limit, e.g.

    systemd-run --user --scope -p MemoryMax=12G \
        python scripts/make_tilted_gframe_w7x.py W7X W7X/gframe-tilted

Requires gvec >= 1.5.
"""

import argparse
import os
from pathlib import Path

import gvec
import numpy as np
from gvec import fourier
from gvec.gframe import cut_surf, minimal_modes, write_Gframe_ncfile
from gvec.util import read_parameters

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("source", type=Path, help="directory with the input GVEC run")
parser.add_argument("output", type=Path, help="directory for the new GVEC run")
parser.add_argument("--n-zeta", type=int, default=61, help="frame points per period")
parser.add_argument("--n-theta", type=int, default=61, help="boundary points in theta")
parser.add_argument("--tolerance", type=float, default=1e-6, help="boundary modes")
args = parser.parse_args()

source = args.source.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
os.chdir(output)

state = gvec.find_state(source)
nfp = state.nfp
nz, nt = args.n_zeta, args.n_theta
zeta = np.linspace(0, 2 * np.pi, nz * nfp, endpoint=False)
theta = np.linspace(0, 2 * np.pi, nt, endpoint=False)

# Frame: origin on the magnetic axis, plane normal along the axis tangent T,
# N = e_R projected into the plane, B = T x N.
axis = state.evaluate("pos", rho=0.0, theta=0.0, zeta=zeta)["pos"].values[:, 0, 0, :]
X0 = np.moveaxis(axis, 0, -1)
k = np.fft.fftfreq(len(zeta), d=1.0 / len(zeta))
T = np.real(np.fft.ifft(1j * k[:, None] * np.fft.fft(X0, axis=0), axis=0))
T /= np.linalg.norm(T, axis=1, keepdims=True)
e_R = np.stack([X0[:, 0], X0[:, 1], np.zeros(len(zeta))], axis=1)
e_R /= np.linalg.norm(e_R, axis=1, keepdims=True)
N = e_R - np.sum(e_R * T, axis=1, keepdims=True) * T
N /= np.linalg.norm(N, axis=1, keepdims=True)
B = np.cross(T, N)
e_phi = np.cross([0.0, 0.0, 1.0], e_R)
tilt = np.degrees(np.arccos(np.abs(np.sum(e_phi * T, axis=1))))
print(f"Tilt against the phi = const planes: up to {tilt.max():.1f} deg")

lcfs = state.evaluate("pos", rho=1.0, theta=theta, zeta=zeta)["pos"].values[:, 0]
x1, x2 = cut_surf(np.moveaxis(lcfs, 0, -1).transpose(1, 0, 2), nfp, X0, N, B)
_, X1s = fourier.fft2d(x1.T)
X2c, _ = fourier.fft2d(x2.T)
if max(np.abs(X1s).max(), np.abs(X2c).max()) > 1e-10:
    raise ValueError("The cut boundary is not stellarator symmetric.")
M, Nn = minimal_modes(x1.T, x2.T, tolerance=args.tolerance)
print(f"Boundary modes (m, n) <= ({M}, {Nn})")

name = output.name
write_Gframe_ncfile(
    f"{name}-Gframe.nc",
    {
        "nfp": nfp,
        "axis": {"nzeta": nz, "zetafull": zeta, "xyz": X0.T, "Nxyz": N.T, "Bxyz": B.T},
        "boundary": {
            "ntheta": nt,
            "nzeta": nz,
            "theta": theta,
            "zeta": zeta[:nz],
            "lasym": False,
            "m_max": M,
            "n_max": Nn,
            "X1": x1.T,
            "X2": x2.T,
        },
    },
)

source_parameters = read_parameters(next(source.glob("parameter*.toml")))
parameters = {
    "ProjectName": name,
    "which_hmap": 21,
    "hmap_ncfile": f"{name}-Gframe.nc",
    "getBoundaryFromFile": 1,
    "boundary_filename": f"{name}-Gframe.nc",
    "X1X2_deg": 5,
    "LA_deg": 5,
    "X1_mn_max": (M, Nn),
    "X2_mn_max": (M, Nn),
    "LA_mn_max": (M, Nn),
    "X1_sin_cos": "_cos_",
    "X2_sin_cos": "_sin_",
    "LA_sin_cos": "_sin_",
    # The frame origin is the magnetic axis.
    "init_average_axis": False,
    "X1_a_cos": {(0, 0): 0.0},
    "X2_a_sin": {(0, 1): 0.0},
}
for key in ("phiedge", "iota", "pres", "sgrid", "minimize_tol", "totalIter", "logIter"):
    parameters[key] = source_parameters[key]

run = gvec.run(parameters, runpath=output)
print(f"Final state: {run.state}")
