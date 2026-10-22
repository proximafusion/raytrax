---
icon: lucide/triangle-alert
---

# Limitations

The following is a list of current limitations of Raytrax.

## Input formats

- MHD equilibria can be loaded from [VMEC++](https://proximafusion.github.io/vmecpp/) and [GVEC](https://gvec.readthedocs.io). Note that VMEC++ is interoperable with SIMSOPT and can read legacy VMEC output files as well.
- Only stellarator-symmetric equilibria are supported.
- GVEC equilibria must be periodic about the $Z$-axis, and every half-plane $\phi = \text{const}$ must cut each flux surface in a single closed curve. This excludes figure-8 and knotted geometries.

## Beam representation

- The beam is currently represented as a single ray, which is a good approximation for narrow beams with small divergence. However, for wider beams or those with significant divergence, a more accurate representation would be to use a bundle of rays or a Gaussian beam model, as implemented e.g. in TRAVIS. This is planned for future releases.

## Physics

- For tracing, only the cold plasma dielectric tensor is implemented so far.
- The power flux in the denominator of the absorption coefficient is evaluated with the cold dielectric tensor. This is least accurate at the fundamental harmonic, in particular for X mode.
- Electron cyclotron emission (ECE) is not yet implemented.
- Electron cyclotron current drive (ECCD) is not yet implemented.

## Validation

- A comparison with TRAVIS for W7-X will be presented in a forthcoming publication.
- Comparisons with experimental data have not been carried out.