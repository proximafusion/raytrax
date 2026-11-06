---
icon: lucide/triangle-alert
---

# Limitations

The following is a list of current limitations of Raytrax.

## Input formats

- To load MHD equilibria, only [VMEC++](https://proximafusion.github.io/vmecpp/) is supported so far. Note that VMEC++ is interoperable with SIMSOPT and can read legacy VMEC output files as well.

## Beam representation

- The beam is currently represented as a single ray, which is a good approximation for narrow beams with small divergence. However, for wider beams or those with significant divergence, a more accurate representation would be to use a bundle of rays or a Gaussian beam model, as implemented e.g. in TRAVIS. This is planned for future releases.

## Physics

- For tracing, only the cold plasma dielectric tensor is implemented so far.
- The power flux in the denominator of the absorption coefficient is evaluated with the cold dielectric tensor. This is least accurate at the fundamental harmonic, in particular for X mode.
- Electron cyclotron emission (ECE) is not yet implemented.
- Electron cyclotron current drive (ECCD) is not yet implemented.

## Ray tracing

- O-X mode conversion is not modelled. At $n_\parallel^2 = Y/(1+Y)$ the O- and X-mode roots of the dispersion relation touch at the cutoff $X = 1$, and the Hamiltonian flow has a saddle there: a ray aimed exactly at this point needs infinite Hamiltonian time, and neighbouring rays separate exponentially. Close to it the O mode is evanescent only in a thin layer between its cutoff $X = 1$ and the Z-mode cutoff $X = (1+Y)(1-n_\parallel^2)$. The root $N_\perp^2$ is continuous across this layer, so a step that straddles it continues on the X-mode sheet with $\mathcal H$ unchanged, and the step-size control cannot detect this; whether it happens depends on where the steps fall. The wave does the same: the transmission through such a sub-wavelength gap is close to one.
- The root $N_\perp^2(n_\parallel^2, X, Y)$ has a branch point at $X_{\rm fold} = 1 + Y^2 (1-n_\parallel^2)^2 / (4 n_\parallel^2)$, beyond the O-mode cutoff. Only the X mode at $Y > 1$ gets there, on its way from $X = 1$ to the L cutoff $X = 1 + Y$; such a ray fails with NaN at the fold. ECRH rays do not go there.
- There is no termination at the upper-hybrid resonance $X = 1 - Y^2$, where $N_\perp^2 \to \infty$ for the X mode. A ray that reaches it, e.g. an X-mode ray launched from the high-field side across the cyclotron layer, is integrated through it with a frequency mismatch of order one and is not flagged.

## Validation

- A comparison with TRAVIS for W7-X will be presented in a forthcoming publication.
- Comparisons with experimental data have not been carried out.