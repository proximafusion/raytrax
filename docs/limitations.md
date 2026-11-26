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
- The cold Hamiltonian vanishes on both roots of the dispersion relation and tells the O and the X mode apart only while $|\mathcal H|$ on the ray is small against the separation of the roots, $X Y \sqrt g / |\tilde S|$ with $g = Y^2 (1-n_\parallel^2)^2 + 4 n_\parallel^2 (1-X)$, which vanishes at the plasma edge. A ray that carries $\mathcal H = h \neq 0$ can be reflected in a thin layer at $X \approx |h|\,|\tilde S| / (Y \sqrt{g_0})$, for one sign of $h$ and $n_\parallel \neq 0$. $\mathcal H$ is conserved, so $h$ is set at the launch: the beam has to start on the dispersion surface, that is in vacuum with $n^2 = 1$; `trace` scales `Beam.direction` to a unit vector for this. `TraceResult.launched_on_hamiltonian` reports whether $|h| \le 10^{-6}$, and `trace` warns if it is not. With unit launch vectors no such reflection was seen in 5000 random rays, although the integration error of the default settings reaches $|h| \sim 10^{-4}$ where the ray leaves the plasma: the layer is micrometres thick and the steps do not resolve it.
- The ray is not followed through the upper-hybrid resonance $X = 1 - Y^2$, where $N_\perp^2 \to \infty$ for the X mode. A ray that runs into it, e.g. an X-mode ray launched from the high-field side across the cyclotron layer, is stopped where $N_\perp$ reaches `TracerSettings.max_refractive_index_perp` (default 10), which for perpendicular propagation is where $X + Y^2 - 1 = X (1 - X) / 100$, and `TraceResult.stopped_at_resonance` and `TraceResult.incomplete` are set. In a warm plasma the wave is converted there to the electron Bernstein wave, which the cold dispersion relation does not describe, so the power the ray still carries is neither followed nor counted as absorbed.

## Validation

- A comparison with TRAVIS for W7-X will be presented in a forthcoming publication.
- Comparisons with experimental data have not been carried out.