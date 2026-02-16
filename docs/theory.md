---
icon: lucide/orbit
---

Raytrax solves the ray equations of microwave rays employed for electron cyclotron resonance heating of magnetic confinement fusion plasmas. With peak magnetic flux densities of several Teslas, electron cyclotron resonance frequencies are in the range of hundreds of GHz and corresponding wave lengths of the order of millimeters. As the length scales of variation of density and temperature are typically much larger, the use of the **geometrical optics** (GO) approximation (sometimes called WKB approximation in analogy with quantum mechanics) is justified.


## Ray Representation

In the GO approximation, the monochromatic wave field of a single ray is represented as

$$\boldsymbol{E}(\boldsymbol{r}, t) = \boldsymbol{E}_0(\boldsymbol{r}) e^{i \left(S(\boldsymbol{r}) - \omega t\right)}$$

where $\boldsymbol{E}_0(\boldsymbol{r})$ is the slowly varying amplitude and $S(\boldsymbol{r})$ is the rapidly varying eikonal (scalar phase function). Here and in the following, boldface symbols denote vector or tensor quantities, and we use SI units.

The most relevant quantities describing wave propagation in the plasma are:

- **$\boldsymbol{r}$**: Position vector along the ray trajectory
- **$\boldsymbol{k}(\boldsymbol{r})$**: Local complex wave vector, defined as $\boldsymbol{k} = \nabla S$, which is tangential to the ray direction $\hat{\boldsymbol{s}}$, $\boldsymbol{k} = k_r \hat{\boldsymbol{s}} + i k_i \hat{\boldsymbol{s}}$
- **$\boldsymbol{n}(\boldsymbol{r})$**: Vectorial index of refraction, defined as $\boldsymbol{n} = (c/\omega) \boldsymbol{k}$ with magnitude $n = |\boldsymbol{n}|$
- **$\alpha$**: Absorption coefficient, given by $\alpha = 2\,\text{Im}(\boldsymbol{k} \cdot \hat{\boldsymbol{s}}) = 2 k_i$, leading to exponential decay $e^{-\alpha s}$ of the wave amplitude along the ray

## Dispersion Relation

The plasma response to the electromagnetic wave is described by the complex dielectric tensor $\boldsymbol{\varepsilon}(\boldsymbol{r}, \omega)$, which depends on the local magnetic field $\boldsymbol{B}$, electron density $n_e$, and temperature $T$. From Maxwell's equations, the wave equation in the frequency domain is:

$$\nabla \times (\nabla \times \boldsymbol{E}) + \frac{\omega^2}{c^2} \boldsymbol{\varepsilon} \cdot \boldsymbol{E} = 0$$

Substituting the eikonal ansatz in the locally homogeneous approximation and retaining leading-order terms yields an eigenvalue problem for the wave field:

$$\boldsymbol{D} \cdot \boldsymbol{E}_0 = 0$$

Here, $\boldsymbol{D}$ is the dispersion tensor:

$$\boldsymbol{D} = \boldsymbol{\varepsilon} - n^2 \boldsymbol{I} + \boldsymbol{n}\boldsymbol{n}$$

where $\boldsymbol{I}$ is the identity tensor and $\boldsymbol{n}\boldsymbol{n}$ is the dyadic product. For a nontrivial wave field to exist, the determinant must vanish:

$$\det \boldsymbol{D} = 0$$

This dispersion relation connects the wave vector $\boldsymbol{k}$ to the frequency $\omega$ and the local plasma parameters through $\boldsymbol{\varepsilon}(\boldsymbol{B}, n_e, T)$. See [ray tracing](theory/ray-tracing.md) for how this dispersion relation is used to derive the ray equations.

## Raytrax Inputs and Outputs

The main outputs of raytrax are the **ray trajectory** $\boldsymbol{r}(s)$, where $s$ is the arc length along the ray, and the **linear deposition power density** $dP/ds$, which can be computed from the absorption coefficient $\alpha$.

The inputs are the **magnetic field** $\boldsymbol{B}(\boldsymbol{r})$, **electron density** $n_e(\boldsymbol{r})$, and **temperature** $T(\boldsymbol{r})$, which together determine the dielectric tensor $\boldsymbol{\varepsilon}$.

Additionally, the initial conditions for the ray (initial position $\boldsymbol{r}_0$ and direction $\hat{\boldsymbol{s}}_0$) and the wave frequency $\omega$ are required.